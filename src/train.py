import sys
import os
from time import perf_counter as pc
import tempfile
import ray
import torch
from torch.utils.tensorboard import SummaryWriter
from ray.tune.logger import pretty_print
import ray.rllib.algorithms.sac as sac
from ray.rllib.utils.metrics import NUM_ENV_STEPS_SAMPLED
from ray.rllib.models import ModelCatalog
import ray.train.torch
from ray import train
from ray.train import RunConfig, CheckpointConfig, Checkpoint, ScalingConfig
from ray.train.torch import TorchTrainer

from highway_env_wrapper import HighwayEnvWrapper
from cda_callbacks import CdaCallbacks
from bridgit_nn import BridgitNN

"""This program trains the agent with the given environment, using the specified hyperparameters."""


def train_fn(config):
    """Executes the training loop & collects performance metrics."""

    # Identify our custom NN model
    ModelCatalog.register_custom_model("bridgit_policy_model", BridgitNN)

    # Define which learning algorithm we will use and set up is default config params
    algo = "SAC"
    cfg = sac.SACConfig()
    cfg.framework("torch")
    cfg_dict = cfg.to_dict()

    # Define the custom environment for Ray
    env_config = {}
    env_config["time_step_size"]                = 0.2
    env_config["episode_length"]                = 100 #80 steps gives roughly 470 m of travel @29 m/s
    env_config["debug"]                         = 0
    env_config["crash_report"]                  = False
    env_config["vehicle_file"]                  = "/home/starkj/projects/cda1/config/vehicle_config_ego_training.yaml"
    env_config["verify_obs"]                    = False
    env_config["training"]                      = True
    env_config["ignore_neighbor_crashes"]       = True  #if true, a crash between two neighbor vehicles won't stop the episode
    env_config["scenario"]                      = 0
    cfg.environment(env = HighwayEnvWrapper, env_config = env_config)
    #cfg.environment(env = DummyEnv, env_config = env_config)

    # Add exploration noise params
    #cfg.rl_module(_enable_rl_module_api = False) #disables the RL module API, which allows exploration config to be defined for ray 2.6

    explore_config = cfg_dict["exploration_config"]
    explore_config["type"]                      = "GaussianNoise" #default OrnsteinUhlenbeckNoise doesn't work well here
    explore_config["stddev"]                    = 0.25 #this param is specific to GaussianNoise
    explore_config["random_timesteps"]          = 1000000
    explore_config["initial_scale"]             = 1.0
    explore_config["final_scale"]               = 0.1
    explore_config["scale_timesteps"]           = 80000000
    exp_switch                                  = True
    cfg.exploration(explore = exp_switch, exploration_config = explore_config)
    #cfg.exploration(explore = False)

    # Computing resources - Ray allocates 1 cpu per rollout worker and one cpu per env (2 cpus) per trial.
    # The number of workers does not equal the number of trials.
    # NOTE: if num_gpus = 0 then policies will always be built/evaluated on cpu, even if gpus are specified for workers;
    #       to get workers (only) to use gpu, num_gpus needs to be positive (e.g. 0.0001).
    # NOTE: local worker needs to do work for every trial, so needs to divide its time among simultaneous trials. Therefore,
    #       if gpu is to be used for local workder only, then the number of gpus available need to be divided among the
    #       number of possible simultaneous trials (as well as gpu memory).

    cfg.resources(  num_gpus                    = 0.2, #for the local worker, which does the learning & evaluation runs
                    num_cpus_for_local_worker   = 4,
                    num_cpus_per_worker         = 2,   #also applies to the evaluation workers
                    num_gpus_per_worker         = 0.2, #this has to allow gpu left over for local worker & evaluation workers also
    )

    cfg.rollouts(   #num_rollout_workers         = config["num_workers"], #num remote workers _per trial_ (remember that there is a local worker also)
                                                     # 0 forces rollouts to be done by local worker
                    num_envs_per_worker         = 8,
                    rollout_fragment_length     = 32, #timesteps pulled from a sampler
    )

    cfg.fault_tolerance(
                    recreate_failed_workers     = True,
    )

    # Debugging assistance
    cfg.debugging(  log_level                   = "WARN",
                    seed                        = 17,
    )

    # Set up custom callbacks for RLlib to use
    cfg.callbacks(  CdaCallbacks)

    # Checkpoint behavior
    cfg.checkpointing(export_native_model_files = True)

    # ===== Training algorithm HPs for SAC ==================================================
    opt_config = cfg_dict["optimization"]
    opt_config["actor_learning_rate"]           = 3e-5
    opt_config["critic_learning_rate"]          = 3e-5
    opt_config["entropy_learning_rate"]         = 5e-5

    policy_config = cfg_dict["policy_model_config"]
    policy_config["custom_model"]               = "bridgit_policy_model"

    q_config = cfg_dict["q_model_config"]
    q_config["fcnet_hiddens"]                   = [1024, 256, 128]
    q_config["fcnet_activation"]                = "relu"

    replay_config = cfg_dict["replay_buffer_config"]
    replay_config["type"]                       = "MultiAgentPrioritizedReplayBuffer"
    replay_config["capacity"]                   = 1000000 #1M seems to be the max allowable
    replay_config["prioritized_replay"]         = True

    cfg.training(   twin_q                      = True,
                    gamma                       = 0.995,
                    train_batch_size            = 1024, #must be an int multiple of rollout_fragment_length * num_rollout_workers * num_envs_per_worker
                    initial_alpha               = 0.2, #tune.choice([0.002, 0.2]),
                    tau                         = 0.005,
                    n_step                      = 1, #tune.choice([1, 2, 3]),
                    grad_clip                   = 1.0, #tune.uniform(0.5, 1.0),
                    optimization_config         = opt_config,
                    policy_model_config         = policy_config,
                    q_model_config              = q_config,
                    replay_buffer_config        = replay_config,
    )

    # ===== Final setup =========================================================================

    #print("\n///// {} training params are:\n".format(algo))
    #print(pretty_print(cfg.to_dict()))

    # Set up starting counters to handle possible checkpoint start
    starting_step_count = 0

    # Build the algorithm object, and load the starting checkpoint if one was specified
    algo = cfg.build()
    bc = config["base_checkpoint"]
    if bc is not None:
        try:
            algo.restore(bc)
            starting_step_count = algo._counters[NUM_ENV_STEPS_SAMPLED]
            print("///// Successfully restored baseline checkpoint {} with {} steps already trained".format(bc, starting_step_count))
        except Exception as e:
            print("\n///// ERROR restoring checkpoint {}.\n{}\n".format(bc, e))
            sys.exit(1)

    # Training iterations
    result = None
    start_time = pc()
    for iter in range(1, config["max_iterations"]+1):
        result = algo.train()

        # Prep data for Tensorboard
        #TODO: add mean episode length
        rmin = result["episode_reward_min"]
        rmean = result["episode_reward_mean"]
        rmax = result["episode_reward_max"]
        metrics = {"episode_reward_min":    rmin,
                   "episode_reward_mean":   rmean,
                   "episode_reward_max":    rmax,
                  }

        # Handle periodic checkpoints & status reporting
        with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
            checkpoint = None

            # use RLModule.save_to_checkpoint(<dir>) to save a checkpoint
            if iter % config["chkpt_int"] == 0:
                #algo.save(checkpoint_dir = DATA_PATH)
                policy = algo.get_policy()
                model = policy.get_weights()
                torch.save(model, os.path.join(temp_checkpoint_dir, "model-{:05d}.pt".format(iter)))
                checkpoint = Checkpoint.from_directory(temp_checkpoint_dir)
                train.report(metrics, checkpoint = checkpoint)

            if iter % config["status_int"] == 0:
                elapsed_sec = pc() - start_time
                elapsed_hr = elapsed_sec / 3600.0
                perf = int(iter/elapsed_hr)
                steps = result["num_env_steps_sampled"] - starting_step_count
                ksteps_per_hr = 0
                if elapsed_hr > 0.01:
                    ksteps_per_hr = int(0.001*steps/elapsed_hr)
                remaining_hrs = 0.0
                if iter > 1:
                    remaining_hrs = (config["max_iterations"] - iter) / perf
                print("///// Iter {} ({} steps): Rew {:7.3f} / {:7.3f} / {:7.3f}.  Ep len = {:.1f}.  "
                    .format(iter, steps, rmin, rmean, rmax, result["episode_len_mean"]), \
                    "Elapsed = {:.2f} hr @{:d} iter/hr, {:d} k steps/hr. Rem hr: {:.1f}".format(elapsed_hr, perf, ksteps_per_hr, remaining_hrs))


def main(argv):

    # Initialize per https://docs.ray.io/en/latest/workflows/management.html?highlight=local%20storage#storage-configuration
    # Can use arg num_cpus = 1 to force single-threading for debugging purposes (along with setting num_gpus = 0)
    DATA_PATH = "/home/starkj/ray_results/cda1"
    ray.init(storage = DATA_PATH) #CAUTION! storage is an experimental arg (in Ray 2.5.1), and intended to be a URL for cluster-wide access
    print("///// Ray initialized.")

    # Define the basic training loop control stuff
    num_workers                                 = 1
    training_loop_config = {}
    #training_loop_config["base_checkpoint"]     = argv[1] if len(argv) > 1  and  argv[1] is not None  and  len(argv[1]) > 0  else  None
    training_loop_config["chkpt_dir"]           = DATA_PATH
    training_loop_config["status_int"]          = 1 #200     #num iters between status logs
    training_loop_config["chkpt_int"]           = 2 #1000    #num iters between storing new checkpoints
    training_loop_config["max_iterations"]      = 6 #30000
    training_loop_config["num_workers"]         = num_workers

    # Set up the run configuration
    run_config = train.RunConfig(storage_path = DATA_PATH,
                                 checkpoint_config = CheckpointConfig(checkpoint_score_attribute    = "episode_reward_mean",
                                                                      #checkpoint_frequency          = training_loop_config["chkpt_int"],
                                                                      num_to_keep                   = 5
                                 ),
    )

    if argv[1] is not None  and  len(argv[1]) > 1:
        trainer = TorchTrainer.restore(argv[1])
        print("///// Restored checkpoint {}".format(argv[1]))

    # Build the trainer object
    else:
        trainer = TorchTrainer(train_fn, train_loop_config = training_loop_config, run_config = run_config,
                               scaling_config = train.ScalingConfig(num_workers = num_workers, use_gpu = True))

    # Run the training loop
    print("///// Training loop beginning.  Checkpoints stored every {} iters in {}".format(training_loop_config["chkpt_int"], DATA_PATH))
    result = trainer.fit()

    print("\n///// Training completed.  Final iteration results:\n")
    print(result)
    #tensorboard.flush()
    ray.shutdown()


######################################################################################################
######################################################################################################

if __name__ == "__main__":
   main(sys.argv)
