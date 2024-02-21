import sys
from time import perf_counter as pc
from datetime import datetime
import ray
from ray import train
from torch.utils.tensorboard import SummaryWriter
from ray.tune.logger import pretty_print
import ray.rllib.algorithms.sac as sac
from ray.rllib.utils.metrics import NUM_ENV_STEPS_SAMPLED
from ray.rllib.models import ModelCatalog

from stop_simple import StopSimple
from highway_env_wrapper import HighwayEnvWrapper
from cda_callbacks import CdaCallbacks
from bridgit_nn import BridgitNN

"""This program trains the agent with the given environment, using the specified hyperparameters."""

def main(argv):

    SINGLE_WORKER = False #set True for debugging, but use False for normal production training

    # Identify our custom NN model
    ModelCatalog.register_custom_model("bridgit_policy_model", BridgitNN)
    print("///// ModelCatalog registered.")

    # Initialize per https://docs.ray.io/en/latest/workflows/management.html?highlight=local%20storage#storage-configuration
    # Can use arg num_cpus = 1 to force single-threading for debugging purposes, then comment out the cfg.resources and
    # cfg.rollouts sections below.
    t = datetime.now()
    DATA_PATH = "/home/starkj/ray_results/cda1/{:4d}{:02d}{:02d}-{:02d}{:02d}".format(t.year, t.month, t.day, t.hour, t.minute)
    if SINGLE_WORKER:
        ray.init(storage = DATA_PATH, num_cpus = 1, num_gpus = 1)
    else:
        ray.init(storage = DATA_PATH)

    # Define which learning algorithm we will use and set up is default config params
    algo = "SAC"
    cfg = sac.SACConfig()
    cfg.framework("torch")
    cfg_dict = cfg.to_dict()

    # Define training control
    status_int          = 200    #num iters between status logs
    chkpt_int           = 1000    #num iters between storing new checkpoints
    max_iterations      = 10000

    # Define the custom environment for Ray
    env_config = {}
    env_config["time_step_size"]                = 0.2
    env_config["episode_length"]                = 100 #80 steps gives roughly 470 m of travel @29 m/s
    env_config["debug"]                         = 0
    env_config["valid_targets"]                 = "all"
    env_config["randomize_targets"]             = True
    env_config["crash_report"]                  = False
    env_config["vehicle_file"]                  = "/home/starkj/projects/cda1/config/vehicle_config_multi.yaml"
    env_config["verify_obs"]                    = False
    env_config["training"]                      = True
    env_config["ignore_neighbor_crashes"]       = True  #if true, a crash between two neighbor vehicles won't stop the episode
    env_config["scenario"]                      = 0
    cfg.environment(env = HighwayEnvWrapper, env_config = env_config)
    #cfg.environment(env = DummyEnv, env_config = env_config)

    # Add exploration noise params
    #cfg.rl_module(_enable_rl_module_api = False) #disables the RL module API, which allows exploration config to be defined for ray 2.6

    explore_config = cfg_dict["exploration_config"]
    #print("///// Explore config:\n", pretty_print(explore_config))
    explore_config["type"]                      = "GaussianNoise" #default OrnsteinUhlenbeckNoise doesn't work well here
    explore_config["stddev"]                    = 0.1 #0.25 #this param is specific to GaussianNoise
    explore_config["random_timesteps"]          = 0 #100_000 #provides random experiences to pre-populate the experience buffer
    explore_config["initial_scale"]             = 1.0
    explore_config["final_scale"]               = 0.01 #0.1
    explore_config["scale_timesteps"]           = 1_000_000 #50_000_000
    exp_switch                                  = True
    cfg.exploration(explore = exp_switch, exploration_config = explore_config)
    #cfg.exploration(explore = False)

    """ TODO: turn this on if we go back to Tuner
    cfg.evaluation( evaluation_duration         = 10,   #num episodes
                    evaluation_duration_unit    = "episodes",
                    evaluation_interval         = 100,  #num iterations between evals
                    evaluation_parallel_to_training = True,
                    evaluation_num_workers      = 2,
    )
    """

    # Computing resources - Ray allocates 1 cpu per rollout worker and one cpu per env (2 cpus) per trial.
    # The number of workers does not equal the number of trials.
    # NOTE: if num_gpus = 0 then policies will always be built/evaluated on cpu, even if gpus are specified for workers;
    #       to get workers (only) to use gpu, num_gpus needs to be positive (e.g. 0.0001).
    # NOTE: local worker needs to do work for every trial, so needs to divide its time among simultaneous trials. Therefore,
    #       if gpu is to be used for local workder only, then the number of gpus available need to be divided among the
    #       number of possible simultaneous trials (as well as gpu memory).

    if not SINGLE_WORKER:
        cfg.resources(  num_gpus                    = 0.5, #for the local worker, which does the learning & evaluation runs
                        num_cpus_for_local_worker   = 4,
                        num_cpus_per_worker         = 2,   #also applies to the evaluation workers
                        num_gpus_per_worker         = 0.1, #this has to allow gpu left over for local worker & evaluation workers also
        )

        cfg.rollouts(   num_rollout_workers         = 4, #num remote workers _per trial_ (remember that there is a local worker also)
                                                        # 0 forces rollouts to be done by local worker
                        num_envs_per_worker         = 8,
                        rollout_fragment_length     = 8, #timesteps pulled from a sampler
                        #batch_mode                  = "complete_episodes",
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
    LR                                          = 1e-5
    opt_config = cfg_dict["optimization"]
    opt_config["actor_learning_rate"]           = LR
    opt_config["critic_learning_rate"]          = LR
    opt_config["entropy_learning_rate"]         = LR

    policy_config = cfg_dict["policy_model_config"]
    policy_config["custom_model"]               = "bridgit_policy_model"

    q_config = cfg_dict["q_model_config"]
    q_config["fcnet_hiddens"]                   = [1024, 256, 128]
    q_config["fcnet_activation"]                = "relu"

    replay_config = cfg_dict["replay_buffer_config"]
    replay_config["type"]                       = "MultiAgentPrioritizedReplayBuffer"
    replay_config["capacity"]                   = 1_000_000 #1M seems to be the max allowable
    replay_config["prioritized_replay"]         = True

    cfg.training(   twin_q                      = True,
                    gamma                       = 0.995,
                    train_batch_size            = 256, #must be an int multiple of rollout_fragment_length * num_rollout_workers * num_envs_per_worker
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
    if len(argv) > 1  and  argv[1] is not None  and  len(argv[1]) > 0:
        try:
            algo.restore(argv[1])
            starting_step_count = algo._counters[NUM_ENV_STEPS_SAMPLED]
            print("///// Successfully restored baseline checkpoint {} with {} steps already trained".format(argv[1], starting_step_count))
        except Exception as e:
            print("\n///// ERROR restoring checkpoint {}.\n{}\n".format(argv[1], e))
            sys.exit(1)

    # Run the training loop
    print("///// Training loop beginning.  Checkpoints stored every {} iters in {}".format(chkpt_int, DATA_PATH))
    print("/////   using vehicle config file ", env_config["vehicle_file"])
    tensorboard = SummaryWriter(DATA_PATH)
    result = None
    start_time = pc()
    for iter in range(1, max_iterations+1):
        result = algo.train()
        #print("///// train step result object is:")
        #print(pretty_print(result))
        #if iter == 1:
        #    print("Sample of results from train() call:\n", pretty_print(result))

        # Write data to Tensorboard
        #train.report(result)
        rmin = result["episode_reward_min"]
        rmean = result["episode_reward_mean"]
        rmax = result["episode_reward_max"]
        #ermean = result["evaluation_reward_mean"]
        """
        tensorboard.add_scalar("num_agent_steps_trained", result["num_agent_steps_trained"])
        tensorboard.add_scalar("timesteps_total", result["timesteps_total"])
        tensorboard.add_scalar("episode_reward_mean", rmean)
        tensorboard.add_scalar("episode_reward_min", rmin)
        tensorboard.add_scalar("episode_reward_max", rmax)
        """

        # use RLModule.save_to_checkpoint(<dir>) to save a checkpoint
        if iter % chkpt_int == 0:
            path = "{}/{:05d}".format(DATA_PATH, iter)
            ckpt_res = algo.save(checkpoint_dir = path)
            #print("///// Checkpoint saved in {}".format(ckpt_res.checkpoint.path))
            #print(pretty_print(ckpt_res.metrics))

        if iter % status_int == 0:
            elapsed_sec = pc() - start_time
            elapsed_hr = elapsed_sec / 3600.0
            perf = int(iter/elapsed_hr)
            steps = result["num_env_steps_sampled"] - starting_step_count
            ksteps_per_hr = 0
            if elapsed_hr > 0.01:
                ksteps_per_hr = 0.001*steps/elapsed_hr
            remaining_hrs = 0.0
            if iter > 1:
                remaining_hrs = (max_iterations - iter) / perf
            print("///// Iter {} ({} steps): Rew {:5.1f} /{:5.1f} /{:5.1f}.  Ep len = {:.1f}.  "
                  .format(iter, steps, rmin, rmean, rmax, result["episode_len_mean"]), \
                  "Elapsed = {:.1f} hr @{:d} iter/hr, {:.1f} M steps/hr. Rem hr: {:.1f}".format(elapsed_hr, perf, 0.001*ksteps_per_hr, remaining_hrs))

    print("\n///// Training completed.  Final iteration results:\n")
    print(pretty_print(result))
    tensorboard.flush()
    algo.stop()
    ray.shutdown()


######################################################################################################
######################################################################################################

if __name__ == "__main__":
   main(sys.argv)
