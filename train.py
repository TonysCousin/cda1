import sys
from time import perf_counter as pc
import ray
import torch
from torch.utils.tensorboard import SummaryWriter
from ray.tune.logger import pretty_print
import ray.rllib.algorithms.sac as sac

from stop_simple import StopSimple
from highway_env_wrapper import HighwayEnvWrapper
from cda_callbacks import CdaCallbacks

"""This program trains the agent with the given environment, using the specified hyperparameters.
"""

# Identify a baseline checkpoint from which to continue training
_checkpoint_path = None


def main(argv):

    # Initialize per https://docs.ray.io/en/latest/workflows/management.html?highlight=local%20storage#storage-configuration
    # Can use arg num_cpus = 1 to force single-threading for debugging purposes (along with setting num_gpus = 0)
    DATA_PATH = "/home/starkj/ray_results/cda1"
    ray.init(storage = DATA_PATH) #CAUTION! storage is an experimental arg (in Ray 2.5.1), and intended to be a URL for cluster-wide access

    # Define which learning algorithm we will use and set up is default config params
    algo = "SAC"
    cfg = sac.SACConfig()
    cfg.framework("torch")
    cfg_dict = cfg.to_dict()

    # Define the stopper object that decides when to terminate training.
    status_int          = 500    #num iters between status logs
    chkpt_int           = 500    #num iters between storing new checkpoints
    max_iterations      = 80000

    # Define the custom environment for Ray
    env_config = {}
    env_config["time_step_size"]                = 0.2
    env_config["episode_length"]                = 80 #80 steps gives roughly 470 m of travel @29 m/s
    env_config["debug"]                         = 0
    env_config["crash_report"]                  = False
    env_config["vehicle_file"]                  = "/home/starkj/projects/cda1/vehicle_config.yaml"
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
    explore_config["stddev"]                    = 0.25 #tune.uniform(0.2, 0.6) #this param is specific to GaussianNoise
    explore_config["random_timesteps"]          = 10000 #tune.qrandint(0, 20000, 50000) #was 20000
    explore_config["initial_scale"]             = 1.0
    explore_config["final_scale"]               = 0.1 #tune.choice([1.0, 0.01])
    explore_config["scale_timesteps"]           = 12000000 #tune.choice([12000000, 8000000])
    exp_switch                                  = True #tune.choice([False, True, True]) #should the algo use exploration?
    cfg.exploration(explore = exp_switch, exploration_config = explore_config)
    #cfg.exploration(explore = False)

    # Computing resources - Ray allocates 1 cpu per rollout worker and one cpu per env (2 cpus) per trial.
    # The number of workers does not equal the number of trials.
    # NOTE: if num_gpus = 0 then policies will always be built/evaluated on cpu, even if gpus are specified for workers;
    #       to get workers (only) to use gpu, num_gpus needs to be positive (e.g. 0.0001).
    # NOTE: local worker needs to do work for every trial, so needs to divide its time among simultaneous trials. Therefore,
    #       if gpu is to be used for local workder only, then the number of gpus available need to be divided among the
    #       number of possible simultaneous trials (as well as gpu memory).

    cfg.resources(  num_gpus                    = 1, #for the local worker, which does the learning & evaluation runs
                    num_cpus_for_local_worker   = 2,
                    num_cpus_per_worker         = 2,  #also applies to the evaluation workers
                    num_gpus_per_worker         = 0,  #this has to allow gpu left over for local worker & evaluation workers also
    )

    cfg.rollouts(   num_rollout_workers         = 2, #num remote workers _per trial_ (remember that there is a local worker also)
                                                     # 0 forces rollouts to be done by local worker
                    num_envs_per_worker         = 1,
                    rollout_fragment_length     = 80, #timesteps pulled from a sampler
                    batch_mode                  = "complete_episodes",
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
    opt_config["actor_learning_rate"]           = 5e-5
    opt_config["critic_learning_rate"]          = 5e-5
    opt_config["entropy_learning_rate"]         = 5e-5

    policy_config = cfg_dict["policy_model_config"]
    policy_config["fcnet_hiddens"]              = [600, 256, 128]
    policy_config["fcnet_activation"]           = "relu"

    q_config = cfg_dict["q_model_config"]
    q_config["fcnet_hiddens"]                   = [600, 256, 128]
    q_config["fcnet_activation"]                = "relu"

    replay_config = cfg_dict["replay_buffer_config"]
    replay_config["type"]                       = "MultiAgentPrioritizedReplayBuffer"
    replay_config["capacity"]                   = 1000000
    replay_config["prioritized_replay"]         = True

    cfg.training(   twin_q                      = True,
                    gamma                       = 0.995,
                    train_batch_size            = 1040, #must be an int multiple of rollout_fragment_length * num_rollout_workers * num_envs_per_worker
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

    print("\n///// {} training params are:\n".format(algo))
    print(pretty_print(cfg.to_dict()))

    algo = cfg.build()

    # Run the training loop
    print("///// Training loop beginning.  Checkpoints stored every {} iters in {}".format(chkpt_int, DATA_PATH))
    tensorboard = SummaryWriter(DATA_PATH)
    result = None
    start_time = pc()
    for iter in range(max_iterations):
        result = algo.train()

        # Write data to Tensorboard
        rmin = result["episode_reward_min"]
        rmean = result["episode_reward_mean"]
        rmax = result["episode_reward_max"]
        tensorboard.add_scalar("episode_reward_mean", rmean)
        tensorboard.add_scalar("episode_reward_min", rmin)
        tensorboard.add_scalar("episode_reward_max", rmax)

        # use RLModule.save_to_checkpoint(<dir>) to save a checkpoint
        if iter % chkpt_int == 0:
            cp = algo.save(checkpoint_dir = DATA_PATH)

        if iter % status_int == 0:
            elapsed_sec = pc() - start_time
            elapsed_hr = elapsed_sec / 3600.0
            perf = int(iter/elapsed_hr)
            print("///// Iter {}: Rewards = {:7.3f} / {:7.3f} / {:7.3f}.   Elapsed = {:.2f} hr.   Perf = {:d} iter/hr"
                  .format(iter, rmin, rmean, rmax, elapsed_hr, perf))



    print("\n///// Training completed.  Final iteration results:\n")
    print(pretty_print(result))
    tensorboard.flush()
    ray.shutdown()


    """ for reference:
    run_config = RunConfig( #some commented-out items will allegedly be needed for Ray 2.6
                    name                        = "cda1",
                    local_dir                   = "~/ray_results", #for ray <= 2.5
                    #storage_path                = "~/ray_results", #required if not using remote storage for ray 2.6
                    stop                        = stopper,
                    sync_config                 = tune.SyncConfig(syncer = None), #for single-node or shared checkpoint dir, ray 2.5
                    #sync_config                 = tune.SyncConfig(syncer = None, upload_dir = None), #for single-node or shared checkpoint dir, ray 2.6
                    checkpoint_config           = air.CheckpointConfig(
                                                    checkpoint_frequency        = chkpt_int,
                                                    checkpoint_score_attribute  = "episode_reward_mean",
                                                    num_to_keep                 = 2, #if > 1 hard to tell which one is the best
                                                    checkpoint_at_end           = False
                    )
                )

    # Execute the HP tuning job, beginning with a previous checkpoint, if one was specified in the CdaCallbacks.
    if len(argv) > 1:
        tuner = Tuner.restore(path = argv[1], trainable = algo, resume_errored = True, param_space = cfg.to_dict())
        print("\n///// Tuner created to continue checkpoint ", argv[1])
    else:
        tuner = Tuner(algo, param_space = cfg.to_dict(), tune_config = tune_config, run_config = run_config)
        print("\n///// New tuner created.\n")
    """


######################################################################################################
######################################################################################################

if __name__ == "__main__":
   main(sys.argv)
