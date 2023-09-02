from cmath import inf
import sys
import time
from datetime import datetime
from typing import List
import numpy as np
import ray
import ray.rllib.algorithms.ppo as ppo
import ray.rllib.algorithms.sac as sac
from ray.tune.logger import pretty_print

from highway_env_wrapper import HighwayEnvWrapper
from roadway_b import Roadway
from graphics import Graphics

"""This program runs the selected policy checkpoint for one episode and captures key state variables throughout."""

def main(argv):

    # Handle any args
    num_args = len(argv)
    if num_args == 1  or  num_args > 3:
        print("Usage is: {} [ego checkpoint] [scenario #]".format(argv[0]))
        print("Default scenario is 0 (everything randomized).")
        sys.exit(1)

    checkpoint = None
    scenario = 0
    if num_args > 1:
        if argv[1] != "none":
            checkpoint = argv[1]
        if num_args > 2:
            s = int(argv[2])
            if 0 <= s < 90 + Roadway.NUM_LANES:
                scenario = s
    print("inputs: checkpoint = {}, scenario = {}".format(checkpoint, scenario))

    # Set up the environment
    env_config = {  "time_step_size":           0.2,
                    "debug":                    0,
                    "verify_obs":               True,
                    "scenario":                 scenario, #90-95 run single bot on lane 0-5, respectively; 0 = fully randomized
                    "vehicle_file":             "vehicle_config.yaml",
                    "ignore_neighbor_crashes":  True,
                }
    env = HighwayEnvWrapper(env_config)
    #print("///// Environment configured. Params are:")
    #print(pretty_print(cfg.to_dict()))
    env.reset()
    print("///// inference: env reset complete.")

    # Set up a local copy of all vehicles that have been configured
    vehicles = env.get_vehicle_data()

    # If we are using a checkpointed NN for the ego vehicle, then start up rllib to run it
    if checkpoint is not None:

        # Set up the Ray framework
        ray.init()
        print("///// inference: ray init complete.")
        cfg = sac.SACConfig()
        cfg.framework("torch").exploration(explore = False)
        cfg_dict = cfg.to_dict()
        policy_config = cfg_dict["policy_model_config"]
        policy_config["fcnet_hiddens"]              = [256, 256]
        policy_config["fcnet_activation"]           = "relu"
        q_config = cfg_dict["q_model_config"]
        q_config["fcnet_hiddens"]                   = [256, 256]
        q_config["fcnet_activation"]                = "relu"
        cfg.training(policy_model_config = policy_config, q_model_config = q_config)

        cfg.environment(env = HighwayEnvWrapper, env_config = env_config)
        print("///// inference: environment specified.")

        # Restore the selected checkpoint file
        # Note that the raw environment class is passed to the algo, but we are only using the algo to run the NN model,
        # not to run the environment, so any environment info we pass to the algo is irrelevant for this program.
        algo = cfg.build()
        try:
            algo.restore(checkpoint)
            print("///// Checkpoint {} successfully loaded.".format(checkpoint))
        except ValueError as e:
            print("///// Checkpoint {} could not be loaded. {}. Aborting.".format(checkpoint, e))
            sys.exit(2)

   # Set up the graphic display
    graphics = Graphics(env)

    # Run a complete episode
    episode_reward = 0
    done = False
    action_list = [0, 0]
    raw_obs, _ = env.unscaled_reset()
    if checkpoint is None:
        vehicles[0].active = False
    graphics.update(action_list, raw_obs, vehicles)
    obs = env.scale_obs(raw_obs)
    step = 0
    time.sleep(8) #allow viewer to orient and time to turn on video capture
    while not done:
        step += 1

        # Grab ego vehicle actions if it is participating, or use dummies if not
        action = np.zeros(2)
        if vehicles[0].active:
            action = algo.compute_single_action(obs, explore = False)

        # Command masking for first few steps to allow feedback obs to populate
        if step < 2:
            action[1] = 0.0

        # Move the environment forward one time step
        raw_obs, reward, done, truncated, info = env.step(np.ndarray.tolist(action)) #obs returned is UNSCALED
        episode_reward += reward

        # Display current status of all the vehicles
        vehicles = env.get_vehicle_data()
        graphics.update(action, raw_obs, vehicles)

        # Wait for user to indicate okay to begin animation
        """
        if step == 1:
            print("///// Press Down key to begin...")
            go = False
            while not go:
                keys = pygame.key.get_pressed() #this doesn't work
                if keys[pygame.K_DOWN]:
                    go = True
                    break;
        """

        # Scale the observations to be ready for NN ingest next time step
        obs = env.scale_obs(raw_obs)

        print("///// step {:3d}: scaled action = [{:5.2f} {:5.2f}], lane = {}, speed = {:.2f}, p = {:.1f}, r = {:7.4f} {}"
                .format(step, action[0], action[1], vehicles[0].lane_id, vehicles[0].cur_speed, vehicles[0].p, reward, info["reward_detail"]))

        if done:
            print("///// Episode complete: {}. Total reward = {:.2f}".format(info["reason"], episode_reward))
            input("///// Press Enter to close...")
            graphics.close()
            sys.exit()

            # Get user input before closing the window
            for event in pygame.event.get(): #this isn't working
                if event.type == pygame.QUIT:
                    graphics.close()
                    sys.exit()


######################################################################################################
######################################################################################################

if __name__ == "__main__":
   main(sys.argv)
