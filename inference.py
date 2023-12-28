from cmath import inf
import sys
import time
from datetime import datetime
from typing import List
import numpy as np
import argparse
import pygame
import ray
import ray.rllib.algorithms.ppo as ppo
import ray.rllib.algorithms.sac as sac
from ray.tune.logger import pretty_print
from ray.rllib.models import ModelCatalog

from obs_vec import ObsVec
from highway_env_wrapper import HighwayEnvWrapper
from roadway_b import Roadway
from bridgit_nn import BridgitNN
from graphics import Graphics

"""This program runs the selected policy checkpoint for one episode and captures key state variables throughout."""

def main(argv):

    # Handle any args
    program_desc = "Runs a single episode of the cda1 vehicle ML agent in inference mode in its roadway environment with other vehicles."
    scenario_desc = " 0:  (default) everything randomized.\n" \
                    + " 1:  all neighbor vehicles in same lane.\n" \
                    + " 2:  no neighbor vehicles in ego's lane.\n" \
                    + "10-15:  ego starts in lane 0-5, respectively; neighbor vehicles are randomized.\n" \
                    + "20-25:  embedding run where vehicle 0 has Embed guidance but Bridgit model, starting in lanes 0-5 (primarily testing).\n" \
                    + "29:     embedding run where vehicle 0 has Embed guidance but Bridgit model, starting in a random location.\n" \
                    + "90-95:  no ego; a single bot vehicle starts in lane 0-5, respectively, and drives to end of that lane (primarily testing)."
    parser = argparse.ArgumentParser(prog = argv[0], description = program_desc, epilog = "Either -c or -w argument must be used, but never both.")
    parser.add_argument("-c", "--checkpoint", type = str, help = "Ray checkpoint dir containing the model to be run.")
    parser.add_argument("-s", "--scenario", type = int, default = 0, help = scenario_desc)
    parser.add_argument("-L", "--length", type = int, default = inf, help = "Max num time steps to run")
    parser.add_argument("-w", "--weights", type = str, default = None, help = "Filename with policy network weights to control the Bridgit agent.")
    args = parser.parse_args()
    checkpoint = args.checkpoint
    scenario = args.scenario
    episode_len = args.length
    print("***** inputs: scenario = {}, length = {}, checkpoint = {}, weights = {}".format(scenario, episode_len, checkpoint, args.weights))
    if (checkpoint is None  and  args.weights is None)  or  (checkpoint is not None  and  args.weights is not None):
        print("///// ERROR: either checkpoint or weights arg must be specified, but not both.")
        sys.exit(1)

    # Set up the environment
    env_config = {  "time_step_size":           0.2,
                    "debug":                    0,
                    "verify_obs":               True,
                    "scenario":                 scenario, #90-95 run single bot on lane 0-5, respectively; 0 = fully randomized
                    "vehicle_file":             "vehicle_config_ego_training.yaml", #"vehicle_config_embedding.yaml",
                    "ignore_neighbor_crashes":  True,
                    "crash_report":             True,
                }
    env = HighwayEnvWrapper(env_config)
    #print("///// Environment configured. Params are:")
    #print(pretty_print(cfg.to_dict()))
    env.reset()

    # Set up a local copy of all vehicles that have been configured
    vehicles = env.get_vehicle_data()
    #print("///// inference: vehicle[1] = ")
    #vehicles[1].print()

    # If we are using a checkpointed NN for the ego vehicle, then start up rllib to run it
    algo = None
    if checkpoint is not None:

        # Set up the Ray framework
        ModelCatalog.register_custom_model("bridgit_policy_model", BridgitNN)
        ray.init()
        print("///// inference: ray init complete.")
        cfg = sac.SACConfig()
        cfg.framework("torch").exploration(explore = False)
        cfg_dict = cfg.to_dict()
        policy_config = cfg_dict["policy_model_config"]
        policy_config["custom_model"]               = "bridgit_policy_model"
        q_config = cfg_dict["q_model_config"]
        q_config["fcnet_hiddens"]                   = [1024, 256, 128]
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

    # Else, the model weights file was specified, and we don't need Ray at all - just read the file and have PyTorch load it.
    else:
        print("***** Handling model weights is not yet implemented!  Exiting now, until code exists.")
        sys.exit(3) #TODO: remove when implemented.

    # Set up the graphic display & user interaction
    graphics = Graphics(env)
    PAUSE_KEY = pygame.locals.K_SPACE
    RESUME_KEY = pygame.locals.K_SPACE
    END_KEY = pygame.locals.K_ESCAPE

    # Prepare for a complete episode
    episode_reward = 0
    done = False
    action_list = [0, 0]
    raw_obs, _ = env.unscaled_reset()
    if checkpoint is None:
        vehicles[0].active = False

    print("///// inference: ready to update graphics before loop:")
    vehicles[0].print("Ego")

    graphics.update(action_list, raw_obs, vehicles)
    obs = env.scale_obs(raw_obs)
    step = 0

    # Wait for user to indicate okay to begin animation
    key = None
    while key != RESUME_KEY:
        if key == END_KEY:
            print("///// User aborted.")
            graphics.close()
            sys.exit()
        key = graphics.wait_for_key_press()
    print("      Beginning...")

    # Loop on time steps until episode is complete
    while not done  and  step < episode_len:
        step += 1

        # Grab ego vehicle actions if it is participating, or use dummies if not
        action = np.zeros(2)
        if vehicles[0].active:
            try:
                action = algo.compute_single_action(obs, explore = False)
            except:
                print("///// Exception trapped in inference during call to algo.compute_single_action(). obs = ")
                print(obs)

        # Command masking for first few steps to allow feedback obs to populate
        if step < 2:
            action[1] = 0.0

        # Move the environment forward one time step
        raw_obs, reward, done, truncated, info = env.step(np.ndarray.tolist(action)) #obs returned is UNSCALED
        episode_reward += reward

        # Display current status of all the vehicles
        vehicles = env.get_vehicle_data()
        graphics.update(action, raw_obs, vehicles)

        # Scale the observations to be ready for NN ingest next time step
        obs = env.scale_obs(raw_obs)

        print("///// step {:3d}: sc action = [{:5.2f} {:5.2f}], lane = {}, LC # = {}, spd cmd = {:.2f}, spd = {:.2f}, p = {:.1f}, r = {:7.4f} {}"
                .format(step, action[0], action[1], vehicles[0].lane_id, vehicles[0].lane_change_count, \
                        raw_obs[ObsVec.SPEED_CMD], raw_obs[ObsVec.SPEED_CUR], vehicles[0].p, reward, info["reward_detail"]))
        #print("      Vehicle 1 speed = {:.1f}".format(vehicles[1].cur_speed))

        """
        #TODO - this section for debugging only
        # Display the speed limits observed in each sensor zone - loop through longitudinal rows, front to back, then columns, left to right
        print("      Sensed speed limits forward of ego:")
        for row in range(ObsVec.ZONES_FORWARD-1, -1, -1):
            row_res = [0.0]*5
            for col in range(5):
                base = ObsVec.BASE_SPD_LIMIT + col*ObsVec.NUM_ROWS
                z_idx = base + row
                spd_lim = raw_obs[z_idx]
                row_res[col] = spd_lim
            print("      row {:2d}   {:.3f}   {:.3f}   {:.3f}   {:.3f}   {:.3f}".format(row, row_res[0], row_res[1], \
                        row_res[2], row_res[3], row_res[4]))
        """

        # Look for key press to indicate pausing the activity
        if graphics.key_press_event() == PAUSE_KEY:

            # Wait for the resume signal (another key press)
            key = None
            while key != RESUME_KEY:
                if key == END_KEY:
                    print("///// User aborted.")
                    graphics.close()
                    sys.exit()
                key = graphics.wait_for_key_press()

        # If we are doing a special scenario for visualizing a single lane (only runs vehicle 1), then need to check for done
        # based on when that vehicle exits its assigned lane.
        if scenario >= 90:
            if not vehicles[1].active: #gets triggered in env.step()
                done = True

        # Summarize the episode
        if done:
            print("///// Episode complete: {}. Total reward = {:.2f}".format(info["reason"], episode_reward))
            while graphics.wait_for_key_press() != END_KEY:
                pass
            graphics.close()
            sys.exit()

    # If the episode is complete then get user approval to shut down
    if step >= episode_len:
        print("///// Terminated - reached desired episode length. Total reward = {:.2f}".format(episode_reward))
        while graphics.wait_for_key_press() != END_KEY:
            pass
        graphics.close()
        sys.exit()


######################################################################################################
######################################################################################################

if __name__ == "__main__":
   main(sys.argv)
