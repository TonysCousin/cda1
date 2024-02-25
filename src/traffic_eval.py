from cmath import inf
import sys
import time
from datetime import datetime
from typing import List
import numpy as np
import argparse

from obs_vec import ObsVec
from scaler import *
from highway_env_wrapper import HighwayEnvWrapper

"""This program performs a standardized evaluation of traffic flow for purposes of comparing agent models.
    It runs a common set of scenarios that each have specific roadway, destination targets and initial
    conditions on the ego vehicle, as well as numbers and types of neighbor vehicles. Each scenario is run
    for multiple episodes to for a statistically meaningful picture of the stochastic performance.
"""

def main(argv):

    #TODO - update this!

    # Handle any args
    program_desc = "Runs a single episode of the cda1 vehicle ML roadway environment in inference mode with any combination of vehicles."
    epilog = "If a non-learning scenario (*) is chosen then any checkpoint specified is ignored."
    parser = argparse.ArgumentParser(prog = argv[0], description = program_desc, epilog = epilog, formatter_class = argparse.RawTextHelpFormatter)
    parser.add_argument("-b", type = str, default = None, help = "Ray weights file containing the RL model to be run for all vehicles with Bridgit guidance.")
    args = parser.parse_args()
    bridgit_weights = args.b

    # Create the environment
    env_config = {  "time_step_size":           0.2,
                    "debug":                    0,
                    "verify_obs":               False,
                    "randomize_targets":        False,
                    "vehicle_file":             "config/vehicle_config_multi.yaml", #"vehicle_config_embedding.yaml",
                    "ignore_neighbor_crashes":  False,
                    "crash_report":             True,
                }
    env = HighwayEnvWrapper(env_config)

    # Pass in any NN model weights file that may have been specified, in case an override is desired
    env.set_bridgit_model_file(bridgit_weights)

    # Prepare for a complete episode
    env.set_scenario(50)
    episode_reward = 0
    done = False
    raw_obs, _ = env.unscaled_reset()
    print("***** Using roadway {}, targets {}, num_vehicles = {}".format(env.roadway.name, env.roadway.get_active_target_list(), env.num_vehicles))

    # Loop on time steps until episode is complete
    step = 0
    while not done:
        step += 1

        # Use dummy actions to pass to the environment, since all vehicles are non-learning and the environment
        # runs their guidance models.
        action = np.zeros(2)

        # Move the environment forward one time step
        raw_obs, reward, done, truncated, info = env.step(np.ndarray.tolist(action)) #obs returned is UNSCALED
        episode_reward += reward #TODO: need to deal with rewards?

        # Display current status of all the vehicles
        vehicles = env.get_vehicle_data()

        print("///// step {:3d}: ln {}, LC {}/{}, SL = {:.1f}, spd cmd = {:.1f}, spd = {:.1f}, p = {:.1f}, r = {:7.4f} {}"
                    .format(step, vehicles[0].lane_id, int(raw_obs[ObsVec.LC_UNDERWAY]), vehicles[0].lane_change_count, raw_obs[ObsVec.LOCAL_SPD_LIMIT], \
                            raw_obs[ObsVec.SPEED_CMD], raw_obs[ObsVec.SPEED_CUR], vehicles[0].p, reward, info["reward_detail"]))

        """
        # --- this section for debugging only ---
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

        # If we are doing a special scenario for visualizing a single lane (only runs vehicle 1), then need to check for done
        # based on when that vehicle exits its assigned lane.
        if not vehicles[0].active: #gets triggered in env.step() #TODO: needed?
            if not done:
                print("***   done triggered in fail-safe test.")
            done = True

        # Summarize the episode
        if done:
            print("///// Episode complete: {}. Total reward = {:.2f}".format(info["reason"], episode_reward))
            sys.exit()

    # If the episode is complete then get user approval to shut down
    print("///// Terminated - reached desired episode length. Total reward = {:.2f}".format(episode_reward))
    sys.exit()


######################################################################################################
######################################################################################################

if __name__ == "__main__":
   main(sys.argv)
