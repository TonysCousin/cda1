from cmath import inf
import sys
import time
from typing import List
import copy
import numpy as np
import argparse

from obs_vec import ObsVec
from highway_env_wrapper import HighwayEnvWrapper
from graphics import Graphics

"""This program trains an autoencoder to compress the host vehicle's sensor observations, then decompress them to form
    a reasonably accurate reproduction of the original sensor data. Once the training is satisfactory, the weights of
    the encoder layer are saved for future use in our CDA agent.
"""










def main(argv):

    # Handle any args
    filename = "observations.csv"
    max_episodes = 10000
    max_time_steps = 2000000
    program_desc = "Runs data collection for vector embedding in the cda1 project."
    scenario_desc = "20-25:  starting in lanes 0-5 (primarily testing).\n" \
                    + "29:     starting in a random lane.\n"
    parser = argparse.ArgumentParser(prog = argv[0], description = program_desc, epilog = "Will run until either max episodes or max timesteps is reached.")
    parser.add_argument("-e", type = int, default = max_episodes, help = "Max number of episodes to run (default = {})".format(max_episodes))
    parser.add_argument("-t", type = int, default = max_time_steps, help = "Max total timesteps to collect (default = {})".format(max_time_steps))
    parser.add_argument("-g", action = "store_true", default = False, help = "Show each episode graphically (default: no)")
    parser.add_argument("-s", type = int, default = 29, help = scenario_desc)
    parser.add_argument("-f", type = str, default = filename, help = "Name of the data file produced (default: {})".format(filename))
    parser.add_argument("-o", action = "store_true", default = False, help = "If the specified filename already exists, overwrite it (default: no overwrite)")
    args = parser.parse_args()

    max_episodes = args.e
    max_time_steps = args.t
    use_graphics = args.g
    filename = args.f
    overwrite = args.o
    scenario = args.s
    if scenario not in [20, 21, 22, 23, 24, 25, 29]:
        print("///// ERROR: invalid scenario specified: {}".format(scenario))
        sys.exit(1)

    oi = "without"
    if overwrite:
        oi = "WITH"
    print("\n***** inputs: scenario = {}, max_episodes = {}, max_time_steps = {} to file {} {} overwrite"
          .format(scenario, max_episodes, max_time_steps, filename, oi))
    print("      Each obs entry will be {} elements long.".format(ObsVec.FINAL_ELEMENT + 1 - ObsVec.BASE_SENSOR_DATA))

    # Set up the environment
    env_config = {  "time_step_size":           0.2,
                    "debug":                    0,
                    "verify_obs":               True,
                    "scenario":                 scenario, #90-95 run single bot on lane 0-5, respectively; 0 = fully randomized
                    "vehicle_file":             "vehicle_config_embedding.yaml", #"vehicle_config_embedding.yaml",
                    "ignore_neighbor_crashes":  True,
                    "crash_report":             False,
                }
    env = HighwayEnvWrapper(env_config)
    env.reset()

    # Set up a local copy of all vehicles that have been configured
    vehicles = env.get_vehicle_data()

    # Set up the graphic display
    graphics = None
    if use_graphics:
        graphics = Graphics(env)

    # Open the data file that will hold the experiences
    data_file = None
    if overwrite:
        data_file = open(filename, 'w')
    else:
        data_file = open(filename, 'a')

    # Loop on episodes
    total_steps = 0
    for ep in range(max_episodes):

        # Set up for the next episode
        done = False
        action_list = [0, 0]
        raw_obs, _ = env.unscaled_reset()
        if use_graphics:
            graphics.update(action_list, raw_obs, vehicles)
        obs = env.scale_obs(raw_obs)
        step = 0
        if use_graphics:
            time.sleep(2)

        # Loop on time steps until end of episode
        while not done:
            step += 1

            # Move the environment forward one time step (includes dynamics of all vehicles); for scenarios 20-29 the
            # environment will produce commands for vehicle 0 as well, so the action list passed in here is ignored.
            raw_obs, reward, done, truncated, info = env.step(action_list) #obs returned is UNSCALED

            # Scale the new observattion vector, then extract only the "sensor" data from the relative location zones.
            # Then and add that info to the output file.
            obs = copy.copy(env.scale_obs(raw_obs))
            sensor_obs = obs[ObsVec.BASE_SENSOR_DATA : ObsVec.FINAL_ELEMENT+1]
            np.savetxt(data_file, sensor_obs.reshape(1, len(sensor_obs)), delimiter = ", ", fmt = "%f") #requires a 2D array

            # Display current status of all the vehicles
            if use_graphics:
                action = np.array(action_list)
                vehicles = env.get_vehicle_data()
                graphics.update(action, raw_obs, vehicles)
                print("///// step {:3d}: lane = {}, LC # = {}, spd cmd = {:.2f}, spd = {:.2f}, p = {:.1f}, r = {:7.4f} {}"
                    .format(step, vehicles[0].lane_id, vehicles[0].lane_change_count, \
                            raw_obs[ObsVec.SPEED_CMD], raw_obs[ObsVec.SPEED_CUR], vehicles[0].p, reward, info["reward_detail"]))

            # Wrap up the episode
            if done:
                print("//    Episode {} complete: {}".format(ep, info["reason"]))
                if use_graphics:
                    time.sleep(1)

        # If we've maxed out the number of time steps in the run, then exit the loop
        total_steps += step
        if total_steps >= max_time_steps:
            break

    # Summarize the run and close up resources
    print("///// All data collected.  {} episodes complete, covering {} time steps.".format(ep+1, total_steps))
    data_file.close()
    if use_graphics:
        input("///// Press Enter to close...")
        graphics.close()
    sys.exit()


######################################################################################################
######################################################################################################

if __name__ == "__main__":
   main(sys.argv)
