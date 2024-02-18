from cmath import inf
import sys
import time
from datetime import datetime
from typing import List
import numpy as np
import argparse
import pygame

from obs_vec import ObsVec
from scaler import *
from highway_env_wrapper import HighwayEnvWrapper
from bridgit_nn import BridgitNN
from graphics import Graphics

"""This program runs the selected policy checkpoint for one episode and captures key state variables throughout."""

def main(argv):

    # Handle any args
    program_desc = "Runs a single episode of the cda1 vehicle ML roadway environment in inference mode with any combination of vehicles."
    scenario_desc = '''Scenario - initial vehicle locations & speeds:
        0:  (default) everything randomized.
        1:  all neighbor vehicles in same lane.
        2:  no neighbor vehicles in ego's lane.
    10-18:  ego starts in lane 0-5, respectively; neighbor vehicles are randomized.
    20-28*: embedding run where vehicle 0 has Embed guidance but Bridgit model, starting in lanes 0-5 (primarily testing).
       29*: embedding run where vehicle 0 has Embed guidance but Bridgit model, starting in a random location.
    80-89:  special test configurations.
    90-98*: no ego; a single bot vehicle starts in lane 0-5, respectively, and drives to end of that lane (primarily testing).
    '''
    epilog = "If a non-learning scenario (*) is chosen then any checkpoint specified is ignored."
    parser = argparse.ArgumentParser(prog = argv[0], description = program_desc, epilog = epilog, formatter_class = argparse.RawTextHelpFormatter)
    parser.add_argument("-c", type = str, help = "Ray checkpoint dir containing the RL model to be run for the ego vehicle.")
    parser.add_argument("-L", type = int, default = 999999, help = "Max num time steps to run.")
    parser.add_argument("-r", type = str, default = None, help = "Roadway description (letter code); default is random selection.")
    parser.add_argument("-s", type = int, default = 0, help = scenario_desc)
    args = parser.parse_args()
    checkpoint = args.c
    scenario = args.s
    roadway_name = args.r
    episode_len = args.L

    # Verify that checkpoint & scenario are telling the same story. If we are going inference-only then erase the checkpoint.
    inference_only = False
    if 20 <= scenario <= 29  or  scenario >= 90:
        checkpoint = None
        inference_only = True
    else:
        if checkpoint is None:
            print("///// ERROR: Must specify a checkpoint if scenario is trainable (run `{} -h` for help).".format(argv[0]))
            sys.exit(1)

    # Set up the environment
    env_config = {  "time_step_size":           0.2,
                    "debug":                    0,
                    "verify_obs":               True,
                    "roadway_name":             roadway_name,
                    "valid_targets":            "all", #either "all" or comma-separated sequence of IDs (no brackets)
                    "randomize_targets":        True,
                    "scenario":                 scenario, #90-95 run single bot on lane 0-5, respectively; 0 = fully randomized
                    "episode_length":           episode_len,
                    "vehicle_file":             "config/vehicle_config_multi.yaml", #"vehicle_config_embedding.yaml",
                    "ignore_neighbor_crashes":  True,
                    "crash_report":             True,
                }
    env = HighwayEnvWrapper(env_config)
    #print("///// Environment configured. Params are:")
    #print(pretty_print(cfg.to_dict()))
    #env.reset() #TODO can remove this?

    # If we are reading a training checkpoint for the ego guidance model, then start up rllib to run it
    algo = None
    if not inference_only:

        # Set up the Ray framework
        import ray
        import ray.rllib.algorithms.ppo as ppo
        import ray.rllib.algorithms.sac as sac
        from ray.tune.logger import pretty_print
        from ray.rllib.models import ModelCatalog

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

        # Set up the simulation environment; note that we normally want to leave env checking enabled
        cfg.environment(env = HighwayEnvWrapper, env_config = env_config, disable_env_checking = True)
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
        print("\n///// No checkpoint loaded! Assuming that the ego vehicle provides all guidance logic.\n")

    # Prepare for a complete episode
    episode_reward = 0
    done = False
    action_list = [0, 0]
    raw_obs, _ = env.unscaled_reset()

    # Set up a local copy of all vehicles that have been configured
    vehicles = env.get_vehicle_data()
    if scenario >= 90:
        vehicles[0].active = False
    print("///// inference: after env.reset: vehicle[0] = ")
    vehicles[0].print()

    # Set up the graphic display & user interaction
    graphics = Graphics(env) #environment must have been reset prior to this call!
    PAUSE_KEY = pygame.locals.K_SPACE
    RESUME_KEY = pygame.locals.K_SPACE
    END_KEY = pygame.locals.K_ESCAPE
    COMMAND_INPUT_KEY = pygame.locals.K_h

    graphics.update(action_list, raw_obs, vehicles)
    obs = scale_obs(raw_obs)
    step = 0
    user_command = None
    total_override_reps = 1
    override_reps = 0

    # Wait for user to indicate okay to begin animation
    tgt_list = []
    for i, t in enumerate(env.roadway.targets):
        if t.active:
            tgt_list.append(i)
    print("\n///// Beginning episode with env {} and active targets are: {}. inference_only = {}".format(env.rollout_id, tgt_list, inference_only))
    key = None
    while key != RESUME_KEY:
        if key == END_KEY:
            print("///// User aborted.")
            graphics.close()
            sys.exit()
        key, _ = graphics.wait_for_key_press()

    # Loop on time steps until episode is complete
    while not done  and  step <= episode_len:
        step += 1
        #print("    * left bdry = {:1f} {:1f} {:1f}, right bdry = {:1f} {:1f} {:1f}, des = {:.2f} {:.2f} {:.2f}"
        #      .format(obs[ObsVec.BASE_LEFT_CTR_BDRY+0], obs[ObsVec.BASE_LEFT_CTR_BDRY+1], obs[ObsVec.BASE_LEFT_CTR_BDRY+2],
        #              obs[ObsVec.BASE_RIGHT_CTR_BDRY+0], obs[ObsVec.BASE_RIGHT_CTR_BDRY+1], obs[ObsVec.BASE_RIGHT_CTR_BDRY+2],
        #              obs[ObsVec.DESIRABILITY_LEFT], obs[ObsVec.DESIRABILITY_CTR], obs[ObsVec.DESIRABILITY_RIGHT]))

        # Grab ego vehicle actions if it is participating, or use dummies if not
        action = np.zeros(2)
        if not inference_only:
            if vehicles[0].active:

                # Use the hand-entered command list, if defined, for as many reps as specified in the override
                if user_command is not None:
                    action = np.asarray(user_command)
                    override_reps += 1
                    if override_reps >= total_override_reps:
                        user_command = None #so it won't get used in the following step

                # Else, get the action from the NN
                else:
                    try:
                        action = algo.compute_single_action(obs, explore = False)
                    except:
                        print("///// Exception trapped in inference during call to algo.compute_single_action(). obs = ")
                        print(obs)

        # Move the environment forward one time step
        raw_obs, reward, done, truncated, info = env.step(np.ndarray.tolist(action)) #obs returned is UNSCALED
        episode_reward += reward

        # Display current status of all the vehicles
        vehicles = env.get_vehicle_data()
        graphics.update(action, raw_obs, vehicles)

        # Scale the observations to be ready for NN ingest next time step
        obs = scale_obs(raw_obs)

        if inference_only:
            print("///// step {:3d}: ln {}, LC {}, SL = {:.1f}, spd cmd = {:.2f}, spd = {:.2f}, p = {:.1f}, r = {:7.4f} {}"
                    .format(step, vehicles[0].lane_id, vehicles[0].lane_change_count, raw_obs[ObsVec.LOCAL_SPD_LIMIT], \
                            raw_obs[ObsVec.SPEED_CMD], raw_obs[ObsVec.SPEED_CUR], vehicles[0].p, reward, info["reward_detail"]))
        else:
            print("///// step {:3d}: sc act = [{:5.2f} {:5.2f}], ln {}, LC {}, SL = {:.1f}, spd cmd = {:.2f}, spd = {:.2f}, p = {:.1f}, r = {:7.4f} {}"
                    .format(step, action[0], action[1], vehicles[0].lane_id, vehicles[0].lane_change_count, raw_obs[ObsVec.LOCAL_SPD_LIMIT], \
                            raw_obs[ObsVec.SPEED_CMD], raw_obs[ObsVec.SPEED_CUR], vehicles[0].p, reward, info["reward_detail"]))
        #print("      Vehicle 1 speed = {:.1f}".format(vehicles[1].cur_speed))

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

        # Look for key press to indicate pausing the activity
        ev = graphics.key_press_event()
        if ev is not None  and  ev.key == PAUSE_KEY:

            # Wait for the resume signal or another instruction
            key = None
            while key != RESUME_KEY:

                # Handle user abort key
                if key == END_KEY:
                    print("///// User aborted.")
                    graphics.close()
                    sys.exit()

                # Handle user input of ego commands. Since this is an unusual command intended for debugging purposes, no
                # need to put in lots of input validation logic. Expected output is a list containing
                #   speed command (scaled to [-1, 1])
                #   LC command (in [-1, 1])
                #   optional number of steps to repeat (default 1)
                if key == COMMAND_INPUT_KEY  and  user_command is None:
                    cmd_list = graphics.get_command_input()
                    assert type(cmd_list) == list  and  len(cmd_list) >= 2, "///// ERROR: invalid user input command: {}".format(cmd_list)
                    assert -1.0 <= cmd_list[0] <= 1.0, "///// ERROR: Illegal speed command given: {}".format(cmd_list[0])
                    assert -1.0 <= cmd_list[1] <= 1.0, "///// ERROR: Illegal LC command given: {}".format(cmd_list[1])
                    total_override_reps = 1
                    if len(cmd_list) == 3:
                        reps = int(cmd_list[2])
                        if reps > 1:
                            total_override_reps = reps
                            override_reps = 0
                    user_command = cmd_list[:2]
                    print("///// Manual command override: {} applied to next {} steps".format(user_command, total_override_reps))

                key, _ = graphics.wait_for_key_press()

        # If we are doing a special scenario for visualizing a single lane (only runs vehicle 1), then need to check for done
        # based on when that vehicle exits its assigned lane.
        if scenario >= 90:
            if not vehicles[1].active: #gets triggered in env.step()
                done = True

        # Summarize the episode
        if done:
            print("///// Episode complete: {}. Total reward = {:.2f}".format(info["reason"], episode_reward))
            key = None
            while key != END_KEY:
                key, _ = graphics.wait_for_key_press()
            graphics.close()
            sys.exit()

    # If the episode is complete then get user approval to shut down
    if step >= episode_len:
        print("///// Terminated - reached desired episode length. Total reward = {:.2f}".format(episode_reward))
        key = None
        while key != END_KEY:
            key, _ = graphics.wait_for_key_press()
        graphics.close()
        sys.exit()


######################################################################################################
######################################################################################################

if __name__ == "__main__":
   main(sys.argv)
