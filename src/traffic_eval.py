from cmath import inf
import statistics
import sys
import time
from datetime import datetime
from typing import List
import numpy as np
import argparse
from ray.rllib.env.apis.task_settable_env import TaskSettableEnv

from obs_vec import ObsVec
from scaler import *
from highway_env_wrapper import HighwayEnvWrapper

"""This program performs a standardized evaluation of traffic flow for purposes of comparing agent models.
    It runs a common set of scenarios that each have specific roadway, destination targets and initial
    conditions on the ego vehicle, as well as numbers and types of neighbor vehicles. Each scenario is run
    for multiple episodes to for a statistically meaningful picture of the stochastic performance.
"""

class StatsAccumulator:
    """Accumulates performance statistics on an arbitrary set of episodes."""

    def __init__(self):
        self.num_episodes = 0
        self.sd_vals = []
        self.acc_vals = []
        self.dist_vals = []
        self.success_vals = []
        self.vehicle_counts = []
        self.score_vals = []


    def add(self,
            sd      : float,    #speed disadvantage for the episode, m/km
            acc     : float,    #acceleration integration for the episode, m/s/km
            dist    : float,    #distance driving in the episode, km
            success : bool,     #was the episode a success?
            vehicles: int,      #actual number of vehicles used in the episode
            ep_score: float,    #computed score of the episode performance
           ):
        """Adds the stats for a single episode to the collector."""

        self.num_episodes += 1
        self.sd_vals.append(sd)
        self.acc_vals.append(acc)
        self.dist_vals.append(dist)
        self.success_vals.append(1 if success else 0)
        self.vehicle_counts.append(vehicles)
        self.score_vals.append(ep_score)


    def sd_distro(self) -> tuple[float, float]:
        """Returns a tuple of the mean & std dev of the speed disadvantage over the data set."""

        return self._distro(self.sd_vals)


    def acc_distro(self) -> tuple[float, float]:
        """Returns a tuple of the mean & std dev of the integrated acceleration over the data set."""

        return self._distro(self.acc_vals)


    def dist_distro(self) -> tuple[float, float]:
        """Returns a tuple of the mean & std dev of the speed disadvantage over the data set."""

        return self._distro(self.dist_vals)


    def score_distro(self) -> tuple[float, float]:
        """Returns a tuple of the mean & std dev of the episode scores over the data set."""

        return self._distro(self.score_vals)


    def num_successes(self) -> int:
        """Returns the number of successes in the data set."""

        return sum(self.success_vals)


    def num_vehicles(self) -> int:
        """Returns the total number of vehicles participating in all episodes in the data set."""

        return sum(self.vehicle_counts)


    def total_distance(self) -> float: #return distance is in km
        """Returns the total distance driven by all vehicles in all episodes."""

        return sum(self.dist_vals)


    def weighted_score(self) -> float:
        """Returns the average performance score for the set of episodes represented in this accumulator,
            weighted by the number of vehicles participating in each episode.
        """

        assert len(self.score_vals) == len(self.vehicle_counts), "///// ERROR: accumulator has {} score values and {} vehicle counts. Should be the same." \
                                                                    .format(len(self.score_vals), len(self.vehicle_counts))

        sum_score = 0.0
        for i in range(len(self.score_vals)):
            sum_score += self.score_vals[i] * self.vehicle_counts[i]

        return sum_score / sum(self.vehicle_counts)


    def _distro(self,
                values  : list, #a list of the input values
               ) -> tuple[float, float]:
        """Returns a tuple of the mean & std dev of the values in the given list."""

        if self.num_episodes == 0:
            raise ValueError("///// ERROR: StatsAccumulator attempting to compute results on an empty set.")

        mean = sum(values) / float(len(values))
        stddev = statistics.stdev(values)
        return mean, stddev


def main(argv):

    # Handle any args
    program_desc = "Runs a set of pre-defined episode of the cda1 vehicle ML roadway environment and analyzes the overall traffic performance."
    parser = argparse.ArgumentParser(prog = argv[0], description = program_desc, formatter_class = argparse.RawTextHelpFormatter)
    parser.add_argument("-b", type = str, default = None, \
                        help = "PyTorch weights file containing the RL model to be run for all vehicles with Bridgit guidance.")
    args = parser.parse_args()
    bridgit_weights = args.b

    # Create the environment
    env_config = {  "time_step_size":           0.2,
                    "debug":                    0,
                    "verify_obs":               False,
                    "randomize_targets":        False,
                    "vehicle_file":             "config/vehicle_config_multi.yaml", #"vehicle_config_embedding.yaml",
                    "ignore_neighbor_crashes":  False,
                    "crash_report":             False,
                }
    env = HighwayEnvWrapper(env_config)

    # Pass in any NN model weights file that may have been specified, in case an override is desired
    env.set_bridgit_model_file(bridgit_weights)

    # Define the schedule of episodes to be run, per the software rqmts doc
    #           Scenario ID     Num iterations
    #           -----------     --------------
    scenarios = [   (50,            20),#RoadwayB, targets 1,2, 16 neighbors
                    (51,            20),
                    (52,            20),
                    (53,            20),
                    (54,            20),
                    (55,            20),
                    (56,            5), #RoadwayB, targets 1,2, 25 neighbors
                    (57,            5),
                    (58,            5),
                    (59,            15), #RoadwayB, targets 1,2, all Bridgits
                    (60,            15),
                    (61,            15),
                    (62,            5), #RoadwayB, targets 1,2, only 6 neighbors
                    (63,            5),
                    (64,            5),
                    (65,            10), #RoadwayB, target 0
                    (66,            10),
                    (67,            5), #RoadwayC
                    (68,            5),
                    (69,            5),
                    (70,            5),
                    (71,            5), #RoadwayD
                    (72,            5),
                    (73,            5),
                ]

    # Scale factors for computing episode scores
    SD_SCALE    = 0.004
    ACC_SCALE   = 0.004

    # Create stats accumulators for all the subsets of data to be sliced
    road_b_stats = StatsAccumulator()
    road_c_stats = StatsAccumulator()
    road_d_stats = StatsAccumulator()
    b_16_neighbor_stats = StatsAccumulator()
    b_25_neighbor_stats = StatsAccumulator()
    b_7_neighbor_stats = StatsAccumulator()
    b_all_bridgit_stats = StatsAccumulator()
    b_mixed_type_stats = StatsAccumulator()
    b_lane0_stats = StatsAccumulator()
    b_lane1_stats = StatsAccumulator()
    b_lane2_stats = StatsAccumulator()
    b_lane3_stats = StatsAccumulator()
    b_lane4_stats = StatsAccumulator()
    b_lane5_stats = StatsAccumulator()
    all_eval_stats = StatsAccumulator()

    # Loop through each of the scenarios
    for item in scenarios:
        scenario_id, total_iters = item

        # Perform the specified number of iterations
        for iter in range(total_iters):

            # Run an episode and collect its metrics
            success = run_episode(env, scenario_id)

            # If the episode ended successfully, then set up to collect its performance, otherwise score it a 0
            distance = 0.0
            sd = 0.0
            acc = 0.0

            # Retrieve the performance metrics for each vehicle involved
            vehicles = env.get_vehicle_data()
            for v_idx , v in enumerate(vehicles):
                if not v.used:
                    print("*     vehicle {} inactive".format(v_idx))
                    continue
                v_dist, v_sd, v_acc = v.get_performance_metrics()
                distance += v_dist
                sd += v_sd
                acc += v_acc
                #print("***    Vehicle {}, dist = {:.0f}, sd = {:.1f}, acc = {:.1f}".format(v_idx, v_dist, v_sd, v_acc))

            # Normalize the metrics over all of the trajectories
            distance *= 0.001 #convert to km
            sd /= distance
            acc /= distance

            # Compute the episode score
            score = 0.0
            if success:
                score = 1.0 - SD_SCALE* sd - ACC_SCALE*acc
            print("///// Scenario {}, episode {}: {} vehicles, total driven {:4.1f} km, normalized sd = {:5.1f} m/km, acc = {:5.1f} m/s/km, score = {:.2f}"
                    .format(scenario_id, iter, env.get_num_vehicles_used(), distance, sd, acc, score))

            # Accumulate scoring for the eval set based on which scenario fits which subset of the data
            nv = env.get_num_vehicles_used()
            all_eval_stats.add(sd, acc, distance, success, nv, score)
            if 50 <= scenario_id <= 66:
                road_b_stats.add(sd, acc, distance, success, nv, score)

                if 50 <= scenario_id <= 55:
                    b_16_neighbor_stats.add(sd, acc, distance, success, nv, score)
                    b_mixed_type_stats.add(sd, acc, distance, success, nv, score)
                    if scenario_id == 50:
                        b_lane0_stats.add(sd, acc, distance, success, nv, score)
                    elif scenario_id == 51:
                        b_lane1_stats.add(sd, acc, distance, success, nv, score)
                    elif scenario_id == 52:
                        b_lane2_stats.add(sd, acc, distance, success, nv, score)
                    elif scenario_id == 53:
                        b_lane3_stats.add(sd, acc, distance, success, nv, score)
                    elif scenario_id == 54:
                        b_lane4_stats.add(sd, acc, distance, success, nv, score)
                    elif scenario_id == 55:
                        b_lane5_stats.add(sd, acc, distance, success, nv, score)

                elif 56 <= scenario_id <= 58:
                    b_25_neighbor_stats.add(sd, acc, distance, success, nv, score)
                    b_mixed_type_stats.add(sd, acc, distance, success, nv, score)
                    if scenario_id == 56:
                        b_lane0_stats.add(sd, acc, distance, success, nv, score)
                    elif scenario_id == 57:
                        b_lane3_stats.add(sd, acc, distance, success, nv, score)
                    elif scenario_id == 58:
                        b_lane5_stats.add(sd, acc, distance, success, nv, score)

                elif 59 <= scenario_id <= 61:
                    b_16_neighbor_stats.add(sd, acc, distance, success, nv, score)
                    b_all_bridgit_stats.add(sd, acc, distance, success, nv, score)
                    if scenario_id == 59:
                        b_lane0_stats.add(sd, acc, distance, success, nv, score)
                    elif scenario_id == 60:
                        b_lane3_stats.add(sd, acc, distance, success, nv, score)
                    elif scenario_id == 61:
                        b_lane5_stats.add(sd, acc, distance, success, nv, score)

                elif 62 <= scenario_id <= 64:
                    b_7_neighbor_stats.add(sd, acc, distance, success, nv, score)
                    b_mixed_type_stats.add(sd, acc, distance, success, nv, score)
                    if scenario_id == 62:
                        b_lane0_stats.add(sd, acc, distance, success, nv, score)
                    elif scenario_id == 63:
                        b_lane3_stats.add(sd, acc, distance, success, nv, score)
                    elif scenario_id == 64:
                        b_lane5_stats.add(sd, acc, distance, success, nv, score)

                elif scenario_id == 65:
                    b_16_neighbor_stats.add(sd, acc, distance, success, nv, score)
                    b_mixed_type_stats.add(sd, acc, distance, success, nv, score)
                    b_lane1_stats.add(sd, acc, distance, success, nv, score)
                elif scenario_id == 66:
                    b_16_neighbor_stats.add(sd, acc, distance, success, nv, score)
                    b_mixed_type_stats.add(sd, acc, distance, success, nv, score)
                    b_lane5_stats.add(sd, acc, distance, success, nv, score)

            elif 67 <= scenario_id <= 70:
                road_c_stats.add(sd, acc, distance, success, nv, score)
            elif 71 <= scenario_id <= 73:
                road_d_stats.add(sd, acc, distance, success, nv, score)
            else:
                raise ValueError("///// ERROR: scenario_id {} is unknown".format(scenario_id))

    # Display final summary info
    #print("\n///// All episodes complete. Total evaluation score = {:.3f}, {} of {} episodes successful ({:.0f}%) using {} vehicles, {:.0f} km of travel"
    #      .format(eval_score, num_successes, eval_episodes, 100.0*float(num_successes)/eval_episodes, eval_vehicles, eval_distance))

    # Display stats for each accumulator group
    print("\n/////                    Num          Num       Total       Total       Wgted       Score         Speed disadvantage     Integ accel")
    print("///// Data sample      episodes    successes   vehicles  distance, km   score   mean    std dev    mean    std dev     mean    std dev")
    print("///// -----------      --------    ---------   --------  ------------   -----   -----   -------   ------   -------     ----    -------")
    # Name length limited to 14:        "12345678901234"
    print_dataset(all_eval_stats,       "Full eval")
    print_dataset(road_b_stats,         "Roadway B all")
    print_dataset(road_c_stats,         "Roadway C all")
    print_dataset(road_d_stats,         "Roadway D all")
    print_dataset(b_7_neighbor_stats,   "B 7 neighbors")
    print_dataset(b_16_neighbor_stats,  "B 16 neighbors")
    print_dataset(b_25_neighbor_stats,  "B 25 neighbors")
    print_dataset(b_mixed_type_stats,   "B mixed types")
    print_dataset(b_all_bridgit_stats,  "B all Bridgit")
    print_dataset(b_lane0_stats,        "B Lane 0")
    print_dataset(b_lane1_stats,        "B Lane 1")
    print_dataset(b_lane2_stats,        "B Lane 2")
    print_dataset(b_lane3_stats,        "B Lane 3")
    print_dataset(b_lane4_stats,        "B Lane 4")
    print_dataset(b_lane5_stats,        "B Lane 5")


def print_dataset(dataset   : StatsAccumulator, #the dataset to be printed
                  name      : str,              #the name of the dataset
                 ):
    """Prints all of the data required from the specified dataset, according to the required table format."""

    if dataset.num_episodes == 0:
        print("      {:14s}       0".format(name))
        return

    percent_success = 100.0*dataset.num_successes() / dataset.num_episodes
    s_mean, s_std = dataset.score_distro()
    d_mean, d_std = dataset.sd_distro()
    a_mean, a_std = dataset.acc_distro()
    print("      {:14s}     {:3d}       {:3d} ({:2.0f}%)     {:4d}       {:6.1f}      {:5.3f}   {:5.3f}    {:5.3f}    {:5.1f}     {:5.2f}     {:5.1f}     {:5.2f}"
          .format(name, dataset.num_episodes, dataset.num_successes(), percent_success, dataset.num_vehicles(), dataset.total_distance(),
                  dataset.weighted_score(), s_mean, s_std, d_mean, d_std, a_mean, a_std))


def run_episode(env         : TaskSettableEnv,  #the highway environment model
                scenario    : int,              #the scenario to be run
               ) -> bool:                       #returns a flag indicating whether the episode completed successfully
    """Executes a single episode of the specified scenario.  There are several criteria that would cause an episode
        to end unsuccessfully. See the software requirements doc for detailed explanation.
    """

    # Prepare for the episode by identifying the scenario ID and resetting the environment
    env.set_scenario(scenario)
    done = False
    success = True
    env.unscaled_reset() #slightly faster than just reset()

    # Loop on time steps until episode is complete
    step = 0
    while not done:
        step += 1

        # Use dummy actions to pass to the environment, since all vehicles are non-learning and the environment
        # runs their guidance models.
        action = np.zeros(2)

        # Move the environment forward one time step
        raw_obs, reward, done, truncated, info = env.step(np.ndarray.tolist(action)) #obs returned is UNSCALED
        vehicles = env.get_vehicle_data()

        # Check for an unsuccessful situation
        res = info["reward_detail"]
        if  "Crash"             in res  or \
            "off road"          in res  or \
            "Vehicle stopped"   in res  or \
            "unable to reach"   in res  or \
            "not active"        in res:

            success = False

        #print("///// step {:3d}: ln {}, LC {}/{}, SL = {:.1f}, spd cmd = {:.1f}, spd = {:.1f}, p = {:.1f}, r = {:7.4f} {}"
        #            .format(step, vehicles[0].lane_id, int(raw_obs[ObsVec.LC_UNDERWAY]), vehicles[0].lane_change_count, raw_obs[ObsVec.LOCAL_SPD_LIMIT], \
        #                    raw_obs[ObsVec.SPEED_CMD], raw_obs[ObsVec.SPEED_CUR], vehicles[0].p, reward, info["reward_detail"]))

        # Summarize the episode once it is over
        if done  or  not vehicles[0].active:
            #print("///// Episode complete: {}".format(info["reason"]))
            return success


######################################################################################################
######################################################################################################

if __name__ == "__main__":
   main(sys.argv)
