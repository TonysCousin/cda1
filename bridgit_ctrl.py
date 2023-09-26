from typing import List, Dict
import copy
import numpy as np

from constants import Constants
from obs_vec import ObsVec
from hp_prng import HpPrng
from roadway_b import Roadway
from lane_change import LaneChange
from vehicle_controller import VehicleController


class BridgitCtrl(VehicleController):

    """Defines the control algorithm for the Bridgit NN agent, which has learned some optimum driving in the given roadway."""

    PLAN_EVERY_N_STEPS = 5 #num time steps between route plan updates


    def __init__(self,
                 prng       : HpPrng,
                 roadway    : Roadway,
                 targets    : List,
                ):

        super().__init__(prng, roadway, targets)

        self.steps_since_plan = self.PLAN_EVERY_N_STEPS


    def step(self,
             obs    : np.array, #vector of local observations available to the instantiating vehicle
            ) -> List:          #returns a list of action commands, such that
                                # item 0: desired speed, m/s
                                # item 1: lane change command (corresponds to type LaneChange)

        """Applies the control algorithm for one time step to convert vehicle observations into action commands."""

        raise NotImplementedError("///// BridgitCtrl.step has yet to get a checkpoint running capability.")


    def plan_route(self,
                   obs      : np.array,     #the observation vector for this vehicle for the current time step
                  ) -> np.array:

        """Produces a long-range route plan for the agent. This logic will eventually be replaced by a trained NN, but we start with
            fairly simple procedural logic in order to focus training on the control capability first. This method may feel better placed
            in the BridgitController class due to its purpose, but to make the training loop work cleanly for the controller NN, this is
            where all observations are generated, which is how the plan must be communicated to the control NN.

            This algorithm will be executed at a lower frequency than the control loop, so the plan is intended to live for several time
            steps. Its output is a 3-tuple of lane preferences, relative to the host's current lane.  Each element represents the preference
            as a probability of: moving one lane to the left, staying in the current lane, moving one lane to the right. These preferences
            are to be interpreted as desirable in the immediate future (within the next 1 sec or so). As probabilities, the three elements
            will sum to 1.

            The goal is to reach ANY eligible target destination. There is no preference for one over the others. In fact, the plan may
            change its "intended" destination over the course of the route, to accommodate maneuvering that the controller may decide that
            it needs to do.
        """

        # If enough time steps have not passed, then return the input vector
        print("***** Entering BridgitCtrl.plan_route: steps_since_plan = {}, my lane = {}, my p = {:.1f}"
              .format(self.steps_since_plan, self.my_vehicle.lane_id, self.my_vehicle.p))
        self.steps_since_plan += 1
        if self.steps_since_plan < self.PLAN_EVERY_N_STEPS:
            return obs

        # List indices for positions relative to the host's lane
        LEFT = 0
        CENTER = 1
        RIGHT = 2

        SMALL_DISTANCE = 150.0 #distance below which an immediate lane change is necessary in order to get to the target, m
                                # at a nominal highway speed and LC duration, the vehicle will cover 80-110 m.

        # Container for info in each of the 3 positions indicated above
        class PosInfo:

            def __init__(self):
                self.lane_id = -1   #the lane represented by this position
                self.tgt_id = -1    #the index of the target in this lane, if one exists
                self.max_p = 0.0    #the max_p allowable in this lane to reach any target
                self.prob = 0.0     #the desirability of being in this lane now

            def pri(self):
                return "lane = {:2d}, tgt = {:2d}, max_p = {:.1f}, prob = {:.3f}.  ".format(self.lane_id, self.tgt_id, self.max_p, self.prob)

        # Get all of the eligible targets and the starting points for each possible target destination and merge them, taking the
        # set-wise union of points in each lane.
        starting_points = {}
        target_lane = {}
        for idx, tgt in enumerate(self.targets):
            target_lane[tgt.lane_id] = idx
            starts = tgt.get_starting_points()
            starting_points = self._union(starting_points, starts)
            print("      target {} is in lane {}. starting_points = {}".format(idx, tgt.lane_id, starting_points)) #TODO test
        print("      target_lane = ", target_lane)

        # Identify which of the relative positions reprsent which lanes in the roadway
        positions = [PosInfo()]*3
        print("      Positions declared. positions = ") #TODO test
        print([positions[k].pri() for k in range(3)])
        positions[CENTER].lane_id = self.my_vehicle.lane_id
        positions[LEFT].lane_id = self.my_vehicle.lane_id - 1 #if center is 0 then this appropriately indicates no lane present
        positions[RIGHT].lane_id = self.my_vehicle.lane_id + 1
        if positions[RIGHT].lane_id >= self.roadway.NUM_LANES:
            positions[RIGHT].lane_id = -1
        print("      After positions defined. positions = ") #TODO test
        print([positions[k].pri() for k in range(3)])

        # Loop through all 3 relative lane positions and identify which have targets in the same lane, and the max p over all of them
        max_max_p = 0.0
        for pos in positions:
            tgt_idx = -1
            try:
                idx = target_lane[pos.lane_id]
                if idx >= 0:
                    tgt_idx = idx
            except KeyError:
                pass

            pos.tgt_id = tgt_idx
            pos.max_p = starting_points[pos.lane_id]
            if pos.max_p > max_max_p:
                max_max_p = pos.max_p
        print("      After first pos loop: positions = ") #TODO test
        print([positions[k].pri() for k in range(3)])

        # Loop through the relative positions again
        sum_prob = 0.0
        for pos in positions:

            # If there is a target in the current lane, then assign it a value of 1
            if pos.tgt_id >= 0:
                pos.prob = 1.0

            # Else if this position doesn't match a real lane, then assign it a 0
            elif pos.lane_id < 0:
                pos.prob = 0.0

            # Else assign it a prob < 1 based on its max P relative to the other two positions
            else:
                pos.prob = pos.max_p / max_max_p

                # If host vehicle is only a small distance away from having to change lanes, set its prob to 0
                if pos.max_p - self.my_vehicle.p < SMALL_DISTANCE:
                    pos.prob = 0.0

            sum_prob += pos.prob

        # Scale the probabilities and return the obs vector
        if sum_prob == 0.0:
            sum_prob = 1.0
        for pos in positions:
            pos.prob /= sum_prob

            # TODO: update obs vector!



        print("      After final loop: positions = ") #TODO test
        print([positions[k].pri() for k in range(3)])

        # Indicate that planning has been completed for a while
        self.steps_since_plan = 0

        return obs


    def _union(self,
               a        : Dict, #first dict of starting points
               b        : Dict  #second dict of starting points
              ) -> Dict:

        """Merges the two given dicts by taking the set-wise union of the max P values for each indicated lane. Return is the union
            of the two input sets, meaning it contains the largest P value from either dict for a given lane ID. These dicts are assumed
            to have the key (int) representing a lane ID and the value (float) representing the maximum P coordinate allowable to reach
            a target from that lane, in the spirit of TargetDestination.get_starting_point().
        """

        a_copy = copy.deepcopy(a)
        ret = copy.deepcopy(b)

        # Ensure everything in a represents the max value over the shared sets
        for k, va in a_copy.items():
            try:
                vb = b[k]
                if vb > va:
                    a_copy[k] = vb
            except KeyError:
                pass

        # Get any additional items that are only found in b
        ret.update(a_copy)

        return ret
