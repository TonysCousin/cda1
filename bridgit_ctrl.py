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

    # List indices for positions relative to the host's lane
    LEFT = 0
    CENTER = 1
    RIGHT = 2

    class PosInfo:
        """Container for info in each of the 3 relative lane positions."""

        def __init__(self):
            self.lane_id = -1   #the lane represented by this position
            self.tgt_id = -1    #the index of the target in this lane, if one exists
            self.delta_p = 0.0  #the remaining distance (m) from vehicle location to the max_p allowable in this lane to reach any target
            self.prob = 0.0     #the desirability of being in this lane now

        def pri(self):
            return "lane = {:2d}, tgt = {:2d}, delta_p = {:.1f}, prob = {:.3f}.  ".format(self.lane_id, self.tgt_id, self.delta_p, self.prob)


    def __init__(self,
                 prng       : HpPrng,
                 roadway    : Roadway,
                 targets    : List,
                ):

        super().__init__(prng, roadway, targets)

        self.steps_since_plan = self.PLAN_EVERY_N_STEPS
        self.positions = [self.PosInfo() for each in range(3)]

        # Get all of the eligible targets and the starting points for each possible target destination and merge them, taking the
        # set-wise union of points in each lane.
        self.starting_points = {}
        self.lane_to_target = {}
        for idx, tgt in enumerate(self.targets):
            self.lane_to_target[tgt.lane_id] = idx
            starts = tgt.get_starting_points()
            self.starting_points = self._dict_union(self.starting_points, starts)


    def reset(self,
              init_lane     : int,      #the lane the vehicle is starting in
              init_p        : float,    #vehicle's initial P coordinate, m
             ):

        """Makes vehicle's initial location info available in case the instantiated controller wants to use it.
            Overrides the base class method because the logic here depends upon the base member variable my_vehicle, which is not
            defined in the constructor, but the logic doesn't need to be run every time step.
        """

        super().reset(init_lane, init_p)

        # Initialize the relative position info
        self._set_relative_lane_pos()


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
        self.steps_since_plan += 1
        if self.steps_since_plan < self.PLAN_EVERY_N_STEPS:
            return obs

        SMALL_DISTANCE = 150.0 #distance below which an immediate lane change is necessary in order to get to the target, m.
                                # At a nominal highway speed and LC duration, the vehicle will cover 80-110 m.

        # Update the relative lane positions, since the vehicle may have changed lanes since the previous call
        self._set_relative_lane_pos()

        # Loop through all 3 relative lane positions and identify which have targets in the same lane, and the max p over all of them
        max_delta_p = 0.0
        for pos in self.positions:
            if pos.lane_id < 0:
                continue

            tgt_idx = -1
            try:
                idx = self.lane_to_target[pos.lane_id]
                if idx >= 0:
                    tgt_idx = idx
            except KeyError:
                pass

            pos.tgt_id = tgt_idx
            pos.delta_p = self.starting_points[pos.lane_id] - self.my_vehicle.p
            if pos.delta_p > max_delta_p:
                max_delta_p = pos.delta_p

        # Check that the max delta-P is > 0; a value of 0 can happen in the final time step going past the target, so resetting it
        # is not a big deal, but needed dto avoid a divide by zero later.
        if max_delta_p <= 0.0:
            print("///// WARNING - BridgitCtrl.plan_route: max_delta_p = {:.1f}, ego lane = {}, p = {:.1f}. Positions:"
                  .format(max_delta_p, self.my_vehicle.lane_id, self.my_vehicle.p))
            max_delta_p = 3000.0

        # Loop through the relative positions again
        sum_prob = 0.0
        for pos in self.positions:

            # If there is a target in the current lane, then assign it a value of 1
            if pos.tgt_id >= 0:
                pos.prob = 1.0

            # Else if this position doesn't match a real lane, then assign it a 0
            elif pos.lane_id < 0:
                pos.prob = 0.0

            # Else assign it a prob < 1 based on its max P relative to the other two positions
            else:
                pos.prob = max(pos.delta_p / max_delta_p, 0.0)

                # If host vehicle is only a small distance away from having to change out of the indicated lane, set its prob to 0
                if pos.delta_p < SMALL_DISTANCE:
                    pos.prob = 0.0

            sum_prob += pos.prob

        # Scale the probabilities and return the obs vector
        if sum_prob == 0.0:
            print("///// WARNING BridgitCtrl.plan_route sum_prob = 0. ego_lane = {}, ego p = {:.1f}".format(self.my_vehicle.lane_id, self.my_vehicle.p))
            sum_prob = 1.0
            #TODO debug next 2 lines
            for pos in self.positions:
                print(pos.pri())
        for pos in self.positions:
            pos.prob /= sum_prob

        obs[ObsVec.DESIRABILITY_LEFT]   = self.positions[self.LEFT].prob
        obs[ObsVec.DESIRABILITY_CTR]    = self.positions[self.CENTER].prob
        obs[ObsVec.DESIRABILITY_RIGHT]  = self.positions[self.RIGHT].prob
        #print("*     plan_route done: output = ", obs[ObsVec.DESIRABILITY_LEFT : ObsVec.DESIRABILITY_RIGHT+1]) #TODO debug

        # Indicate that planning has been completed for a while
        self.steps_since_plan = 0
        print("***** BridgitCtrl.plan_route returning desirability for lane ", self.my_vehicle.lane_id, ": ", obs[ObsVec.DESIRABILITY_LEFT : ObsVec.DESIRABILITY_RIGHT+1]) #TODO debug

        return obs


    def _set_relative_lane_pos(self):
        """Identifies which of the relative positions represent which lanes in the roadway. All info is stored in object member
            list "positions", so there is no return.
        """

        self.positions[self.CENTER].lane_id = self.my_vehicle.lane_id
        self.positions[self.LEFT].lane_id = self.my_vehicle.lane_id - 1 #if center is 0 then this appropriately indicates no lane present
        self.positions[self.RIGHT].lane_id = self.my_vehicle.lane_id + 1
        if self.positions[self.RIGHT].lane_id >= self.roadway.NUM_LANES:
            self.positions[self.RIGHT].lane_id = -1



    def _dict_union(self,
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
