from typing import List, Dict
import copy
import math
import numpy as np
import torch
from torch.nn import functional as F
from gymnasium.spaces import Box

from constants import Constants
from obs_vec import ObsVec
from hp_prng import HpPrng
from scaler import *
from roadway import Roadway
from lane_change import LaneChange
from vehicle_guidance import VehicleGuidance
from bridgit_nn import BridgitNN


class BridgitGuidance(VehicleGuidance):

    """Defines the guidance algorithms for the Bridgit NN agent, which has learned some optimum driving in the given roadway."""

    PLAN_EVERY_N_STEPS = 5  #num time steps between route plan updates; CAUTION: this needs to be < duration of a lane change

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
                 is_learning: bool = True,
                 obs_space  : Box = None,
                 act_space  : Box = None,
                 name       : str = "BridgitGuidance"
                ):
        super().__init__(prng, is_learning, obs_space, act_space, name)

        self.positions = None

        # If this also is a non-learning AI agent (running in inference mode only), then we need to explicitly instantiate that
        # model here so that it can be executed in inference mode.
        self.tactical_model = None
        if not is_learning:
            config = {"inference_only": True}
            self.tactical_model = BridgitNN(obs_space, act_space, 4, config, "Bridgit") #4 outputs: all means, the ann stddevs
            self.tactical_model.eval()


    def reset(self,
              roadway       : Roadway,  #the roadway geometry used for this episode
              init_lane     : int,      #the lane the vehicle is starting in
              init_p        : float,    #vehicle's initial P coordinate, m
             ):

        """Overrides the base class method because the logic here depends upon the base member variable my_vehicle, which is not
            defined in the constructor, but the logic doesn't need to be run every time step.
        """

        super().reset(roadway, init_lane, init_p)

        # Redefine the relative position objects
        if self.positions is not None:
            del self.positions
        self.positions = [self.PosInfo() for _ in range(3)]

        # Initialize the relative position info
        self._set_relative_lane_pos()

        # Ensure the strategic planning method will get executed on the first step of the episode
        self.steps_since_plan = self.PLAN_EVERY_N_STEPS

        # Get all of the targets and the starting points for each possible target destination and merge them, taking the
        # set-wise intersection of points in each lane. If this is a learning vehicle, then only use active targets;
        # otherwise use every target that is defined.
        self.starting_points = {}
        self.lane_to_target = {}
        for idx, tgt in enumerate(self.targets):
            if self.is_learning  and  not tgt.active:
                continue
            self.lane_to_target[tgt.lane_id] = idx
            starts = tgt.get_starting_points()
            self.starting_points = self._dict_intersection(self.starting_points, starts)


    def step(self,
             obs    : np.array, #vector of local observations available to the instantiating vehicle; MAY BE MODIFIED
            ) -> List:          #returns a list of action commands, such that
                                # item 0: desired speed, m/s
                                # item 1: lane change command (corresponds to type LaneChange)

        """Applies the tactical guidance algorithm for one time step to convert vehicle observations into action commands.
            It invokes the strategic guidance algorithm as well, when appropriate. This means the input obs arg may be
            modified by this method. If the vehicle is in inference-only mode, only the strategic guidance will be executed
            (and obs updated). It will be up to the calling environment to provide the tactical action vector.

            CAUTION: the inputs and outputs are unscaled, but the NN being invoked here operates on scaled data.
        """

        # Invoke the strategic guidance (route planning) every time step, and let it decide when it needs to go to work.
        # NOTE: this may modify the obs vector in step's arg list (updated lane change desirability values).
        obs = self.plan_route(obs)

        # If this object represents a learning vehicle then return a meaningless action list
        if self.is_learning:
            return [0.0, 0.0]

        # Scale the input ndarray then convert it to a dict with a tensor in it
        assert len(obs.shape) == 1, "///// ERROR: BridgitGuidance.step: input obs has incompatible shape: {}".format(obs.shape)
        scaled = scale_obs(obs)
        batch_obs = {"obs": torch.from_numpy(np.expand_dims(scaled, axis = 0)).float()}

        # Run the NN to generate the new action commands. These may not respect the required bounds on actions.
        with torch.no_grad():
            net_out, _ = self.tactical_model(batch_obs)

            # The tactical model runs the NN, which outputs a tuple, the first element of which is a tensor containing all
            # the means then all the stddevs. Use these to generate a tensor of chosen (deterministic) actions, so ignore stddevs.
            # Use tanh to squash the raw network outputs into the required scaled range of [-1, 1]
            scaled_actions = F.tanh(net_out[0, :2])
            actions = unscale_actions(scaled_actions)
            #print("***   BridgitGuidance: scaled_actions = {}, returning actions = {}".format(scaled_actions, actions))

            return actions


    def plan_route(self,
                   obs      : np.array,     #the observation vector for this vehicle for the current time step
                  ) -> np.array:            #returns the (possibly modified) observation vector

        """Produces a strategic guidance (route) plan for the agent. This logic will eventually be replaced by a trained NN, but we
            start with fairly simple procedural logic in order to focus training on the tactical guidance first.

            This algorithm will be executed at a lower frequency than the control loop, so the plan is intended to live for several time
            steps. Its output is a 3-tuple of lane preferences, relative to the host's current lane.  Each element represents the preference
            as a probability of: moving one lane to the left, staying in the current lane, moving one lane to the right. These preferences
            are to be interpreted as desirable in the immediate future (within the next 1 sec or so). As probabilities, the three elements
            will sum to 1.

            The goal is to reach ANY eligible target destination. There is no preference for one over the others. In fact, the plan may
            change its "intended" destination over the course of the route, to accommodate maneuvering that the tactical guidance may decide
            that it needs to do.
        """

        # Check for a lane change being completed in this time step. If so, then force an immediate replan to update the LC desirabilities.
        total_lc_steps = self.my_vehicle.model.lc_compl_steps
        if obs[ObsVec.STEPS_SINCE_LN_CHG] == total_lc_steps:
            self.steps_since_plan = total_lc_steps

        # If not enough time steps have passed, then return the input vector
        self.steps_since_plan += 1
        if self.steps_since_plan < self.PLAN_EVERY_N_STEPS:
            return obs

        # Update the relative lane positions, since the vehicle may have changed lanes since the previous call
        self._set_relative_lane_pos()

        # Loop through all 3 relative lane positions and identify which have targets in the same lane, and find the max distance allowed
        # to travel in that lane before a lane change is necessary.
        for pos in self.positions:
            if pos.lane_id < 0:
                continue

            tgt_idx = -1
            try:
                idx = self.lane_to_target[pos.lane_id] #target is in the indicated lane
                if idx >= 0:
                    tgt_idx = idx
            except KeyError:
                pass

            pos.tgt_id = tgt_idx

            # Get the worst-case starting P location in this lane that could still reach an active target. In case there are no reachable
            # targets, default the starting point to P = 0.
            sp = 0.0
            try:
                sp = self.starting_points[pos.lane_id]
            except KeyError:
                """ #for testing only
                at = []
                for t_idx, t in enumerate(self.targets):
                    if t.active:
                        at.append(t_idx)
                print("//    INFO: BridgitGuidance.plan_route: no starting point found for lane_id = {} to any active targets {}".format(pos.lane_id, at))
                """
                pass

            pos.delta_p = sp - self.my_vehicle.p #may be negative

        # Loop through the relative positions again to assign desirable probabilities of being in that lane
        max_prob = 0.0
        for i, pos in enumerate(self.positions):

            # If there is a target in the current lane, then assign it a value of 1
            if pos.tgt_id >= 0:
                pos.prob = 1.0

            # Else if this position doesn't match a real lane, then assign it a 0
            elif pos.lane_id < 0:
                pos.prob = 0.0

            # Else assign it a prob < 1 based on its delta P
            else:
                if pos.delta_p <= 1.0:
                    pos.prob = 0.0
                else:
                    pos.prob = min(0.04 * math.sqrt(0.3*pos.delta_p), 0.9)

            # If this position is the lane to the left, then zero it out if a lane change is physically prohibited before the next planning cycle
            # (vehicle will traverse approx 6 sensor zones, or 2 boundary regions, during that time).
            if i == self.LEFT:
                for j in range(2):
                    bdry = obs[ObsVec.BASE_LEFT_CTR_BDRY + j]
                    if bdry < 0.0:
                        pos.prob = 0.0
                        break

            # Same test for lane to the right
            elif i == self.RIGHT:
                for j in range(2):
                    bdry = obs[ObsVec.BASE_RIGHT_CTR_BDRY + j]
                    if bdry < 0.0:
                        pos.prob = 0.0
                        break

            if pos.prob > max_prob:
                max_prob = pos.prob

        # If all probs are zero (no hope of reaching any target), then we want to follow center lane, but
        # with a low probability, since this value is used by the environment to reward lane choice behavior; it is in this pickle
        # because of poor choices upstream, so don't want to scoop up reward points now. With this adjustment, we don't want to
        # normalize the values; 1.0 should only be used if a target is straight ahead.
        if max_prob < 0.01:
            self.positions[self.CENTER].prob = 0.01

        # Update the obs vector with the new desirability info
        obs[ObsVec.DESIRABILITY_LEFT]   = self.positions[self.LEFT].prob
        obs[ObsVec.DESIRABILITY_CTR]    = self.positions[self.CENTER].prob
        obs[ObsVec.DESIRABILITY_RIGHT]  = self.positions[self.RIGHT].prob
        #print("*     plan_route done: output = ", obs[ObsVec.DESIRABILITY_LEFT : ObsVec.DESIRABILITY_RIGHT+1])

        # Indicate that planning has been completed for a while
        self.steps_since_plan = 0

        return obs


    ##### Internal methods #####


    def _set_relative_lane_pos(self):
        """Identifies which of the relative positions represent which lanes in the roadway. All info is stored in object member
            list "positions", so there is no return.
        """

        # Clear previously stored data
        for p in self.positions:
            p.delta_p = 0.0
            p.prob = 0.0
            p.tgt_id = -1

        # Update the lane assignments for each
        self.positions[self.CENTER].lane_id = self.my_vehicle.lane_id
        self.positions[self.LEFT].lane_id = self.my_vehicle.lane_id - 1 #if center is 0 then this appropriately indicates no lane present
        self.positions[self.RIGHT].lane_id = self.my_vehicle.lane_id + 1
        if self.positions[self.RIGHT].lane_id >= self.roadway.NUM_LANES:
            self.positions[self.RIGHT].lane_id = -1


    def _dict_intersection(self,
                    a        : Dict, #first dict of starting points
                    b        : Dict  #second dict of starting points
                   ) -> Dict:

        """Merges the two given dicts by taking the set-wise intersection of the max P values for each indicated lane. Return is the
            intersection of the two input sets, meaning it contains the largest P value from either dict for a given lane ID. These
            dicts are assumed to have the key (int) representing a lane ID and the value (float) representing the maximum P coordinate
            allowable to reach a target from that lane, in the spirit of TargetDestination.get_starting_point().
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
