from typing import List
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


    def _plan_route(self,
                    obs     : np.array,     #the observation vector for this vehicle for the current time step
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

        # List indices
        LEFT = 0
        CENTER = 1
        RIGHT = 2

        # Container for info in each of the 3 positions indicated above
        class PosInfo:
            self.lane_id = -1
            self.tgt_id = -1
            self.max_p = 0.0
            self.prob = 0.0

        # Initialize the three probabilities to equal values. It's okay if they don't sum to 1 within this method, as they will be
        # scaled before the return. The important thing for now is their relative values. Higher values are more highly desired.
        p_left = 0.0
        p_center = 0.0
        p_right = 0.0

        # Get all of the eligible targets and the starting points for each possible target destination and merge them, taking the
        # set-wise union of points in each lane.

        # Loop through all 3 relative lane positions and identify which have targets in the same lane, and the max p over all of them

        # Loop through the relative positions again

            # If there is a target in the current lane, then assign it a value of 1

            # Else assign it a prob < 1 based on its max P relative to the other two positions

        # Scale the probabilities and return the obs list slice

        # Indicate that planning has been completed for a while
        self.steps_since_plan = 0
