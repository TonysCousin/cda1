from typing import List
import numpy as np

from constants import Constants
from obs_vec import ObsVec
from hp_prng import HpPrng
from roadway_b import Roadway
from lane_change import LaneChange
from vehicle_controller import VehicleController


class BridgitCtrl(VehicleController):

    """Defines the control algorithm for the Bridgit NN agent, which has learned some optimum driving in the given roadway.
    """

    def __init__(self,
                 prng       : HpPrng,
                 roadway    : Roadway,
                 targets    : List,
                ):

        super().__init__(prng, roadway, targets)


    def step(self,
             obs    : np.array, #vector of local observations available to the instantiating vehicle
            ) -> List:          #returns a list of action commands, such that
                                # item 0: desired speed, m/s
                                # item 1: lane change command (corresponds to type LaneChange)

        """Applies the control algorithm for one time step to convert vehicle observations into action commands."""

        raise NotImplementedError("///// BridgitCtrl.step has yet to get a checkpoint running capability.")
