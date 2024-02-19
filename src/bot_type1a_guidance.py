from vehicle_guidance import VehicleGuidance
import numpy as np
from typing import List
from gymnasium.spaces import Box

from constants import Constants
from obs_vec import ObsVec
from hp_prng import HpPrng
from roadway import Roadway
from lane_change import LaneChange


class BotType1aGuidance(VehicleGuidance):

    """Defines a tactical guidance algorithm for the Type 1 bot vehicle, which tries to drive at the speed limit at all times,
        but uses crude Adaptive Cruise Control (ACC). It also stays in its assigned lane always.
    """

    def __init__(self,
                 prng       : HpPrng,
                 is_learning: bool = True,
                 obs_space  : Box = None,
                 act_space  : Box = None,
                 name       : str = "BotType1aGuidance"
                ):

        # This cannot be a learning vehicle. But it is possible for this type to be specified in the ego
        # slot, intended as inference only. In this case the incoming is_learning flag would be true, but
        # we need to ignore that so the vehicle is not accidentally marked as such.
        super().__init__(prng, False, obs_space, act_space, name)

        # Pick an offset from whatever the posted speed limit
        self.speed_offset = 0.0 #follow the speed limit exactly


    def step(self,
             obs    : np.array, #vector of local observations available to the instantiating vehicle
            ) -> list:          #returns a list of action commands, such that
                                # item 0: desired speed, m/s
                                # item 1: lane change command (-1 = change left, 0 = stay in lane, +1 = change right)

        """Applies the tactical guidance algorithm for one time step to convert vehicle observations into action commands."""

        # Update the target speed based on the local speed limit in this lane segment
        speed_limit = self.roadway.get_speed_limit(self.my_vehicle.lane_id, self.my_vehicle.p)
        tgt = speed_limit + self.speed_offset

        action = [None]*2
        action[0] = self._acc_speed_control(tgt, obs[ObsVec.FWD_DIST], obs[ObsVec.FWD_SPEED] + obs[ObsVec.SPEED_CUR])
        action[1] = LaneChange.STAY_IN_LANE

        return action


    def _acc_speed_control(self,
                           tgt_speed    : float,    #the speed we'd like to achieve, m/s
                           fwd_dist     : float,    #distance to the vehicle in front of us, m
                           fwd_speed    : float,    #speed of the vehicle in front of us, m/s
                          ) -> float:               #returns speed command, m/s

        """Applies a crude adaptive cruise control logic so that our vehicle attempts to follow it's target speed
            whenever possible, but slows to match the speed of a slower vehicle close in front of it to avoid a crash.
        """

        DISTANCE_OF_CONCERN     = 20.0 * self.my_vehicle.model.veh_length #following distance below which the vehicle needs to start slowing, m
        CRITICAL_DISTANCE       =  6.0 * self.my_vehicle.model.veh_length #following distance below which the vehicle needs to be matching speed, m

        speed_cmd = tgt_speed

        # If there is a forward vehicle close to us then
        if fwd_dist <= DISTANCE_OF_CONCERN:

            # Reduce our speed command gradually toward that vehicle's speed, to avoid a collision. Since there could be multiple
            # vehicles within the distance of concern, the limiter must account for the results of a previous iteration of this loop.
            # Note that the min value must be less than the forward vehicle's speed, otherwise we could follow it at exactly the critical
            # distance, then it suddenly slows, and our reaction lag will gradually eat iinto that critical distance until a crash becomes
            # inevitable.
            if fwd_speed < self.my_vehicle.cur_speed:
                f = (fwd_dist - CRITICAL_DISTANCE) / (DISTANCE_OF_CONCERN - CRITICAL_DISTANCE)
                speed_cmd = min(max(f*(tgt_speed - fwd_speed) + fwd_speed, 5.0), tgt_speed)

        return speed_cmd
