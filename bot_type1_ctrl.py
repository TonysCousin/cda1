from vehicle_controller import VehicleController
import numpy as np

from obs_vec import ObsVec
from lane_change import LaneChange


class BotType1Ctrl(VehicleController):

    """Defines the control algorithm for the Type 1 bot vehicle, which has crude Adaptive Cruise Control (ACC)."""

    def __init__(self):
        super().__init__()
        print("///// BotType1Ctrl.__init__ entered.")


    def step(self,
             obs    : np.array, #vector of local observations available to the instantiating vehicle
            ) -> list:          #returns a list of action commands, such that
                                # item 0: desired speed, m/s
                                # item 1: lane change command (-1 = change left, 0 = stay in lane, +1 = change right)

        """Applies the control algorithm for one time step to convert vehicle observations into action commands."""

        #TODO: stub logic needs to be replaced
        action = []*2
        action[0] = self._acc_speed_control(obs[ObsVec.FWD_DIST], obs[ObsVec.FWD_SPEED])
        action[1] = LaneChange.STAY_IN_LANE

        """From cda0:
        #
        #..........Manage lane change for any neighbors in lane 2
        #

        # Loop through all active neighbors, looking for any that are in lane 2
        for n in range(1, self._num_vehicles):
            v = self.vehicles[n]
            if not v.active:
                continue

            if v.lane_id == 2:

                # If it is in the merge zone, then
                progress = v.p - self.roadway.get_lane_start_p(2)
                l2_length = self.roadway.get_total_lane_length(2)
                if progress > 0.7*l2_length:

                    # Randomly decide if it's time to do a lane change
                    if self.prng.random() < 0.05  or  l2_length - progress < 150.0:

                        # Look for a vehicle beside it in lane 1
                        safe = True
                        for j in range(self._num_vehicles):
                            if j == n:
                                continue
                            if self.vehicles[j].lane_id == 1  and  abs(self.vehicles[j].p - v.p) < 2.0*Constants.VEHICLE_LENGTH:
                                safe = False
                                break

                        # If it is safe to move, then just do an immediate lane reassignment (no multi-step process like ego does)
                        if safe:
                            v.lane_id = 1

                        # Else it is being blocked, then slow down a bit
                        else:
                            v.cur_speed *= 0.8

        """

        return action


    def _acc_speed_control(self,
                           fwd_dist : float,    #distance to the vehicle in front of us, m
                           fwd_speed: float,    #speed of the vehicle in front of us, m/s
                          ) -> float:       #returns speed command, m/s

        """Applies a crude adaptive cruise control logic so that our vehicle attempts to follow it's target speed
            whenever possible, but slows to match the speed of a slower vehicle close in front of it to avoid a crash.
        """

        DISTANCE_OF_CONCERN     = 8.0 * self.my_vehicle.model.length #following distance below which the vehicle needs to start slowing, m
        CRITICAL_DISTANCE       = 2.0 * self.my_vehicle.model.length #following distance below which the vehicle needs to be matching speed, m

        speed_cmd = self.my_vehicle.tgt_speed

        # If there is a forward vehicle close to us then
        if fwd_dist <= DISTANCE_OF_CONCERN:

            # Reduce our speed command gradually toward that vehicle's speed, to avoid a collision. Since there could be multiple
            # vehicles within the distance of concern, the limiter must account for the results of a previous iteration of this loop.
            if fwd_speed < self.my_vehicle.cur_speed:
                f = (fwd_dist - CRITICAL_DISTANCE) / (DISTANCE_OF_CONCERN - CRITICAL_DISTANCE)
                speed_cmd = min(max(f*(self.my_vehicle.tgt_speed - fwd_speed) + fwd_speed, fwd_speed), self.my_vehicle.tgt_speed)

        return speed_cmd
