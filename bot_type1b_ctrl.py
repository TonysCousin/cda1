from typing import List
import numpy as np

from constants import Constants
from obs_vec import ObsVec
from hp_prng import HpPrng
from roadway_b import Roadway
from lane_change import LaneChange
from vehicle_controller import VehicleController


class BotType1bCtrl(VehicleController):

    """Defines the control algorithm for the Type 1 bot vehicle, which at a small, constant, random offset to the
        posted speed limit, but uses crude Adaptive Cruise Control (ACC).
    """

    def __init__(self,
                 prng       : HpPrng,
                 roadway    : Roadway,
                 targets    : List,
                ):

        super().__init__(prng, roadway, targets)

        # Pick an offset from the posted speed limit that will define the target speed
        self.speed_offset = (self.prng.random() - 0.5) * 0.2*Constants.MAX_SPEED #gives +/- 10%

        # Define an empty variable to hold the ID of the target destination
        self.target_id = None

        # Other member initializations
        self.prev_lc_cmd = LaneChange.STAY_IN_LANE #lane change command from the previous time step


    def reset(self,
              init_lane     : int,      #the lane the vehicle is starting in
              init_p        : float,    #vehicle's initial P coordinate, m
             ):

        """Resets the target that the vehicle will navigate to, which dictates its lane change behavior."""

        super().reset(init_lane, init_p)
        self.target_id = None

        # Choose one of the targets to drive to, but verify that it is reachable first
        ctr = 0
        while ctr < 10:
            t = int(self.prng.random() * len(self.targets))
            if self.targets[t].is_reachable_from(init_lane, init_p):
                self.target_id = t
                break
            ctr += 1

        # If a random search doesn't find anything suitable, look at every available target
        if ctr >= 10:
            found = False
            for t in range(len(self.targets)):
                if self.targets[t].is_reachable_from(init_lane, init_p):
                    self.target_id = t
                    found = True
                    break
            if not found:
                raise ValueError("///// WARNING: BotType1bCtrl.reset could not find a reachable target from lane {}, p = {:.1f}".format(init_lane, init_p))

        print("***** BotType1bCtrl reset to lane {}, p = {:.1f}, and chose target {}".format(init_lane, init_p, self.target_id)) #TODO test


    def step(self,
             obs    : np.array, #vector of local observations available to the instantiating vehicle
            ) -> List:          #returns a list of float action commands, such that
                                # item 0: desired speed, m/s
                                # item 1: lane change command (corresponds to type LaneChange)

        """Applies the control algorithm for one time step to convert vehicle observations into action commands."""

        # Update the target speed based on the local speed limit in this lane segment
        speed_limit = self.roadway.get_speed_limit(self.my_vehicle.lane_id, self.my_vehicle.p)
        tgt = min(speed_limit + self.speed_offset, Constants.MAX_SPEED)
        self.my_vehicle.tgt_speed = tgt         #TODO: does this need to be stored in Vehicle?

        # Define the speed command action using ACC
        action = [None]*2
        action[0] = self._acc_speed_control(tgt, obs[ObsVec.FWD_DIST], obs[ObsVec.FWD_SPEED])

        #
        #..........Determine lane change command
        #

        # If the target destination is in the current lane then no lane change
        cur_lane = self.my_vehicle.lane_id
        cur_p = self.my_vehicle.p
        tgt_lane = self.targets[self.target_id].lane_id
        cmd = LaneChange.STAY_IN_LANE #set this as the default action in case decision logic below breaks down

        # If reaching the target requires a lane change and one has not yet been initiated (this is a quick and kludgy test
        # that will result in a LC command being issued every second time step until one is no longer needed; this is an
        # attempt to avoid using vehicle info that would not be available if this were a NN).
        if cur_lane != tgt_lane  and  self.prev_lc_cmd == LaneChange.STAY_IN_LANE:

            # Get the roadway geometry at our current location
            #TODO: ideally, this should only look at sensor data; change to use center lane border obs
            _, lid, la, lb, _, rid, ra, rb, _ = self.roadway.get_current_lane_geom(cur_lane, cur_p)

            #TODO: consider slowing the vehicle to get behind neighbors if they continue to obstruct the LC maneuver (see cda0 logic)

            # If the target is to our left (lower ID) then
            if tgt_lane < cur_lane:
                print("***   Ctrl.step left: obs left occupied = ", obs[ObsVec.LEFT_OCCUPIED])

                # If we are within the legal zone to change lanes to the next one, then
                if lid >= 0  and  la <= 0.0  and  lb >= 0.0:

                    # If the sensors detect that the zones immediately to the left are unoccupied then command the change.
                    # Randomize the initiation of the command, since most crashes happen in the first couple seconds of a scenario, when
                    # all vehicles are trying to adjust their lateral position at the same time.
                    if obs[ObsVec.LEFT_OCCUPIED] < 0.5  and  self.prng.random() < 0.2:
                        cmd = LaneChange.CHANGE_LEFT

            # Else (target must be to our right)
            else:
                print("***   Ctrl.step right: obs right occupied = ", obs[ObsVec.RIGHT_OCCUPIED])

                # If we are within the legal zone to change lanes to the next one, then
                if rid >= 0  and ra <= 0.0  and rb >= 0.0:

                    # If the sensors detect that the zones immediately to the left are unoccupied then command the change
                    if obs[ObsVec.RIGHT_OCCUPIED] < 0.5  and  self.prng.random() < 0.2:
                        cmd = LaneChange.CHANGE_RIGHT

        action[1] = cmd
        self.prev_lc_cmd = cmd

        return action


    def _acc_speed_control(self,
                           tgt_speed    : float,    #the speed we'd like to achieve, m/s
                           fwd_dist     : float,    #distance to the vehicle in front of us, m
                           fwd_speed    : float,    #speed of the vehicle in front of us, m/s
                          ) -> float:               #returns speed command, m/s

        """Applies a crude adaptive cruise control logic so that our vehicle attempts to follow it's target speed
            whenever possible, but slows to match the speed of a slower vehicle close in front of it to avoid a crash.
        """

        DISTANCE_OF_CONCERN     = 12.0 * self.my_vehicle.model.veh_length #following distance below which the vehicle needs to start slowing, m
        CRITICAL_DISTANCE       =  3.0 * self.my_vehicle.model.veh_length #following distance below which the vehicle needs to be matching speed, m

        speed_cmd = tgt_speed

        # If there is a forward vehicle close to us then
        if fwd_dist <= DISTANCE_OF_CONCERN:

            # Reduce our speed command gradually toward that vehicle's speed, to avoid a collision. Since there could be multiple
            # vehicles within the distance of concern, the limiter must account for the results of a previous iteration of this loop.
            if fwd_speed < self.my_vehicle.cur_speed:
                f = (fwd_dist - CRITICAL_DISTANCE) / (DISTANCE_OF_CONCERN - CRITICAL_DISTANCE)
                speed_cmd = min(max(f*(tgt_speed - fwd_speed) + fwd_speed, fwd_speed), tgt_speed)

        return speed_cmd
