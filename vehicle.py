from constants import Constants
from typing import Tuple, Dict, List

class Vehicle:
    """Represents a single vehicle on the Roadway."""

    def __init__(self,
                    step_size   : float,    #duration of a time step, s
                    max_jerk    : float,    #max allowed jerk, m/s^3
                    tgt_speed   : float = 0.0,  #the (constant) target speed that the vehicle will try to maintain, m/s
                    cur_speed   : float = 0.0,  #vehicle's current speed, m/s
                    prev_speed  : float = 0.0,  #vehicle's speed in the previous time step, m/s
                    debug       : int = 0   #debug printing level
                ):

        self.time_step_size = step_size
        self.max_jerk = max_jerk
        self.tgt_speed = tgt_speed
        self.cur_speed = cur_speed
        self.prev_speed = prev_speed
        self.debug = debug

        self.lane_id = -1                   #-1 is an illegal value
        self.p = 0.0                        #P coordinate of vehicle center in parametric frame, m
        self.prev_accel = 0.0               #Forward actual acceleration in previous time step, m/s^2
        self.lane_change_status = "none"    #Initialized to no lane change underway
        self.active = True                  #is the vehicle an active part of the scenario? If false, it is invisible
        self.crashed = False                #has this vehicle crashed into another?


    def advance_vehicle_spd(self,
                            new_speed_cmd   : float,    #the newly commanded speed, m/s
                           ) -> Tuple[float, float]:
        """Advances a vehicle's forward motion for one time step according to the vehicle dynamics model.
            Note that this does not consider lateral motion, which needs to be handled elsewhere.

            Returns: tuple of (new speed (m/s), new P location (m))
        """

        # Determine the current & previous effective accelerations
        cur_accel_cmd = (new_speed_cmd - self.cur_speed) / self.time_step_size
        #print("///// Vehicle.advance_vehicle_spd: new_speed_cmd = {:.1f}, cur_speed = {:.1f}, prev_speed = {:.1f}, cur_accel_cmd = {:.2f}, prev_accel = {:.2f}"
        #      .format(new_speed_cmd, cur_speed, prev_speed, cur_accel_cmd, prev_accel))
        return self.advance_vehicle_accel(cur_accel_cmd)


    def advance_vehicle_accel(self,
                              new_accel_cmd   : float,    #newest fwd accel command, m/s^2
                             ) -> Tuple[float, float]:
        """Advances a vehicle's forward motion for one time step according to the vehicle dynamics model.
            Note that this does not consider lateral motion, which needs to be handled elsewhere.

            Returns: tuple of (new speed (m/s), new P location (m))
        """

        # Determine new jerk, accel, speed & location of the vehicle
        new_jerk = min(max((new_accel_cmd - self.prev_accel) / self.time_step_size, -self.max_jerk), self.max_jerk)
        new_accel = min(max(self.prev_accel + self.time_step_size*new_jerk, -Constants.MAX_ACCEL), Constants.MAX_ACCEL)
        new_speed = min(max(self.cur_speed + self.time_step_size*new_accel, 0.0), Constants.MAX_SPEED) #vehicle won't start moving backwards
        new_p = max(self.p + self.time_step_size*(new_speed + 0.5*self.time_step_size*new_accel), 0.0)

        # Update the state variables
        self.p = new_p
        self.prev_speed = self.cur_speed
        self.cur_speed = new_speed
        self.prev_accel = new_accel

        return new_speed, new_p


    def print(self,
                tag     : object = None     #tag to identify the vehicle
             ):
        """Prints the attributes of this vehicle object."""

        print("       [{}]: active = {:5}, lane_id = {:2d}, p = {:.2f}, status = {:5s}, speed = {:.2f}" \
                .format(tag, self.active, self.lane_id, self.p, self.lane_change_status, self.cur_speed))
