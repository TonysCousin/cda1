from abc import ABC, abstractmethod
from typing import Tuple, Dict, List
import math

from constants import Constants
from vehicle_model import VehicleModel
from vehicle_controller import VehicleController
from hp_prng import HpPrng
from roadway_b import Roadway
from lane_change import LaneChange

class Vehicle(ABC):
    """Abstract base class that represents a single vehicle in the environment.  Specific vehicle types must be derived from this."""

    def __init__(self,
                    model       : VehicleModel, #describes the specific capabilities of the vehicle type
                    controller  : VehicleController, #provides the control algo that determines actions for this vehicle
                    prng        : HpPrng,       #the pseudo-random number generator to be used
                    roadway     : Roadway,      #the roadway geometry object for this scenario
                    learning    : bool = False, #is this vehicle going to be learning from the experience?
                    step_size   : float = 0.1,  #duration of a time step, s
                    debug       : int   = 0     #debug printing level
                ):

        self.model = model
        self.controller = controller
        self.prng = prng
        self.roadway = roadway
        self.learning = learning
        self.time_step_size = step_size
        self.debug = debug

        self.cur_speed = 0.0                    #current forward speed, m/s
        self.prev_speed = 0.0                   #forward speed in previous time step, m/s
        self.lane_id = -1                       #vehicle's current lane ID; -1 is an illegal value
        self.p = 0.0                            #P coordinate of vehicle center in parametric frame, m
        self.prev_accel = 0.0                   #forward actual acceleration in previous time step, m/s^2
        self.lane_change_status = "none"        #initialized to no lane change underway; legal values are "left", "right", "none"
        self.lane_change_count = 0              #num consecutive time steps since a lane change was begun; 0 indicates no lc maneuver underway
        self.active = True                      #is the vehicle an active part of the scenario? If false, it cannot move and cannot be reactivated until reset
        self.crashed = False                    #has this vehicle crashed into another?
        self.off_road = False                   #has this vehicle driven off-road?
        self.stopped_count = 0                  #num consecutive time steps that the vehicle's speed is very close to 0


    def reset(self,
              init_lane_id      : int   = -1,   #initial lane assignment; if -1 then will be randomized
              init_ddt          : float = None, #initial dist downtrack from chosen lane start, m; if None then will be randomized
              init_speed        : float = 0.0,  #initial speed of the vehicle, m/s
             ):

        """Reinitializes the vehicle for a new episode.
            NOTE: this is not the same as the reset() method in an environment class, which is required to return observations.
            This method does not return anything.
        """

        self.lane_id = init_lane_id
        if init_lane_id == -1:
            self.lane_id = int(self.prng.random()*self.roadway.NUM_LANES)

        lane_len = self.roadway.get_total_lane_length(self.lane_id)
        ddt = init_ddt
        if init_ddt is None:
            ddt = self.prng.random()*(lane_len - 50.0)
        self.p = self.roadway.get_lane_start_p(self.lane_id) + ddt

        self.cur_speed = init_speed
        self.prev_speed = init_speed

        # Reset other stuff to start the episode with a clean slate
        self.prev_accel = 0.0
        self.lane_change_status = "none"
        self.lane_change_count = 0
        self.active = True
        self.crashed = False
        self.off_road = False
        self.stopped_count = 0


    def advance_vehicle_spd(self,
                            new_speed_cmd   : float,    #the newly commanded speed, m/s
                           ) -> Tuple[float, float]:

        """Advances a vehicle's forward motion for one time step according to the vehicle dynamics model.
            Note that this does not consider lateral motion, which needs to be handled elsewhere.

            Returns: tuple of (new speed (m/s), new P location (m))

            CAUTION: should not be overridden!
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

            CAUTION: should not be overridden!
        """

        # Determine new jerk, accel, speed & location of the vehicle
        new_jerk = min(max((new_accel_cmd - self.prev_accel) / self.time_step_size, -self.model.max_jerk), self.model.max_jerk)
        new_accel = min(max(self.prev_accel + self.time_step_size*new_jerk, -self.model.max_accel), self.model.max_accel)
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

        print("       [{}]: active = {:5b}, lane_id = {:2d}, p = {:.2f}, status = {:5s}, speed = {:.2f}" \
                .format(tag, self.active, self.lane_id, self.p, self.lane_change_status, self.cur_speed))
