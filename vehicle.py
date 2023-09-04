from typing import Tuple, Dict, List

from constants import Constants
from vehicle_model import VehicleModel
from vehicle_controller import VehicleController
from hp_prng import HpPrng
from roadway_b import Roadway
from lane_change import LaneChange

class Vehicle:
    """Represents a single vehicle in the environment.  Contains the physical dynamics logic and is responsible for
        moving the vehicle to a new state in each time step.  Specific vehicle type characteristics are embodied in the
        _model_ and _controller_ members, so this class should never have to be sub-classed.
    """

    def __init__(self,
                    model       : VehicleModel, #describes the specific capabilities of the vehicle type
                    controller  : VehicleController, #provides the control algo that determines actions for this vehicle
                    prng        : HpPrng,       #the pseudo-random number generator to be used
                    roadway     : Roadway,      #the roadway geometry object for this scenario
                    learning    : bool = False, #is this vehicle going to be learning from the experience?
                    step_size   : float = 0.1,  #duration of a time step, s
                    debug       : int   = 0     #debug printing level
                ):

        # Read-only inputs
        self.model = model
        self.controller = controller
        self.prng = prng
        self.roadway = roadway
        self.learning = learning
        self.time_step_size = step_size
        self.debug = debug

        # State info that we need to maintain
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
        self.stopped = False                    #is this vehicle considered effectively stopped? (this is a latch)


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
        #print("///// Vehicle.reset: lane = {}, ddt = {:.3f}, p = {:.3f}".format(init_lane_id, ddt, self.p))

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
        self.stopped = False


    def advance_vehicle_spd(self,
                            new_speed_cmd   : float,        #the newly commanded speed, m/s
                            new_lc_cmd      : LaneChange,   #the newly commanded lane change action (enum)
                           ) -> Tuple[float, float, int, str]:

        """Advances a vehicle's motion for one time step according to the vehicle dynamics model.

            Returns: tuple of
                - new speed (float, m/s)
                - new P location (float, m)
                - new lane ID (int)
                - reason for running off-road (str)

            CAUTION: should not be overridden!
        """

        # Determine the current & previous effective accelerations
        cur_accel_cmd = (new_speed_cmd - self.cur_speed) / self.time_step_size
        #print("///// Vehicle.advance_vehicle_spd: new_speed_cmd = {:.1f}, cur_speed = {:.1f}, prev_speed = {:.1f}, cur_accel_cmd = {:.2f}, prev_accel = {:.2f}"
        #      .format(new_speed_cmd, cur_speed, prev_speed, cur_accel_cmd, prev_accel))
        return self.advance_vehicle_accel(cur_accel_cmd, new_lc_cmd)


    def advance_vehicle_accel(self,
                              new_accel_cmd : float,        #newest fwd accel command, m/s^2
                              new_lc_cmd    : LaneChange,   #newest lane change command (enum)
                             ) -> Tuple[float, float, int, str]:

        """Advances a vehicle's motion for one time step according to the vehicle dynamics model.

            Returns: tuple of
                - new speed (float, m/s)
                - new P location (float, m)
                - new lane ID (int)
                - reason for running off-road (str)
            Note that it also updates state member variables for the vehicle.

            CAUTION: should not be overridden!
        """

        reason = ""

        #
        #..........Update longitudinal motion
        #

        # Determine new longitudinal jerk, accel, speed & location of the vehicle
        new_jerk = min(max((new_accel_cmd - self.prev_accel) / self.time_step_size, -self.model.max_jerk), self.model.max_jerk)
        new_accel = min(max(self.prev_accel + self.time_step_size*new_jerk, -self.model.max_accel), self.model.max_accel)
        new_speed = min(max(self.cur_speed + self.time_step_size*new_accel, 0.0), Constants.MAX_SPEED) #vehicle won't start moving backwards
        new_p = max(self.p + self.time_step_size*(new_speed + 0.5*self.time_step_size*new_accel), 0.0)

        # Update the state variables for longitudinal motion
        self.p = new_p
        self.prev_speed = self.cur_speed
        self.cur_speed = new_speed
        self.prev_accel = new_accel

        # If vehicle has been stopped for several time steps, then declare the episode done as a failure
        if new_speed < 0.5:
            self.stopped_count += 1
            if self.stopped_count > 3:
                done = True
                self.stopped = True
                reason = "Vehicle is crawling to a stop"
                #print("/////+ step: {} step {}, vehicle stopped".format(self.rollout_id, self.total_steps))
        else:
            self.stopped_count = 0

        #
        #..........Update lateral motion
        #

        # Determine if we are beginning or continuing a lane change maneuver.
        # Accept a lane change command that lasts for several time steps or only one time step.  Once the first
        # command is received (when currently not in a lane change), then start the maneuver and ignore future
        # lane change commands until the underway maneuver is complete, which takes several time steps.
        # It's legal, but not desirable, to command opposite lane change directions in consecutive time steps.
        if new_lc_cmd != LaneChange.STAY_IN_LANE  or  self.lane_change_status != "none":
            if self.lane_change_status == "none": #count should always be 0 in this case, so initiate a new maneuver
                if new_lc_cmd == LaneChange.CHANGE_LEFT:
                    self.lane_change_status = "left"
                else:
                    self.lane_change_status = "right"
                self.lane_change_count = 1
                if self.debug > 0:
                    print("      *** New lane change maneuver initiated. new_lc_cmd = {}, status = {}"
                            .format(new_lc_cmd, self.lane_change_status))
            else: #once a lane change is underway, continue until complete, regardless of new commands
                self.lane_change_count += 1

        # Check that an adjoining lane is available in the direction commanded until maneuver is complete
        new_lane = int(self.lane_id)
        tgt_lane = new_lane
        if self.lane_change_count > 0:

            # If we are still in the original lane then
            if self.lane_change_count <= self.model.lc_half_steps:
                # Ensure that there is a lane to change into and get its ID
                tgt_lane = self.roadway.get_target_lane(int(self.lane_id), self.lane_change_status, new_p)
                if tgt_lane < 0:
                    self.off_road = True
                    reason = "Ran off road; illegal lane change"
                    if self.debug > 1:
                        print("      DONE!  illegal lane change commanded.")
                    #print("/////+ step: {} step {}, illegal lane change".format(self.rollout_id, self.total_steps))

                # Else, we are still going; if we are exactly half-way then change the current lane ID
                elif self.lane_change_count == self.model.lc_half_steps:
                    new_lane = tgt_lane

            # Else, we have already crossed the dividing line and are now mostly in the target lane
            else:
                coming_from = "left"
                if self.lane_change_status == "left":
                    coming_from = "right"
                # Ensure the lane we were coming from is still adjoining (since we still have 2 wheels there)
                prev_lane = self.roadway.get_target_lane(tgt_lane, coming_from, new_p)
                if prev_lane < 0: #the lane we're coming from ended before the lane change maneuver completed
                    self.off_road = True
                    reason = "Ran off road; lane change initiated too late"
                    if self.debug > 1:
                        print("      DONE!  original lane ended before lane change completed.")

        # If current lane change is complete, then reset its state and counter
        if self.lane_change_count >= self.model.lc_compl_steps:
            self.lane_change_status = "none"
            self.lane_change_count = 0

        # Update lateral state
        self.lane_id = new_lane

        #
        #..........Look for error conditions
        #

        # Since vehicles may not change lanes when it needs to, we need to take them out of action if they run off the end of a lane
        lane_end = self.roadway.get_lane_start_p(new_lane) + self.roadway.get_total_lane_length(new_lane)
        if new_p > lane_end:
            self.off_road = True
            reason = "Ran off end of a terminating lane."
            self.p = lane_end #don't want it to go past the end of lane, which causes graphics problems
        if self.debug > 1:
            print("      Vehicle in lane {} advanced with new_accel_cmd = {:.2f}. new_speed = {:.2f}, new_p = {:.2f}"
                    .format(self.lane_id, new_accel_cmd, new_speed, new_p))

        # Take the vehicle out of action if it went off-roading or came to a stop
        if self.off_road  or  self.stopped:
            self.active = False

        """
        # for debugging only, this whole section:
        if not self.training:
            new_rem, lid, la, lb, l_rem, rid, ra, rb, r_rem = self.roadway.get_current_lane_geom(new_lane, new_p)
            if self.lane_change_count == Constants.HALF_LANE_CHANGE_STEPS - 1:
                print("   ** LC next step: ego_p = {:.1f}, ego_rem = {:.1f}, lid = {}, la = {:.1f}, lb = {:.1f}, l_rem = {:.1f}".format(new_ego_p, new_rem, lid, la, lb, l_rem))
            elif self.lane_change_count == Constants.HALF_LANE_CHANGE_STEPS:
                print("   ** LC now: ego_p = {:.1f}, ego_rem = {:.1f}, rid = {}, ra = {:.1f}, rb = {:.1f}, r_rem = {:.1f}".format(new_ego_p, new_rem, rid, ra, rb, r_rem))
        """

        if self.debug > 0:
            print("      step: done lane change. underway = {}, new_ego_lane = {}, tgt_lane = {}, count = {}"
                    .format(self.lane_change_status, new_lane, tgt_lane, self.lane_change_count))

        return new_speed, new_p, new_lane, reason


    def print(self,
                tag     : object = None     #tag to identify the vehicle
             ):
        """Prints the attributes of this vehicle object."""

        print("       [{}]: active = {}, lane_id = {:2d}, p = {:.2f}, status = {:5s}, speed = {:.2f}" \
                .format(tag, self.active, self.lane_id, self.p, self.lane_change_status, self.cur_speed))
