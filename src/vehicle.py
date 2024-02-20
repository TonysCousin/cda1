from typing import Tuple, Dict, List

from constants import Constants
from vehicle_model import VehicleModel
from vehicle_guidance import VehicleGuidance
from hp_prng import HpPrng
from roadway import Roadway
from lane_change import LaneChange

class Vehicle:
    """Represents a single vehicle in the environment.  Contains the physical dynamics logic and is responsible for
        moving the vehicle to a new state in each time step.  Specific vehicle type characteristics are embodied in the
        .model and .guidance members, so this class should never have to be sub-classed.
    """

    def __init__(self,
                    model       : VehicleModel, #describes the specific capabilities of the vehicle type
                    guidance    : VehicleGuidance, #provides the guidance algos that determines actions for this vehicle
                    prng        : HpPrng,       #the pseudo-random number generator to be used
                    step_size   : float = 0.1,  #duration of a time step, s
                    debug       : int   = 0     #debug printing level
                ):

        # Read-only inputs
        self.model = model
        self.guidance = guidance
        self.prng = prng
        self.time_step_size = step_size
        self.debug = debug
        self.roadway = None #must be defined in reset() before any other method is called

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
        self.alert = False                      #flag can be used to alert the user to some situation; not used in routine logic.


    def reset(self,
              roadway           : Roadway,      #the roadway geometry model for this episode
              init_lane_id      : int   = -1,   #initial lane assignment; if -1 then will be randomized
              init_p            : float = None, #initial P coordinate, m; if None then will be randomized
              init_speed        : float = 0.0,  #initial speed of the vehicle, m/s
             ):

        """Reinitializes the vehicle for a new episode.  Before the vehicle can be used this method must be called with valid
            location data. However, it may be called with invalid data in order to clear out historical garbage from the object
            attributes, without intention to run the model yet.
            NOTE: this is not the same as the reset() method in an environment class, which is required to return observations.
            This method does not return anything.
        """

        # Determine roadway geometry and initial lane and P location
        self.roadway = roadway
        self.lane_id = init_lane_id
        self.p = init_p

        # Reset other stuff to start the episode with a clean slate
        self.cur_speed = init_speed
        self.prev_speed = init_speed
        self.prev_accel = 0.0
        self.lane_change_status = "none"
        self.lane_change_count = 0
        self.active = True
        self.crashed = False
        self.off_road = False
        self.stopped_count = 0
        self.stopped = False

        # If the location appears valid, then inform the model & guidance objects of the new location
        if roadway is not None  and  init_lane_id > -1  and  init_p is not None:
            self.model.reset(self.roadway)
            self.guidance.reset(self.roadway, self.lane_id, self.p)

        elif roadway is None:
            raise ValueError("///// ERROR in Vehicle.reset: roadway undefined. lane = {}, p = {}".format(init_lane_id, init_p))


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
                             ) -> str:

        """Advances a vehicle's motion for one time step according to the vehicle dynamics model.

            Returns: reason for flagging this episode as failed, if any (str)
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

        # NOTE that, even though the LC maneuver is assumed to be a smooth motion over many time steps, we use
        # the associate the vehicle with the lane ID that it is leaving for the entire duration of the maneuver,
        # only updating it to the target lane once the maneuver is complete. This is necessary to support other
        # classes or vehicle objects that need to observe this vehicle's lateral motion, so that they can simply
        # and consistently figure out the origin and target lanes, regardless of where it is in the maneuver.
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
        new_lane = int(self.lane_id) #TODO: still need the int call?
        tgt_lane = new_lane
        if self.lane_change_count > 0: #TODO: could probably do this just for == 1 to avoid repeated calls to get_target_lane

            # Ensure that there is a lane to change into and get its ID
            tgt_lane = self.roadway.get_target_lane(new_lane, self.lane_change_status, new_p)
            if tgt_lane < 0:
                self.off_road = True
                reason = "Ran off road; illegal lane change"
                if self.debug > 1:
                    print("      DONE!  illegal lane change commanded.")
                #print("/////+ step: {} step {}, illegal lane change".format(self.rollout_id, self.total_steps))

            if self.debug > 0:
                print("      Vehicle.advance_vehicle_accel: bottom of lane change. underway = {}, new_ego_lane = {}, tgt_lane = {}, count = {}"
                        .format(self.lane_change_status, new_lane, tgt_lane, self.lane_change_count))

        # If current lane change is complete, then reset its state and counter and update its new lane ID
        if self.lane_change_count >= self.model.lc_compl_steps  and  not self.off_road:
            self.lane_change_status = "none"
            self.lane_change_count = 0
            new_lane = tgt_lane

        # Update lateral state
        self.lane_id = new_lane
        assert self.lane_id >= 0, \
                "///// ERROR: Vehicle.advance_vehicle_accel: lane_id = {}, p = {}, LC count = {}, LC status = {}, active = {}, off_road = {}" \
                .format(self.lane_id, self.p, self.lane_change_count, self.lane_change_status, self.active, self.off_road)

        #
        #..........Look for error conditions
        #

        # Since a vehicle may not change lanes when it needs to, we need to take it out of action if it runs off the end of a lane
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
            lid, la, lb, rid, ra, rb = self.roadway.get_current_lane_geom(new_lane, new_p)
            if self.lane_change_count == Constants.HALF_LANE_CHANGE_STEPS - 1:
                print("   ** LC next step: ego_p = {:.1f}, lid = {}, la = {:.1f}, lb = {:.1f}".format(new_ego_p, lid, la, lb))
            elif self.lane_change_count == Constants.HALF_LANE_CHANGE_STEPS:
                print("   ** LC now: ego_p = {:.1f}, rid = {}, ra = {:.1f}, rb = {:.1f}".format(new_ego_p, rid, ra, rb))
        """

        return reason


    def print(self,
                tag     : object = None     #tag to identify the vehicle
             ):
        """Prints the attributes of this vehicle object."""

        if self.p is None:
            print("       [{}]: vehicle not initialized.".format(tag))
        else:
            print("       [{}]: active = {}, lane_id = {:2d}, p = {:.2f}, status = {:5s}, speed = {:.2f}"
                    .format(tag, self.active, self.lane_id, self.p, self.lane_change_status, self.cur_speed))
