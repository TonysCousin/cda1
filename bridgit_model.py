import math
import numpy as np

from constants import Constants
from obs_vec import ObsVec
from roadway_b import Roadway, PavementType
from vehicle_model import VehicleModel

class BridgitModel(VehicleModel):

    """Realizes a concrete model for the Bridgit RL agent vehicle."""

    def __init__(self,
                 roadway    : Roadway,      #roadway geometry model
                 max_jerk   : float = 3.0,  #forward & backward, m/s^3
                 max_accel  : float = 2.0,  #forward & backward, m/s^2
                 length     : float = 5.0,  #length of the vehicle, m
                 lc_duration: float = 3.0,  #time to complete a lane change, sec; must result in an even number when divided by time step
                 time_step  : float = 0.1,  #duration of a single time step, sec
                ):

        super().__init__(roadway, max_jerk = max_jerk, max_accel = max_accel, length = length, lc_duration = lc_duration, time_step = time_step)


    def get_obs_vector(self,
                       my_id    : int,      #ID of this vehicle (its index into the vehicles list)
                       vehicles : list,     #list of all Vehicles in the scenario
                       actions  : list,     #list of action commands for this vehicle
                       obs      : np.array, #array of observations for this vehicle from the previous time step
                      ) -> np.array:

        """Produces the observation vector for this vehicle object if it is active. An inactive vehicle produces all 0s. See the ObsVec
            class description for details on the obs vector and its sensor zone construction.

            CAUTION: the returned observation vector is at actual world scale and needs to be preprocessed before going into a NN!

            NOTE: we use Vehicle objects here, but there is no import statment for that type in this class or in the base class, since it
            creates a circular reference during construction. But Python seems to give us full knowledge of those objects' structures
            anyway.
        """

        # Save key values from previous time step, then clear the obs vector to prepare for new values
        prev_speed = obs[ObsVec.SPEED_CUR]
        prev_speed_cmd = obs[ObsVec.SPEED_CMD]
        prev_lc_cmd = obs[ObsVec.LC_CMD]
        steps_since_lc = obs[ObsVec.STEPS_SINCE_LN_CHG]
        lane_change_des = obs[ObsVec.DESIRABILITY_LEFT : ObsVec.DESIRABILITY_RIGHT+1]
        obs = np.zeros(ObsVec.OBS_SIZE, dtype = float)
        me = vehicles[my_id]

        # If this vehicle is inactive, keep collecting data; doing so may mean an extra time step of work, but it prevents the storage
        # of an all-zero obs vector for a key vehicle in its final time step.

        # Build the common parts of the obs vector
        obs[ObsVec.SPEED_CMD_PREV] = prev_speed_cmd
        obs[ObsVec.SPEED_CMD] = actions[0]
        obs[ObsVec.LC_CMD_PREV] = prev_lc_cmd
        obs[ObsVec.LC_CMD] = actions[1]
        obs[ObsVec.SPEED_PREV] = prev_speed
        obs[ObsVec.SPEED_CUR] = me.cur_speed
        obs[ObsVec.LOCAL_SPD_LIMIT] = self.roadway.get_speed_limit(me.lane_id, me.p)
        steps_since_lc += 1
        if steps_since_lc > Constants.MAX_STEPS_SINCE_LC:
            steps_since_lc = Constants.MAX_STEPS_SINCE_LC
        if me.lane_change_count >= self.lc_compl_steps - 1: #a new LC maneuver has just completed, so a new mvr can now be considered
            steps_since_lc = self.lc_compl_steps - 1
        obs[ObsVec.STEPS_SINCE_LN_CHG] = steps_since_lc

        # Put the lane change desired values back into place, since that planning doesn't happen every time step
        obs[ObsVec.DESIRABILITY_LEFT : ObsVec.DESIRABILITY_RIGHT+1] = lane_change_des

        # Skip a few here that are used for bots or reserved for future

        # Find the host vehicle in the roadway (parametric frame)
        # NOTE: allow the case where the host is not on any pavement - it could be sitting in the grass with sensors on watching the world go by
        host_lane_id = me.lane_id
        host_p = me.p

        #
        #..........Determine pavement observations in each zone
        #

        # Loop through each of the columns of zones; start with the far left, proceeding to the right
        half_zone = 0.5*ObsVec.OBS_ZONE_LENGTH #half the lenght of a zone, m
        grid_rear_edge = -ObsVec.OBS_ZONE_LENGTH*(ObsVec.ZONES_BEHIND + 0.5) #fwd dist from center of host vehicle to rear of rear-most zone
        for col in range(ObsVec.NUM_COLUMNS):
            col_lane = host_lane_id + col - 2 #lane ID represented by this column
            col_base_pvmt = ObsVec.BASE_PVMT_TYPE + col*ObsVec.NUM_ROWS

            # Initialize all zones as not driveable
            for z in range(ObsVec.NUM_ROWS):
                obs[col_base_pvmt + z] = -1.0

            # If the lane that coincides with this column does not exist then skip to next lane
            if col_lane < 0  or  col_lane >= self.roadway.NUM_LANES:
                continue

            # Get extent of the lane that coincides with this column
            lane_begin_p = self.roadway.get_lane_start_p(col_lane)
            lane_end_p = self.roadway.get_total_lane_length(col_lane) + lane_begin_p

            # Loop through each zone in this column, rear to front
            for zone_id in range(ObsVec.NUM_ROWS):

                # Determine the zone's boundaries & center (P coordinate)
                zone_rear_p = grid_rear_edge + zone_id*ObsVec.OBS_ZONE_LENGTH + host_p
                zone_ctr_p = zone_rear_p + half_zone

                # Indicate lane existence in this zone, and if it exists, its pavement type & speed limit
                if lane_begin_p <= zone_ctr_p <= lane_end_p:
                    pvmt_idx = col_base_pvmt + zone_id #index of the pavement type data element for this zone
                    obs[pvmt_idx] = 1.0
                    if self.roadway.get_pavement_type(col_lane, zone_ctr_p) == PavementType.EXIT_RAMP:
                        obs[pvmt_idx] = 0.0

                    col_base_sl = ObsVec.BASE_SPD_LIMIT + col*ObsVec.NUM_ROWS
                    sl_idx = col_base_sl + zone_id
                    obs[sl_idx] = self.roadway.get_speed_limit(col_lane, zone_ctr_p) / Constants.MAX_SPEED

        # Get lane connectivity details for the center lane (all distances are downtrack from the host location)
        try:
            _, lid, la, lb, _, rid, ra, rb, _ = self.roadway.get_current_lane_geom(host_lane_id, host_p)
        except AssertionError as e:
            print("///// Trapped AssertionError in BridgitModel.get_obs_vector: ", e)
            raise e

        la_p = math.inf
        lb_p = -math.inf
        if lid >= 0:
            la_p = la + host_p
            lb_p = lb + host_p
        ra_p = math.inf
        rb_p = -math.inf
        if rid >= 0:
            ra_p = ra + host_p
            rb_p = rb + host_p

        # Loop through the center column, boundary regions
        zone_min = 0
        zone_max = 2
        for bdry_region in range(ObsVec.NUM_BDRY_REGIONS):

            # Define the zone IDs covered by this region. (see ObsVec class description). Note that the loop index
            # covers one more than the number of forward sensor zones; the first zone addressed here is the host's
            # own zone.
            if bdry_region == 1:
                zone_min = 3
                zone_max = 9
            elif bdry_region == 2:
                zone_min = 10
                zone_max = 20

            # Initialize the boundary to passable
            left_idx = ObsVec.BASE_LEFT_CTR_BDRY + bdry_region
            right_idx = ObsVec.BASE_RIGHT_CTR_BDRY + bdry_region
            obs[left_idx] = 1.0
            obs[right_idx] = 1.0

            # Loop through each zone in this region
            for zone_id in range(zone_min, zone_max+1):

                # Determine the zone's center P
                zone_rear_p = host_p - half_zone + zone_id*ObsVec.OBS_ZONE_LENGTH
                zone_ctr_p = zone_rear_p + half_zone

                # If there is no reachable lane to the side of interest, then mark this region impassable. We are only
                # concerned with physical connectivity here, not painted lines.
                if zone_ctr_p <= la_p  or  zone_ctr_p >= lb_p:
                    obs[left_idx] = -1.0
                if zone_ctr_p <= ra_p  or  zone_ctr_p >= rb_p:
                    obs[right_idx] = -1.0

        #
        #..........Map vehicles to zones for those that are within the grid
        #

        # Loop through all active vehicles that are not the host
        for v_idx in range(len(vehicles)):
            if v_idx == my_id  or  not vehicles[v_idx].active:
                continue

            nv = vehicles[v_idx]

            # Determine if it's in the bounding box of our sensor grid. If not, contine to the next vehicle.
            if abs(nv.lane_id - host_lane_id) > 2:
                continue

            grid_front_edge = (ObsVec.ZONES_FORWARD + 0.5)*ObsVec.OBS_ZONE_LENGTH #distance, not P coord
            ddt_ctr = nv.p - host_p
            half_length = 0.5*nv.model.veh_length
            n_rear = ddt_ctr - half_length
            n_front = ddt_ctr + half_length

            if n_front < grid_rear_edge  or  n_rear > grid_front_edge:
                continue

            #TODO: represent nv changing lanes & covering two of them (look at its lane change count)

            # At this point we have a vehicle that is somewhere on the grid. Now to figure out which zone(s) it occupies.
            z_num_rear = None #num zones in front of the base zone where the rear of the neighbor exists
            z_num_front = None #num zones in front of the base zone where the front of the neighbor exists

            # Find the column and set its base indices
            lane_diff = nv.lane_id - host_lane_id
            base_occ_idx = ObsVec.BASE_OCCUPANCY + (lane_diff + 2)*ObsVec.NUM_ROWS
            base_rs_idx = ObsVec.BASE_REL_SPEED + (lane_diff + 2)*ObsVec.NUM_ROWS

            # Get the zone numbers for both ends of the vehicle, limited to the valid range of zones in a column
            z_num_rear = min(max(math.floor((n_rear - grid_rear_edge + 0.001)/ObsVec.OBS_ZONE_LENGTH), 0), ObsVec.NUM_ROWS - 1)
            z_num_front = min(max(math.floor((n_front - grid_rear_edge - 0.001)/ObsVec.OBS_ZONE_LENGTH), z_num_rear), ObsVec.NUM_ROWS - 1)

            assert z_num_rear <= z_num_front, \
                    "///// BridgitModel.get_obs_vector: host lane = {}, host_p = {:.2f}; vehicle {} in lane {}, p = {:.2f} has has z_num_rear = {}, " \
                    .format(host_lane_id, host_p, v_idx, nv.lane_id, nv.p, z_num_rear) + \
                    "z_num_front = {}. n_rear = {:.2f}, n_front = {:.2f}".format(z_num_front, n_rear, n_front)


            # Find the relative speed, normalized to [-1, 1]
            rel_speed = (nv.cur_speed - me.cur_speed)/Constants.MAX_SPEED

            # Loop through all the zones in the column that are at least partially occupied by this neighbor
            for z in range(z_num_rear, z_num_front+1):

                # Compute the zone index in the obs vector, and set the occupied flag and relative speed for that zone
                obs[base_occ_idx + z] = 1.0 #occupied
                obs[base_rs_idx] = rel_speed

        return obs
