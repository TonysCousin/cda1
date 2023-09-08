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
                      ) -> np.array:

        """Produces the observation vector for this vehicle object if it is active. An inactive vehicle produces all 0s. See the ObsVec
            class description for details on the obs vector and its sensor zone construction.

            CAUTION: the returned observation vector is at actual world scale and needs to be
                     preprocessed before going into a NN!

            NOTE: we use Vehicle objects here, but there is no import statment for that type in this class or in the base class, since it
            creates a circular reference during construction. But Python seems to give us full knowledge of those objects' structures
            anyway.
        """

        obs = np.zeros(ObsVec.OBS_SIZE, dtype = float)

        # If this vehicle is inactive, then stop now
        me = vehicles[my_id]
        if not me.active:
            return obs

        # Build the common parts of the obs vector
        obs[ObsVec.SPEED_CMD_PREV] = obs[ObsVec.SPEED_CMD]
        obs[ObsVec.SPEED_CMD] = actions[0]
        obs[ObsVec.LC_CMD_PREV] = obs[ObsVec.LC_CMD_PREV]
        obs[ObsVec.LC_CMD] = actions[1]
        obs[ObsVec.STEPS_SINCE_LN_CHG] = me.lane_change_count
        obs[ObsVec.SPEED_PREV] = obs[ObsVec.SPEED_CUR]
        obs[ObsVec.SPEED_CUR] = me.cur_speed

        # Skip a few here that are used for bots or reserved for future

        # Find the host vehicle in the roadway (parametric frame)
        # NOTE: allow the case where the host is not on any pavement - it could be sitting in the grass with sensors on watching the world go by
        host_lane_id = me.lane_id
        host_p = me.p

        #
        #..........Determine pavement observations in each zone
        #

        # Loop through each of the off-center columns of zones; start with the far left (LL), proceeding to the right
        # (L then R then RR)
        zones_per_column = ObsVec.ZONES_BEHIND + 1 + ObsVec.ZONES_FORWARD
        elements_per_column = ObsVec.NORM_ELEMENTS*zones_per_column
        half_zone = 0.5*ObsVec.OBS_ZONE_LENGTH #half the lenght of a zone, m
        grid_rear_edge = -ObsVec.OBS_ZONE_LENGTH*(ObsVec.ZONES_BEHIND + 0.5) #fwd dist from center of host vehicle to rear of rear-most zone
        for col in range(4):
            col_lane = host_lane_id + col - 2 #index to the lane ID in this column
            if col > 1: #now on right side of vehicle, so skip host's lane
                col_lane += 1
            col_base = ObsVec.BASE_LL + col*elements_per_column

            # Initialize all zones as not driveable
            for z in range(zones_per_column):
                obs[col_base + z*ObsVec.NORM_ELEMENTS + 0] = -1.0 #drivable flag is the 0th element in a zone

            # If the lane that coincides with this column does not exist then skip to next lane
            if col_lane < 0  or  col_lane >= self.roadway.NUM_LANES:
                continue

            # Get extent of the lane that coincides with this column
            lane_begin_p = self.roadway.get_lane_start_p(col_lane)
            lane_end_p = self.roadway.get_total_lane_length(col_lane) + lane_begin_p

            # Loop through each zone in this column, rear to front
            for zone_id in range(zones_per_column):
                z_idx = col_base + zone_id*ObsVec.NORM_ELEMENTS

                # Determine the zone's boundaries & center - P coordinate
                zone_rear_p = grid_rear_edge + zone_id*ObsVec.OBS_ZONE_LENGTH + host_p
                zone_ctr_p = zone_rear_p + half_zone

                # Indicate lane existence in this zone, and if it exists, its pavement type & speed limit
                if lane_begin_p <= zone_ctr_p <= lane_end_p:
                    obs[z_idx + 0] = 1.0
                    if self.roadway.get_pavement_type(col_lane, zone_ctr_p) == PavementType.EXIT_RAMP:
                        obs[z_idx + 0] = 0.0
                    obs[z_idx + 1] = self.roadway.get_speed_limit(col_lane, zone_ctr_p) / Constants.MAX_SPEED

        # Set up to handle zones in the center column - first behind the host
        col_base = ObsVec.BASE_CTR_REAR

        # Get extent of the lane that coincides with this column
        lane_begin_p = self.roadway.get_lane_start_p(host_lane_id)
        lane_end_p = self.roadway.get_total_lane_length(host_lane_id) + lane_begin_p

        # Loop through each zone in the rear of this column, rear to front
        for zone_id in range(ObsVec.ZONES_BEHIND):
            z_idx = col_base + zone_id*ObsVec.NORM_ELEMENTS

            # Determine the zone's center P
            zone_rear_p = grid_rear_edge + zone_id*ObsVec.OBS_ZONE_LENGTH + host_p
            zone_ctr_p = zone_rear_p + half_zone

            # Indicate lane existence in this zone, and if it exists, its pavement type & speed limit
            obs[z_idx + 0] = -1.0 #default to non-existent pavement
            if lane_begin_p <= zone_ctr_p <= lane_end_p:
                obs[z_idx + 0] = 1.0
                if self.roadway.get_pavement_type(host_lane_id, zone_ctr_p) == PavementType.EXIT_RAMP:
                    obs[z_idx + 0] = 0.0
                obs[z_idx + 1] = self.roadway.get_speed_limit(host_lane_id, zone_ctr_p) / Constants.MAX_SPEED

        # Get lane connectivity details for the center lane (all distances are downtrack from the host location)
        _, lid, la, lb, _, rid, ra, rb, _ = self.roadway.get_current_lane_geom(host_lane_id, host_p)
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

        # Loop through each zone in the front of this column, rear to front
        col_base = ObsVec.BASE_CTR_FRONT
        for zone_id in range(ObsVec.ZONES_FORWARD):
            z_idx = col_base + zone_id*ObsVec.CTR_ELEMENTS

            # Determine the zone's center P
            zone_rear_p = half_zone + zone_id*ObsVec.OBS_ZONE_LENGTH + host_p
            zone_ctr_p = zone_rear_p + half_zone

            # Indicate lane existence in this zone, and if it exists, its pavement type & speed limit
            obs[z_idx + 0] = -1.0 #default to non-existent pavement
            if lane_begin_p <= zone_ctr_p <= lane_end_p:
                obs[z_idx + 0] = 1.0
                if self.roadway.get_pavement_type(host_lane_id, zone_ctr_p) == PavementType.EXIT_RAMP:
                    obs[z_idx + 0] = 0.0
                obs[z_idx + 1] = self.roadway.get_speed_limit(host_lane_id, zone_ctr_p) / Constants.MAX_SPEED

            # In the forward center column we also need to indicate lane boundary type. If there is a reachable lane
            # to the side of interest, consider the boundary passable. We are only concerned with physical connectivity
            # here, not painted lines.
            obs[z_idx + 4] = -1.0 #default to unpassable - left side
            if la_p < zone_ctr_p < lb_p:
                obs[z_idx + 4] = 1.0
            obs[z_idx + 5] = -1.0 #default to unpassable - right side
            if ra_p < zone_ctr_p < rb_p:
                obs[z_idx + 5] = 1.0

        #
        #..........Map vehicles to zones for those that are within the grid
        #

        # Loop through all active vehicles that are not the host
        for v_idx in range(len(vehicles)):
            if v_idx == my_id  or  not vehicles[v_idx].active:
                continue

            nv = vehicles[v_idx]

            # Determine if it's in the bounding box of our observation region. If not, contine to the next vehicle.
            if abs(nv.lane_id - host_lane_id) > 2:
                continue

            grid_front_edge = (ObsVec.ZONES_FORWARD + 0.5)*ObsVec.OBS_ZONE_LENGTH #distance, not P coord
            ddt_ctr = nv.p - host_p
            half_length = 0.5*nv.model.veh_length
            n_rear = ddt_ctr - half_length
            n_front = ddt_ctr + half_length

            if n_front < grid_rear_edge  or  n_rear > grid_front_edge:
                continue

            #TODO: represent nv changing lanes & covering two of them

            # At this point we have a vehicle that is somewhere on the grid. Now to figure out which zone(s) it occupies.
            z_num_rear = None #num zones in front of the base zone for rear of the neighbor
            z_num_front = None #num zones in front of the base zone for front of the neighbor
            base_idx = None #index of the first element of the first zone in the given column (center lane is broken into 2 pieces, effectively 2 columns)
            elements_per_zone = None

            # Try the center column - get the zone numbers for both ends of the vehicle, limited to the valid range of zones
            if nv.lane_id == host_lane_id:
                my_half_length = 0.5 * me.model.veh_length
                if n_front >= -my_half_length  and  n_rear <= my_half_length: #neighbor is on top of host (crash will soon be flagged)
                    z_num_front = ObsVec.ZONES_BEHIND
                    z_num_rear = ObsVec.ZONES_BEHIND
                    base_idx = ObsVec.BASE_CTR_REAR
                    elements_per_zone = ObsVec.NORM_ELEMENTS

                elif n_front < 0.0: #vehicle is behind host
                    z_num_rear = max(ObsVec.ZONES_BEHIND + math.floor((n_rear + half_zone + 0.001)/ObsVec.OBS_ZONE_LENGTH), 0)
                    z_num_front = min(ObsVec.ZONES_BEHIND + math.floor((n_front + half_zone - 0.001)/ObsVec.OBS_ZONE_LENGTH), ObsVec.ZONES_BEHIND - 1)
                    base_idx = ObsVec.BASE_CTR_REAR
                    elements_per_zone = ObsVec.NORM_ELEMENTS #no lane boundary info behind the host

                else: #vehicle is in front of host
                    z_num_rear = max(math.floor((n_rear - half_zone + 0.001)/ObsVec.OBS_ZONE_LENGTH), 0)
                    z_num_front = min(math.floor((n_front - half_zone - 0.001)/ObsVec.OBS_ZONE_LENGTH), ObsVec.ZONES_FORWARD - 1)
                    base_idx = ObsVec.BASE_CTR_FRONT
                    elements_per_zone = ObsVec.CTR_ELEMENTS

            # Or one of the side columns
            else:

                # Find the column and set its base index
                lane_diff = nv.lane_id - host_lane_id
                if lane_diff == -2:
                    base_idx = ObsVec.BASE_LL
                elif lane_diff == -1:
                    base_idx = ObsVec.BASE_L
                elif lane_diff == 1:
                    base_idx = ObsVec.BASE_R
                elif lane_diff == 2:
                    base_idx = ObsVec.BASE_RR
                else:
                    raise ValueError("///// BridgitModel.get_obs_vector: lane_diff = {} is not valid for vehicle {} in lane {}."
                                    .format(lane_diff, v_idx, nv.lane_id))
                elements_per_zone = ObsVec.NORM_ELEMENTS

                # Get the zone numbers for both ends of the vehicle, limited to the valid range of zones in a column
                z_num_rear = max(math.floor((n_rear - grid_rear_edge + 0.001)/ObsVec.OBS_ZONE_LENGTH), 0)
                z_num_front = min(math.floor((n_front - grid_rear_edge - 0.001)/ObsVec.OBS_ZONE_LENGTH), (ObsVec.ZONES_BEHIND + ObsVec.ZONES_FORWARD + 1))

            assert z_num_rear <= z_num_front, \
                    "///// BridgitModel.get_obs_vector: host lane = {}, host_p = {:.2f}; vehicle {} in lane {}, p = {:.2f} has has z_num_rear = {}, " \
                    .format(host_lane_id, host_p, v_idx, nv.lane_id, nv.p, z_num_rear) + \
                    "z_num_front = {}. n_rear = {:.2f}, n_front = {:.2f}".format(z_num_front, n_rear, n_front)


            # Find the relative speed, normalized to [-1, 1]
            rel_speed = (nv.cur_speed - me.cur_speed)/Constants.MAX_SPEED

            # Loop through all the zones in the column that are at least partially occupied by this neighbor
            for z in range(z_num_rear, z_num_front+1):

                # Compute the zone index in the obs vector, and set the occupied flag and relative speed for that zone
                z_idx = base_idx + z*elements_per_zone
                obs[z_idx + 2] = 1.0 #occupied
                obs[z_idx + 3] = rel_speed

        return obs
