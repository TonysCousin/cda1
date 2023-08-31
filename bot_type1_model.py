import numpy as np

from constants import Constants
from obs_vec import ObsVec
from vehicle_model import VehicleModel

class BotType1Model(VehicleModel):

    """Realizes a concrete model for the Type 1 bot vehicle."""

    def __init__(self,
                 max_jerk   : float = 3.0,  #forward & backward, m/s^3
                 max_accel  : float = 2.0,  #forward & backward, m/s^2
                 length     : float = 5.0,  #length of the vehicle, m
                 lc_duration: float = 3.0,  #time to complete a lane change, sec; must result in an even number when divided by time step
                 time_step  : float = 0.1,  #duration of a single time step, sec
                ):

        super().__init__(max_jerk, max_accel, length, lc_duration, time_step)


    def get_obs_vector(self,
                       my_id    : int,      #ID of this vehicle (its index into the vehicles list)
                       vehicles : list,     #list of all Vehicles in the scenario
                       actions  : list,     #list of action commands for this vehicle
                      ) -> np.array:

        """Produces the observation vector for this vehicle object.

            CAUTION: the returned observation vector is at actual world scale and needs to be
                     preprocessed before going into a NN!

            NOTE: we use Vehicle objects here, but there is no import statment for that type in this class or in the base class, since it
            creates a circular reference during construction. But Python seems to give us full knowledge of those objects' structures
            anyway.
        """

        # Identify the closest neighbor downtrack of this vehicle in the same lane
        me = vehicles[my_id]
        closest_id = None
        closest_dist = Constants.REFERENCE_DIST #we don't need to worry about anything farther than this
        for i in range(len(vehicles)):
            if i == my_id:
                continue

            v = vehicles[i]
            if v.lane_id == me.lane_id:
                fwd_dist = v.p - me.p
                if fwd_dist > 0.0  and  fwd_dist < closest_dist:
                    closest_dist = fwd_dist
                    closest_id = i

        # Build the obs vector
        obs = np.zeros(ObsVec.OBS_SIZE, dtype = np.float32)
        speed_limit = me.roadway.get_speed_limit(me.lane_id, me.p)
        obs[ObsVec.EGO_DES_SPEED_PREV] = obs[ObsVec.EGO_DES_SPEED]
        obs[ObsVec.EGO_DES_SPEED] = actions[0]
        obs[ObsVec.LC_CMD_PREV] = obs[ObsVec.LC_CMD_PREV]
        obs[ObsVec.LC_CMD] = actions[1]
        obs[ObsVec.STEPS_SINCE_LN_CHG] = me.lane_change_count
        obs[ObsVec.EGO_SPEED_PREV] = obs[ObsVec.EGO_SPEED]
        obs[ObsVec.EGO_SPEED] = me.cur_speed

        obs[ObsVec.FWD_DIST] = closest_dist
        obs[ObsVec.FWD_SPEED] = Constants.MAX_SPEED
        if closest_id is not None:
            obs[ObsVec.FWD_SPEED] = vehicles[closest_id].cur_speed

        return obs

        # Reinitialize the remainder of the observation vector
        self._verify_obs_limits("reset after populating main obs with ego stuff")

        # Reinitialize the ego vehicle and the whole observation vector
        ego_rem, lid, la, lb, l_rem, rid, ra, rb, r_rem = self.roadway.get_current_lane_geom(ego_lane_id, ego_p)
