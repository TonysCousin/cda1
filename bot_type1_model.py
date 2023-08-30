import numpy as np
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


    def get_obs_vector(self) -> np.array:
        """Produces the observation vector for this vehicle object.

            CAUTION: the returned observation vector is at actual world scale and needs to be
                     preprocessed before going into a NN!
        """

        #TODO: flesh this out - return array should be dtype=np.float32
        obs = np.zeros(7, dtype = np.float32)
        return obs

        # Reinitialize the remainder of the observation vector
        self._verify_obs_limits("reset after populating main obs with ego stuff")
        self._update_obs_zones()

        # Reinitialize the ego vehicle and the whole observation vector
        ego_rem, lid, la, lb, l_rem, rid, ra, rb, r_rem = self.roadway.get_current_lane_geom(ego_lane_id, ego_p)
