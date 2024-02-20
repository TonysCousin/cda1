import numpy as np

from constants import Constants
from obs_vec import ObsVec
from roadway import Roadway
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


    def reset(self,
              roadway       : Roadway,  #the roadwaay geometry to be used for this episode
             ):
        """Resets the model for a new episode."""

        super().reset(roadway)


    def get_obs_vector(self,
                       my_id    : int,      #ID of this vehicle (its index into the vehicles list)
                       vehicles : list,     #list of all Vehicles in the scenario
                       actions  : list,     #list of action commands for this vehicle
                       obs      : np.array, #array of observations from the previous time step
                      ) -> np.array:        #returns a new observation vector for this vehicle

        """Produces the observation vector for this vehicle object if it is still active. An inactive vehicle produces all 0s.

            CAUTION: the returned observation vector is at actual world scale and needs to be preprocessed before going into a NN!

            NOTE: we use Vehicle objects here, but there is no import statment for that type in this class or in the base class, since it
            creates a circular reference during construction. But Python seems to give us full knowledge of those objects' structures
            anyway.
        """

        # Save values from previous time step, then clear the obs vector to prepare for new values
        prev_speed = obs[ObsVec.SPEED_CUR]
        prev_speed_cmd = obs[ObsVec.SPEED_CMD]
        prev_lc_cmd = obs[ObsVec.LC_CMD]
        steps_since_lc = obs[ObsVec.STEPS_SINCE_LN_CHG]
        obs = np.zeros(ObsVec.OBS_SIZE, dtype = float)

        # If this vehicle is inactive, then stop now
        me = vehicles[my_id]
        if not me.active:
            return obs

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
        obs[ObsVec.LC_UNDERWAY] = 0.0
        if me.lane_change_count > 0:
            obs[ObsVec.LC_UNDERWAY] = 1.0
            if me.lane_change_count >= self.lc_compl_steps - 1: #this is the final time step of a LC mvr, so a new mvr can now be considered
                obs[ObsVec.LC_UNDERWAY] = 0.0
                steps_since_lc = self.lc_compl_steps - 1
        obs[ObsVec.STEPS_SINCE_LN_CHG] = steps_since_lc

        # Identify the closest neighbor downtrack of this vehicle; could be a vehicle in the same lane, or in adjacent lane and is
        # maneuvering to get into this lane
        closest_id = None
        closest_dist = Constants.REFERENCE_DIST #we don't need to worry about anything farther than this
        for i in range(len(vehicles)):
            if i == my_id:
                continue

            v = vehicles[i]
            if not v.active:
                continue

            if v.lane_id == me.lane_id  or  \
              (v.lane_id == me.lane_id + 1  and  v.lane_change_status == "left")  or  \
              (v.lane_id == me.lane_id - 1  and  v.lane_change_status == "right"): #this works even for v.lane_id = -1, since it will never change lanes
                fwd_dist = v.p - me.p
                if fwd_dist > 0.0  and  fwd_dist < closest_dist:
                    closest_dist = fwd_dist
                    closest_id = i
                    #if i == 0  and  fwd_dist < 100.0:
                    #    print("*     bot {} model in lane {}: fwd dist to ego in lane {} is {:.1f}.".format(my_id, me.lane_id, v.lane_id, fwd_dist))
        #print("///// BotType1Model.get_obs_vector: closest neighbor ID = {}, dist = {}".format(closest_id, closest_dist))

        # Build the downtrack portions of the obs vector
        obs[ObsVec.FWD_DIST] = closest_dist
        obs[ObsVec.FWD_SPEED] = Constants.MAX_SPEED - me.cur_speed
        if closest_id is not None:
            obs[ObsVec.FWD_SPEED] = vehicles[closest_id].cur_speed - me.cur_speed

        # Check for neighboring vehicles in the 9 zones immediately to the left or right
        obs[ObsVec.LEFT_OCCUPIED] = 0.0
        obs[ObsVec.RIGHT_OCCUPIED] = 0.0
        for i in range(len(vehicles)):
            if i == my_id:
                continue
            v = vehicles[i]
            if not v.active:
                continue

            if v.lane_id == me.lane_id - 1: #it is to our left
                if abs(v.p - me.p) < 4.5*ObsVec.OBS_ZONE_LENGTH:
                    obs[ObsVec.LEFT_OCCUPIED] = 1.0

            elif v.lane_id == me.lane_id + 1: #it is to our right
                if abs(v.p - me.p) < 4.5*ObsVec.OBS_ZONE_LENGTH:
                    obs[ObsVec.RIGHT_OCCUPIED] = 1.0

        #print("///// BotType1Model.get_obs_vector returning ", obs)
        return obs
