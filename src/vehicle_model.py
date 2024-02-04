from abc import ABC, abstractmethod
import numpy as np

from roadway import Roadway, PavementType

class VehicleModel(ABC):
    """Abstract base class defining the interface for models that describe specific vehicle type capabilities, including
        physical performance limits and the gathering of observations.
    """

    def __init__(self,
                 max_jerk   : float = 3.0,  #forward & backward, m/s^3
                 max_accel  : float = 2.0,  #forward & backward, m/s^2
                 length     : float = 5.0,  #length of the vehicle, m
                 lc_duration: float = 3.0,  #time to complete a lane change, sec; must be a multiple of time_step
                 time_step  : float = 0.1,  #duration of a single time step, sec
                ):

        self.max_jerk = max_jerk
        self.max_accel = max_accel
        self.veh_length = length
        self.roadway = None #this must be defined in reset, before anything else is called

        # Ensure lane change attributes are reasonable
        rsteps = lc_duration/time_step
        nsteps = int(rsteps + 0.5)
        if abs(rsteps - nsteps) > 0.01:
            raise ValueError("///// VehicleModel constructed with illegal lc_duration = {}, time_step = {}, resulting in LC steps = {}"
                             .format(lc_duration, time_step, rsteps))
        self.lc_compl_steps = nsteps    #num time steps required to complete a lane change maneuver


    def reset(self,
              roadway       : Roadway,  #the roadwaay geometry to be used for this episode
             ):
        """Resets the model for a new episode."""

        self.roadway = roadway


    @abstractmethod
    def get_obs_vector(self,
                       my_id    : int,      #ID of this vehicle (its index into the vehicles list)
                       vehicles : list,     #list of all Vehicles in the scenario
                       actions  : list,     #list of action commands for this vehicle
                       obs      : np.array, #array of observations from the previous time step
                      ) -> np.array:

        """Gathers all of the vehicle's observations and returns them as a numpy vector.

            CAUTION: the returned observation vector must be at actual world scale and needs to be
                     preprocessed before going into a NN!
        """

        raise NotImplementedError("///// VehicleModel.get_obs_vector() must be overridden.")
