from typing import List
from abc import ABC, abstractmethod
import numpy as np
from gymnasium.spaces import Box

from hp_prng import HpPrng
from roadway import Roadway

class VehicleGuidance(ABC):
    """Abstract base class for vehicle guidance algorithms that map observations to action commands for a vehicle."""

    def __init__(self,
                 prng       : HpPrng,       #pseudo-random number generator
                 is_learning: bool = True,  #is the guidance algorithm an RL agent under training?
                 obs_space  : Box = None,   #the observation space used by the environment model
                 act_space  : Box = None,   #the action space used by the environment model
                 name       : str = "Unknown", #a descriptive name for this instance (or class)
                ):

        self.prng = prng
        self.my_vehicle = None
        self.is_learning = is_learning
        self.name = name
        self.roadway = None #this must be defined in reset() before any other methods are called
        self.targets = None #this must be defined in reset() before any other methods are called

        # Environment obs/action spaces that are needed by objects used for inference only
        self.obs_space = obs_space
        self.action_space = act_space


    def set_vehicle(self,
                    vehicle         : object   #the vehicle that owns this guidance object (type Vehicle is not available during construction)
                   ):
        """Stores the host vehicle's info.  This must be called before step(); ideally as soon as the vehicle object is constructed."""

        self.my_vehicle = vehicle


    def reset(self,
              roadway       : Roadway,  #the roadway geometry used for this episode
              init_lane     : int,      #the lane the vehicle is starting in
              init_p        : float,    #vehicle's initial P coordinate, m
             ):

        """Makes vehicle's roadway and initial location info available in case the instantiated guidance wants to use it."""

        self.roadway = roadway
        self.targets = roadway.targets

        # Leave it up to implementing classes whether they want to use location data.


    @abstractmethod
    def step(self,
             obs    : np.array, #vector of local observations available to the instantiating vehicle
            ) -> List:          #returns the observation vector (allows an opportunity to modify its plan elements)

        """Applies the tactical guidance algorithm for one time step to convert vehicle observations into action commands."""

        raise NotImplementedError("///// VehicleGuidance.step needs to be overridden by a concrete instantiation.")


    def plan_route(self,
                   obs      : np.array,     #the observation vector for this vehicle for the current time step
                  ) -> np.array:

        """Produces a strategic guidance (route) plan for the agent. It does not have to be implemented for every vehicle type,
            but the environment expects this method to be overridden and implemented for the ego vehicle (vehicle ID 0).
        """

        raise NotImplementedError("///// VehicleGuidance.plan_route needs to be overridden for the selected vehicle type (probably in vehicle[0]).")
