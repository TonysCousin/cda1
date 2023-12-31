from typing import List
from abc import ABC, abstractmethod
import numpy as np
from hp_prng import HpPrng
from roadway_b import Roadway
from target_destination import TargetDestination

class VehicleGuidance(ABC):
    """Abstract base class for vehicle guidance algorithms that map observations to action commands for a vehicle."""

    def __init__(self,
                 prng       : HpPrng,   #pseudo-random number generator
                 roadway    : Roadway,  #the roadway geometry
                 targets    : List,     #a list of the possible targets that the host vehicle may choose as its destination
                ):

        self.prng = prng
        self.roadway = roadway
        self.targets = targets
        self.vehicles = None #TODO: remove this?
        self.my_vehicle = None


    def set_vehicle(self,
                    vehicle         : object   #the vehicle that owns this guidance object (type Vehicle is not available during construction)
                   ):
        """Stores the host vehicle's info.  This must be called before step(); ideally as soon as the vehicle object is constructed."""

        self.my_vehicle = vehicle


    def reset(self,
              init_lane     : int,      #the lane the vehicle is starting in
              init_p        : float,    #vehicle's initial P coordinate, m
             ):

        """Makes vehicle's initial location info available in case the instantiated guidance wants to use it."""

        pass


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
