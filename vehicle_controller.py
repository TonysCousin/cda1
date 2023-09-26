from typing import List
from abc import ABC, abstractmethod
import numpy as np
from hp_prng import HpPrng
from roadway_b import Roadway
from target_destination import TargetDestination

class VehicleController(ABC):
    """Abstract base class for vehicle control algorithms that map observations to action commands for a vehicle."""

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
                    vehicle         : object   #the vehicle that owns this controller (type Vehicle is not available during construction)
                   ):
        """Stores the host vehicle's info.  This must be called before step(); ideally as soon as the vehicle object is constructed."""

        self.my_vehicle = vehicle


    #TODO: do we really need this for the controller?
    def set_vehicle_list(self,
                         vehicles   : list
                        ):
        """Stores the list of vehicles.  This MUST be called as soon as all vehicles are created (before the step() method is called).

            NOTE: we use Vehicle objects here, but there is no import statment for that type in this class or in the base class, since it
            creates a circular reference during construction. But Python seems to give us full knowledge of those objects' structures
            anyway.
        """

        print("***** DEPRECATED METHOD VehicleController.set_vehicle_list() invoked, recording my_vehicle as #0.") #TODO

        self.my_vehicle = self.vehicle[0] #TODO bogus

        self.vehicles = vehicles
        self._num_vehicles = len(vehicles)
        assert self._num_vehicles > 0, "///// VehicleController.set_vehicle_list: no vehicles defined!"


    def reset(self,
              init_lane     : int,      #the lane the vehicle is starting in
              init_p        : float,    #vehicle's initial P coordinate, m
             ):

        """Makes vehicle's initial location info available in case the instantiated controller wants to use it."""

        print("***** VehicleController.reset: vehicle lane_id = {}, p = {:.1f}".format(self.my_vehicle.lane_id, self.my_vehicle.p)) #TODO debug
        #pass


    @abstractmethod
    def step(self,
             obs    : np.array, #vector of local observations available to the instantiating vehicle
            ) -> List:          #returns a list of action commands, such that
                                # item 0: desired speed, m/s
                                # item 1: lane change command (corresponds to type LaneChange)

        """Applies the control algorithm for one time step to convert vehicle observations into action commands."""

        raise NotImplementedError("///// VehicleController.step needs to be overridden by a concrete instantiation.")
