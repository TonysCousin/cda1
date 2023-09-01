from abc import ABC, abstractmethod
import numpy as np
from hp_prng import HpPrng
from roadway_b import Roadway

class VehicleController(ABC):
    """Abstract base class for vehicle control algorithms that map observations to action commands for a vehicle."""

    def __init__(self,
                 prng   : HpPrng,   #pseudo-random number generator
                 roadway: Roadway,  #the roadway geometry
                ):

        self.prng = prng
        self.roadway = roadway
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

        self.my_vehicle = self.vehicle[0] #TODO bogus

        self.vehicles = vehicles
        self._num_vehicles = len(vehicles)
        assert self._num_vehicles > 0, "///// VehicleController.set_vehicle_list: no vehicles defined!"


    @abstractmethod
    def step(self,
             obs    : np.array, #vector of local observations available to the instantiating vehicle
            ) -> list:          #returns a list of action commands, such that
                                # item 0: desired speed, m/s
                                # item 1: lane change command (-1 = change left, 0 = stay in lane, +1 = change right)

        """Applies the control algorithm for one time step to convert vehicle observations into action commands."""

        raise NotImplementedError("///// VehicleController.step needs to be overridden by a concrete instantiation.")
