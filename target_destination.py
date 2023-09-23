from roadway_b import Roadway
from feeder_lane import FeederLane

class TargetDestination:

    """Stores the location of a target destination within the roadway network.
        Vehicles may compare their location to this for evaluating their success.
    """

    def __init__(self,
                 roadway: Roadway,  #the Roadway geometry that this target lives in
                 lane   : int,      #ID of the lane where the target lives
                 p      : float,    #P coordinate of the target, m
                ):

        self.lane_id = lane
        self.p = p
        #print("\n***** TargetDestination created in lane {} at p = {:.1f}".format(lane, p))

        # Define the tree of all lanes that feed into this target point
        self.feeder_lane = FeederLane(roadway, lane, p)


    def is_reachable_from(self,
                          lane  : int,      #lane ID in question
                          p     : float     #P coordinate on the given lane, m
                         ) -> bool:

        """Returns True if this target is reachable from the specified lane ID and P location on that lane."""

        return self.feeder_lane.is_reachable_from(lane, p)
