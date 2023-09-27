from typing import Dict
from roadway_b import Roadway

class FeederLane:
    """Describes a lane that, directly or indirectly, feeds a TargetDestination. FeederLanes can be chained
        together to describe a complex set of paths to a target. Objects of this class are nodes in a tree
        whose root is the target in question.
    """

    LC_DURATION = 4.0 #a conservative estimate of how long it takes most vehicles to change lanes, sec (each vehicle model can be different)


    def __init__(self,
                 roadway    : Roadway,          #the Roadway geometry in which this target feeder tree lives
                 lane_id    : int,              #ID of the lane represented
                 max_p      : float,            #Farthest point downtrack (in parametric frame) that a vehicle can be and still reach the target, m
                 parent     : object = None,    #the FeederLane that this one feeds (next one closer to the target); None indicates this directly
                                                # feeds the target itself
                ):

        self.roadway = roadway
        self.lane_id = lane_id
        self.max_p = max_p
        self.parent_lane = parent

        # Build the tree of children feeders - start by getting the geometry of the current lane and it's adjacent ones
        self.feeders = []           #list of all lanes that can reach this lane (children); currently limited to no more than 2 (1 on each side)
        _, lid, la, lb, _, rid, ra, rb, _ = roadway.get_current_lane_geom(lane_id, max_p)
        #print("***** FeederLane: lane_id = {}, max_p = {:.1f}, lid = {}, lb = {:.1f}, rid = {}, rb = {:.1f}"
        #      .format(lane_id, max_p, lid, lb, rid, rb))

        # If a feeding lane exists on the left side, add it
        if self._is_feeder(lid):
            self._add_feeder(lid, max_p, la, lb)

        # If a feedling lane exists on the right side, add it
        if self._is_feeder(rid):
            self._add_feeder(rid, max_p, ra, rb)


    def get_starting_points(self) -> Dict:

        """Returns a dict of all locations that can be used as worst-case starting points to reach this lane. Dict entries are
            {lane_id: max_p} defining a lane and the max P coordinate within that lane that a vehicle can have and still
            be able to make it to this lane in time to reach the parent target destination.
        """

        # Start with my own info
        ret = {self.lane_id: self.max_p}

        # Recursively add info from any children
        for f in self.feeders:
            ret.update(f.get_starting_points())

        return ret


    def is_reachable_from(self,
                          lane      : int,      #the starting lane in question
                          p         : float     #the P location on the given lane, m
                         ) -> bool:

        """Returns True if the portion of of lane that this object represents (which can reach its owner) is reachable from
            the specified lane & P location. This is a recursive method.
        """

        # If the request if for this lane then ensure the P location is far enough uptrack
        if lane == self.lane_id:
            if p <= self.max_p:
                return True

        # Else if there are children then inquire of them
        elif len(self.feeders) > 0:
            for f in self.feeders:
                if f.is_reachable_from(lane, p):
                    return True

        # All tests fail, so there is no way to get here from there
        return False


    def _is_feeder(self,
                   id           : int,
                  ) -> bool:
        """Returns True if the indicated lane is a legitimate feeder to the subject lane, and is not the subject's parent.
            This check is essential, since there is an associative property between two adjacent lanes, where each could think of
            the other as its child, unless one is explicitly identified as the parent.
        """

        if id >= 0:
            if self.parent_lane is None:
                return True
            elif id != self.parent_lane.lane_id:
                return True

        return False


    def _add_feeder(self,
                    id          : int,      #ID of the lane to be added
                    ref_p       : float,    #reference P coordinate where any lane change must be completed, m
                    join_a      : float,    #distance downtrack of the reference point where this and the calling lane join (Roadway point 'A'), m
                    join_b      : float,    #distance downtrack of the reference point where this and the calling lane separate (Roadway point 'B'), m
                   ):
        """Creates a FeederLane child and adds it to the list of children of this one, if the geometry allows it."""

        # Determine the speed limit in the both lanes where left joins the subject lane
        p_sl = min(ref_p, ref_p + join_b)
        this_speed_limit = self.roadway.get_speed_limit(self.lane_id, p_sl)
        adj_speed_limit = self.roadway.get_speed_limit(id, p_sl)

        # Find the (conservative) distance required to change lanes to the subject lane, assuming vehicle is going 20% above the higher of
        # these two speed limits.
        lc_dist = 1.2 * max(this_speed_limit, adj_speed_limit) * FeederLane.LC_DURATION

        # This is the latest distance uptrack of the join point that a LC can begin if it is to reach the subject lane in time
        p_lc_begin = p_sl - lc_dist

        # If the required begin distance is uptrack of where the two lanes first join, then we cannot have a feeding relationship (not enough space
        # to perform the lane change maneuver)
        if p_lc_begin < ref_p + join_a:
            #print("***** FeederLane._add_feeder: can't be reached. id = {}, p_lc_begin = {:.1f}, ref_p = {:.1f}, join_a = {:.1f}, join_b = {:.1f}"
            #      .format(id, p_lc_begin, ref_p, join_a, join_b))
            return

        # Create a child node in the tree using recursion
        f = FeederLane(self.roadway, id, p_lc_begin, self)
        self.feeders.append(f)
