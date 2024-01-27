from abc import ABC, abstractmethod
import math
from typing import Tuple, Dict, List


class PavementType:
    """Enumeration to represent types of pavement."""
    GRASS       = 0 #anything not driveable
    ASPHALT     = 1 #everyday drivability
    GRAVEL      = 2 #gravel road with frequent ruts & potholes, limited traction


class Roadway(ABC):
    """Defines the geometry of a roadway's lanes and their drivable connections.  All dimensions are
        physical quantities, measured in meters from an arbitrary map origin and are in the map
        coordinate frame.  This is an abstract interface to specific instances of roadways.
        This class provides convertor methods to/from the parametric coordinate
        frame, which abstracts it slightly to be more of a structural "schematic" for better
        representation in our NN observation space. To that end, all lanes in the parametric frame are
        considered parallel and physically next to each other (even though the boundary separating two
        lanes may not be permeable, e.g. a jersey wall).

        Traffic flows from left to right on the page, with travel direction being to the right. The coordinate
        system is oriented so that the origin is at the left end of one of the farthest-left reaching lanes,
        so that there is never a negative X coordinate in the map (with the X axis going to the right
        and Y axis going upward on the page). Not all lanes have to begin at X = 0, but at least one does.
        Others begin at X > 0. Y locations are only used for the graphics output; they are not needed for the
        environment calculations.

        Lane connectivities are defined by three parameters that define the adjacent lane, as shown in the following
        series of diagrams.  These parameters work the same whether they describe a lane on the right or left of the
        ego vehicle's lane, so we only show the case of an adjacent lane on the right side.  In this diagram, '[x]'
        is the agent (ego) vehicle location.

        Case 1, adjacent to on-ramp:
                           |<...............rem....................>|
                           |<................B.....................>|
                           |<....A.....>|                           |
                           |            |                           |
            Ego lane  ----[x]---------------------------------------------------------------------------->
                                        /---------------------------/
            Adj lane  -----------------/

        ==============================================================================================================

        Case 2, adjacent to exit ramp:
                           |<..................................rem.......................................>
                           |<.........B..........>|
                           | A < 0                |
                           |                      |
            Ego lane  ----[x]---------------------------------------------------------------------------->
            Adj lane  ----------------------------\
                                                  \------------------------------------------------------>

        ==============================================================================================================

        Case 3, adjacent to mainline lane drop:
                           |<...............rem....................>|
                           |<................B.....................>|
                           | A < 0                                  |
                           |                                        |
            Ego lane  ----[x]---------------------------------------------------------------------------->
            Adj lane  ----------------------------------------------/

        ==============================================================================================================

        Case 4, two parallel lanes indefinitely long:  no diagram needed, but A = 0, B = inf, rem = inf.

        ==============================================================================================================

        As a result of the definition constraints, we CANNOT represent the following:

            * A lane that has more than one connecting lane (joining or separating) on the same side, such as

                Lane 1  --------------------------------------------------------------------------------->
                Lane 2  --------------/                   Lane 3 ----------------/

            * A lane that joins, separates, then joins again, such as

                Lane 1  --------------------------------------------------------------------------------->
                Lane 2  ------------------\                         /------------------------------------>
                                           \-----------------------/

            * Two logical lanes connected end-to-end (e.g. to try getting around the previous constraints), such as

                Lane 1  ---------------------------------->Lane 2---------------------------------------->
    """

    LANE_WIDTH              = 30.0      #lane width, m; this is crazy big to support aesthetic graphic display
    COS_RAMP_ANGLE          = 0.8660    #cosine of the angle of any ramp segment from the X axis
    SIN_RAMP_ANGLE          = 0.5       #sine of the angle of ramp segments from the X axis


    def __init__(self,
                 debug      : int = 0   #debug printing level
                ):

        self.name = "UNDEFINED!"
        self.debug = debug
        if self.debug > 1:
            print("///// Entering Roadway.__init__")

        # THESE NEED TO BE OVERRIDDEN!
        NUM_LANES = 0
        NUM_TARGETS = 0

        # Set up empty lists for items contained in the roadway
        self.lanes = []     #All the lanes in the scenario; list index is lane ID
        self.targets = []   #All of the possible target destinations (each one can be activated individually)



    def map_to_param_frame(self,
                           x                : float,        #X coordinate in the map frame, m
                           lane             : int           #lane ID (0-indexed)
                          ) -> float:                       #Returns P coordinate, m
        """Converts a point in the map coordinate frame (x, y) to a corresponding point in the parametric coordinate
            frame (p, q). Since the vehicles have no freedom of lateral movement other than whole-lane changes, Y
            coordinates are not important, only lane IDs. These will not change between the frames.

            This method may be overridden to handle specific geometries that involve angled ramps, etc.
        """

        p = x
        return p


    def param_to_map_frame(self,
                           p                : float,        #P coordinate in the parametric frame, m
                           lane             : int           #lane ID (0-indexed)
                          ) -> Tuple:                       #Returns (X, Y) coordinate, m
        """Converts a point in the parametric coordinate frame (p, q) to a corresponding point in the map frame (x, y).
            Since the vehicles have no freedom of lateral movement other than whole-lane changes, we us lane ID as a
            suitable proxy for the vertical location.

            This method may be overridden to handle specific geometries that involve angled ramps, etc.
        """

        x = max(p, self.lanes[lane].start_x)
        y = self.lanes[lane].segments[0][1] #left end of the lane

        return x, y


    def get_current_lane_geom(self,
                                lane_id         : int   = 0,    #ID of the lane in question
                                p_loc           : float = 0.0   #P coordinate of the point in question, in the parametric frame, m
                             ) -> Tuple[int, float, float, int, float, float]:

        """Determines all of the variable roadway geometry relative to the given point.
            Returns a tuple of (ID of left neighbor lane (or -1 if none),
                                dist to left adjoin point A, m,
                                dist to left adjoin point B, m,
                                ID of right neighbor lane (or -1 if none),
                                dist to right adjoin point A, m,
                                dist to right adjoin point B, m,
            If either adjoining lane doesn't exist, its return values will be 0, 0, inf.  All distances are in m.

            This is a general method that should not be overridden.
        """

        # Ensure that the given location is not prior to beginning of the lane
        assert self.param_to_map_frame(p_loc, lane_id)[0] >= self.lanes[lane_id].start_x, \
                "///// Roadway.get_current_lane_geom: p_loc of {} is prior to beginning of lane {}".format(p_loc, lane_id)

        if self.debug > 0:
            print("///// Entering Roadway.get_current_lane_geom for lane_id = ", lane_id, ", p_loc = ", p_loc)

        la = 0.0
        lb = math.inf
        left_id = self.lanes[lane_id].left_id
        if left_id >= 0:
            la = self.lanes[lane_id].left_join - p_loc
            lb = self.lanes[lane_id].left_sep - p_loc

        ra = 0.0
        rb = math.inf
        right_id = self.lanes[lane_id].right_id
        if right_id >= 0:
            ra = self.lanes[lane_id].right_join - p_loc
            rb = self.lanes[lane_id].right_sep - p_loc

        if self.debug > 1:
            print("///// get_current_lane_geom complete. Returning")
            print("      lid = {}, la = {:.2f}, lb = {:.2f}".format(left_id, la, lb))
            print("      rid = {}, ra = {:.2f}, rb = {:.2f}".format(right_id, ra, rb))
        return left_id, la, lb, right_id, ra, rb


    def get_target_lane(self,
                        lane        : int,  #ID of the given lane
                        direction   : str,  #either "left" or "right"
                        p           : float #P coordinate for the location of interest, m
                       ) -> int:
        """Returns the lane ID of the adjacent lane on the indicated side of the given lane, or -1 if there is none
            currently adjoining.
        """

        if self.debug > 1:
            print("///// Entering Roadway.get_target_lane. lane = ", lane, ", direction = ", direction, ", p = ", p)
        assert 0 <= lane < len(self.lanes), "get_adjoining_lane_id input lane ID {} invalid.".format(lane)
        if direction != "left"  and  direction != "right":
            return -1

        # Find the adjacent lane ID, then if one exists ensure that current location is between the join & separation points.
        this_lane = self.lanes[lane]
        tgt_id = -1
        if direction == "left":
            tgt_id = this_lane.left_id
            if tgt_id >= 0:
                if p < this_lane.left_join  or  p > this_lane.left_sep:
                    tgt_id = -1

        else: #right
            tgt_id = this_lane.right_id
            if tgt_id >= 0:
                if p < this_lane.right_join  or  p > this_lane.right_sep:
                    tgt_id = -1

        if self.debug > 1:
            print("///// get_target_lane complete. Returning ", tgt_id)
        return tgt_id


    def get_total_lane_length(self,
                                lane    : int   #ID of the lane in question
                             ) -> float:
        """Returns the total length of the requested lane, m"""

        assert 0 <= lane < len(self.lanes), "Roadway.get_total_lane_length input lane ID {} invalid.".format(lane)
        return self.lanes[lane].length


    def get_lane_start_p(self,
                         lane   : int   #ID of the lane in question
                        ) -> float:
        """Returns the P coordinate of the beginning of the lane (in parametric frame)."""

        assert 0 <= lane < len(self.lanes), "Roadway.get_lane_start_p input lane ID {} invalid.".format(lane)
        return self.map_to_param_frame(self.lanes[lane].start_x, lane)


    def get_speed_limit(self,
                        lane    : int,  #ID of the lane in question
                        p       : float,#P coordinate in question
                       ) -> float:

        """Returns the posted speed limit applicable to the given location."""

        assert 0 <= lane < len(self.lanes), "Roadway.get_speed_limit input lane ID {} invalid.".format(lane)

        x, _ = self.param_to_map_frame(p, lane)
        for s in self.lanes[lane].segments:
            if s[0] <= x <= s[2]:
                return s[5]

        raise ValueError("///// Roadway.get_speed_limit requested for illegal p = {} on lane {}".format(p, lane))


    def get_pavement_type(self,
                          lane    : int,    #ID of the lane in question
                          p       : float,  #P coordinate in question
                        ) -> PavementType:  #returns the type of pavement at the given location

        """Returns the pavement type at the indicated lane & P coordinate."""

        x, _ = self.param_to_map_frame(p, lane)
        for s in self.lanes[lane].segments:
            if s[0] <= x <= s[2]:
                return s[6]

        raise ValueError("///// Roadway.get_speed_pavement_type requested for illegal p = {} on lane {}".format(p, lane))


    def get_active_target_list(self):
        """Returns a list of the indexes of the active targets."""

        at = []
        for i, t in enumerate(self.targets):
            if t.active:
                at.append(i)

        return at


    def is_any_target_reachable_from(self,
                                      lane_id   : int,      #lane ID in question
                                      p         : float,     #P coordinate in question, m
                                     ) -> bool:             #returns True if at least 1 active target is reachable from here
        """Determines if at least one active target is reachable from the given location."""

        for t in self.targets:
            if t.active  and  t.is_reachable_from(lane_id, p):
                return True

        return False
