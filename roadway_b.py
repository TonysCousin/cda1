import math
from typing import Tuple, Dict, List
from constants import Constants
from lane import Lane

class Roadway:
    """Defines the geometry of the roadway lanes and their drivable connections.  All dimensions are
        physical quantities, measured in meters from an arbitrary map origin and are in the map
        coordinate frame.  The roadway being modeled is diagramed in the software requirements doc.
        This class provides convertor methods to/from the parametric coordinate
        frame, which abstracts it slightly to be more of a structural "schematic" for better
        representation in our NN observation space. To that end, all lanes in the parametric frame are
        considered parallel and physically next to each other (even though the boundary separating two
        lanes may not be permeable, e.g. a jersey wall).

        Traffic flows from left to right on the page, with travel direction being to the right. The coordinate
        system is oriented so that the origin is at the left end of lane 3, with the X axis going to the right
        and Y axis going upward on the page. Not all lanes have to begin at X = 0, but at least one does.
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

        Case 4, two parallel lanes indefinitely long:  no diagram needed, but A < 0 and B = inf, rem = inf.


        CAUTION: This is not a general container.  This __init__ code defines the exact geometry of the highway.
    """

    NUM_LANES               = 6         #total number of unique lanes in the scenario
    WIDTH                   = 30.0      #lane width, m; using a crazy large number so that grapics are pleasing
    COS_RAMP_ANGLE          = 0.8660    #cosine of the angle of any ramp segment from the X axis


    def __init__(self,
                 debug      : int = 0   #debug printing level
                ):

        self.debug = debug
        if self.debug > 1:
            print("///// Entering Roadway.__init__")
        self.lanes = [] #list of all the lanes in the scenario; list index is lane ID

        NORMAL_SL   = 29.1 #speed limit, m/s (65 mph)
        HIGH_SL     = 33.5 #speed limit, m/s (75 mph)
        RAMP_SL     = 20.1 #speed limit, m/s (45 mph)

        # NOTE: all values used in this geometry are lane centerlines

        # Segment values in the following columns:
        #       x0          y0              x1          y1              len     speed limit

        # Lane 0 - entrance ramp, short merge segment, then exit ramp
        L0_Y = 3*Roadway.WIDTH #the merge segment
        segs = [(1653.6,    L0_Y+0.5*400.0, 2000.0,     L0_Y,           400.0,  RAMP_SL),
                (2000.0,    L0_Y,           2400.0,     L0_Y,           400.0,  NORMAL_SL),
                (2400.0,    L0_Y,           2486.6,     L0_Y+0.5*100.0, 100.0,  RAMP_SL)]
        lane = Lane(0, 1653.6, 900.0, segs, right_id = 1, right_join = 2000.0, right_sep = 2400.0)
        self.lanes.append(lane)

        # Lane 1 - short full-speed ramp, then straight to the end, with high speed final segment
        L1_Y = L0_Y - Roadway.WIDTH
        segs = [(626.8,     L1_Y+0.5*200.0, 800.0,      L1_Y,           200.0,  NORMAL_SL),
                (800.0,     L1_Y,           2400.0,     L1_Y,           1600.0, NORMAL_SL),
                (2400.0,    L1_Y,           3000.0,     L1_Y,           600.0,  HIGH_SL)]
        lane = Lane(1, 626.8, 2400.0, segs, left_id = 0, left_join = 2000.0, left_sep = 2400.0,
                    right_id = 2, right_join = 800.0, right_sep = 3000.0)
        self.lanes.append(lane)

        # Lane 2 - spawned from lane 3, then straight to the end
        L2_Y = L1_Y - Roadway.WIDTH
        segs = [(500.0,     L2_Y,           3000.0,     L2_Y,           2500.0, NORMAL_SL)]
        lane = Lane(2, 500.0, 2500.0, segs, left_id = 1, left_join = 800.0, left_sep = 3000.0,
                    right_id = 3, right_join = 500.0, right_sep = 2200.0)
        self.lanes.append(lane)

        # Lane 3 - high-speed single-lane road, then runs parallel to lane 2 for a while
        L3_Y = 0.0
        segs = [(0.0,       L3_Y,           500.0,      L3_Y,           500.0,  HIGH_SL),
                (500.0,     L3_Y,           2200.0,     L3_Y,           1700.0, NORMAL_SL)]
        lane = Lane(3, 0.0, 2200.0, segs, left_id = 2, left_join = 500.0, left_sep = 2200.0,
                    right_id = 4, right_join = 1300.0, right_sep = 1500.0)
        self.lanes.append(lane)

        # Lane 4 - entrance ramp with short merge area then exit ramp
        L4_Y = L3_Y - Roadway.WIDTH #the merge segment
        segs = [(953.6,     L4_Y-0.5*400.0, 1300.0,     L4_Y,           400.0,  RAMP_SL),
                (1300.0,    L4_Y,           1500.0,     L4_Y,           200.0,  NORMAL_SL),
                (1500.0,    L4_Y,           1586.6,     L4_Y-0.5*100.0, 100.0,  RAMP_SL)]
        lane = Lane(4, 953.6, 700.0, segs, left_id = 3, left_join = 1300.0, left_sep = 1500.0,
                    right_id = 5, right_join = 953.6, right_sep = 1350.0)                       #CAUTION: right_join is in parametric frame!
        self.lanes.append(lane)

        # Lane 5 - secondary entrance ramp
        L5_Y = L4_Y - Roadway.WIDTH #the stubby merge segment
        segs = [(953.6,     L5_Y-0.5*400.0, 1300.0,     L5_Y,           400.0,  RAMP_SL),
                (1300.0,    L5_Y,           1350.0,     L5_Y,           50.0,   RAMP_SL)]
        lane = Lane(5, 953.6, 450.0, segs, left_id = 4, left_join = 953.6, left_sep = 1350.0)
        self.lanes.append(lane)


    def map_to_param_frame(self,
                           x                : float,        #X coordinate in the map frame, m
                           lane             : int           #lane ID (0-indexed)
                          ) -> float:                       #Returns P coordinate, m
        """Converts a point in the map coordinate frame (x, y) to a corresponding point in the parametric coordinate
            frame (p, q). Since the vehicles have no freedom of lateral movement other than whole-lane changes, Y
            coordinates are not important, only lane IDs. These will not change between the frames.

            CAUTION: this logic is specific to the roadway geometry defined in __init__().
        """

        p = x
        if lane == 0:
            join_point = self.lanes[0].segments[0][2]
            sep_point = self.lanes[0].segments[2][0]
            if x < join_point:
                p = join_point - (join_point - x)/self.COS_RAMP_ANGLE
            elif x > sep_point:
                p = sep_point + (x - sep_point)/self.COS_RAMP_ANGLE

        elif lane == 1:
            join_point = self.lanes[1].segments[0][2]
            if x < join_point:
                p = join_point - (join_point - x)/self.COS_RAMP_ANGLE

        elif lane == 4:
            join_point = self.lanes[4].segments[0][2]
            sep_point = self.lanes[4].segments[2][0]
            if x < join_point:
                p = join_point - (join_point - x)/self.COS_RAMP_ANGLE
            elif x > sep_point:
                p = sep_point + (x - sep_point)/self.COS_RAMP_ANGLE

        elif lane == 5:
            join_point = self.lanes[5].segments[0][2]
            if x < join_point:
                p = join_point - (join_point - x)/self.COS_RAMP_ANGLE

        return p


    def param_to_map_frame(self,
                           p                : float,        #P coordinate in the parametric frame, m
                           lane             : int           #lane ID (0-indexed)
                          ) -> float:                       #Returns X coordinate, m
        """Converts a point in the parametric coordinate frame (p, q) to a corresponding point in the map frame (x, y).
            Since the vehicles have no freedom of lateral movement other than whole-lane changes, Q and Y coordinates
            are not important, only lane IDs, which will not change between coordinate frames.

            CAUTION: this logic is specific to the roadway geometry defined in __init__().
        """

        # We need to use max() here to compensate for small round-off errors
        x = p
        if lane == 0:
            join_point = self.lanes[0].segments[0][2]
            sep_point = self.lanes[0].segments[2][0]
            if p < join_point:
                x = max(join_point - (join_point - p)*self.COS_RAMP_ANGLE, self.lanes[0].start_x)
            elif p > sep_point:
                x = sep_point + (p - sep_point)*self.COS_RAMP_ANGLE

        elif lane == 1:
            join_point = self.lanes[1].segments[0][2]
            if p < join_point:
                x = max(join_point - (join_point - p)*self.COS_RAMP_ANGLE, self.lanes[1].start_x)

        elif lane == 4:
            join_point = self.lanes[4].segments[0][2]
            sep_point = self.lanes[4].segments[2][0]
            if p < join_point:
                x = max(join_point - (join_point - p)*self.COS_RAMP_ANGLE, self.lanes[4].start_x)
            elif p > sep_point:
                x = sep_point + (p - sep_point)*self.COS_RAMP_ANGLE

        elif lane == 5:
            join_point = self.lanes[5].segments[0][2]
            if p < join_point:
                x = max(join_point - (join_point - p)*self.COS_RAMP_ANGLE, self.lanes[5].start_x)

        return x


    def get_current_lane_geom(self,
                                lane_id         : int   = 0,    #ID of the lane in question
                                p_loc           : float = 0.0   #P coordinate of the point in question, in the parametric frame, m
                             ) -> Tuple[float, int, float, float, float, int, float, float, float]:
        """Determines all of the variable roadway geometry relative to the given point.
            Returns a tuple of (remaining dist in this lane, m,
                                ID of left neighbor ln (or -1 if none),
                                dist to left adjoin point A, m,
                                dist to left adjoin point B, m,
                                remaining dist in left ajoining lane, m,
                                ID of right neighbor lane (or -1 if none),
                                dist to right adjoin point A, m,
                                dist to right adjoin point B, m,
                                remaining dist in right adjoining lane, m).
            If either adjoining lane doesn't exist, its return values will be 0, 0, inf, inf.  All distances are in m.
        """

        # Ensure that the given location is not prior to beginning of the lane
        assert self.param_to_map_frame(p_loc, lane_id) >= self.lanes[lane_id].start_x, \
                "///// Roadway.get_current_lane_geom: p_loc of {} is prior to beginning of lane {}".format(p_loc, lane_id)

        if self.debug > 1:
            print("///// Entering Roadway.get_current_lane_geom for lane_id = ", lane_id, ", p_loc = ", p_loc)
        rem_this_lane = self.lanes[lane_id].length - (p_loc - self.map_to_param_frame(self.lanes[lane_id].start_x, lane_id))

        #TODO: will we still need l_rem & r_rem if agent doesn't know how long the lane is (and it is non-episodic)?

        la = 0.0
        lb = math.inf
        l_rem = math.inf
        left_id = self.lanes[lane_id].left_id
        if left_id >= 0:
            la = self.lanes[lane_id].left_join - p_loc
            lb = self.lanes[lane_id].left_sep - p_loc
            l_rem = self.lanes[left_id].length - (p_loc - self.map_to_param_frame(self.lanes[left_id].start_x, left_id))

        ra = 0.0
        rb = math.inf
        r_rem = math.inf
        right_id = self.lanes[lane_id].right_id
        if right_id >= 0:
            ra = self.lanes[lane_id].right_join - p_loc
            rb = self.lanes[lane_id].right_sep - p_loc
            r_rem = self.lanes[right_id].length - (p_loc - self.map_to_param_frame(self.lanes[right_id].start_x, right_id))

        if self.debug > 0:
            print("///// get_current_lane_geom complete. Returning rem = ", rem_this_lane)
            print("      lid = {}, la = {:.2f}, lb = {:.2f}, l_rem = {:.2f}".format(left_id, la, lb, l_rem))
            print("      rid = {}, ra = {:.2f}, rb = {:.2f}, r_rem = {:.2f}".format(right_id, ra, rb, r_rem))
        return rem_this_lane, left_id, la, lb, l_rem, right_id, ra, rb, r_rem


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

        x = self.param_to_map_frame(p, lane)
        for s in range(len(self.lane[lane].segments)):
            if s[0] <= x <= s[2]:
                return s[5]

        raise ValueError("///// Roadway.get_speed_limit requested for illegal p = {} on lane {}".format(p, lane))
