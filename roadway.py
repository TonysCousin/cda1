
class Roadway:
    """Defines the geometry of the roadway lanes and their drivable connections.  All dimensions are
        physical quantities, measured in meters from an arbitrary map origin and are in the map
        coordinate frame.  The roadway being modeled looks roughly like the diagram at the top of this
        code file.  However, this class provides convertor methods to/from the parametric coordinate
        frame, which abstracts it slightly to be more of a structural "schematic" for better
        representation in our NN observation space. To that end, all lanes in the parametric frame are
        considered parallel and physically next to each other (even though the boundary separating two
        lanes may not be permeable, e.g. a jersey wall).

        All lanes go from left to right, with travel direction being to the right. The coordinate system
        is oriented so that the origin is at the left (beginning of the first lane), with the X axis
        going to the right and Y axis going upward on the page. Not all lanes have to begin at X = 0,
        but at least one does. Others may begin at X > 0. Y locations and the lane segments are only used
        for the graphics output; they are not needed for the environment calculations, per se.

        CAUTION: This is not a general container.  This __init__ code defines the exact geometry of the
        scenario being used by this version of the code.
    """

    WIDTH = 20.0 #lane width, m; using a crazy large number so that grapics are pleasing
    COS_LANE2_ANGLE = 0.8660 #cosine of the angle of lane 2, segment 0, between the map frame and parametric frame

    def __init__(self,
                 debug      : int   #debug printing level
                ):

        self.debug = debug
        if self.debug > 1:
            print("///// Entering Roadway.__init__")
        self.lanes = [] #list of all the lanes in the scenario; list index is lane ID

        # Full length of the modeled lane, extends beyond the length of the scenario so the agent
        # views the road as a continuing situation, rather than an episodic game.
        really_long = SimpleHighwayRamp.SCENARIO_LENGTH + SimpleHighwayRamp.SCENARIO_BUFFER_LENGTH

        # Lane 0 - single segment as the left through lane
        L0_Y = 300.0 #arbitrary y value for the east-bound lane
        segs = [(0.0, L0_Y, really_long, L0_Y, really_long)]
        lane = Lane(0, 0.0, really_long, segs,
                    right_id = 1, right_join = 0.0, right_sep = really_long)
        self.lanes.append(lane)

        # Lane 1 - single segment as the right through lane
        L1_Y = L0_Y - Roadway.WIDTH
        segs = [(0.0, L1_Y, really_long, L1_Y, really_long)]
        lane = Lane(1, 0.0, really_long, segs,
                    left_id = 0, left_join = 0.0, left_sep = really_long,
                    right_id = 2, right_join = 800.0, right_sep = 1320.0)
        self.lanes.append(lane)

        # Lane 2 - two segments as the merge ramp; first seg is separate; second is adjacent to L1.
        # Segments show the lane at an angle to the main roadway, for visual appeal & clarity.
        L2_Y = L1_Y - Roadway.WIDTH
        segs = [(159.1, L2_Y-370.0,  800.0, L2_Y, 740.0),
                (800.0, L2_Y,       1320.0, L2_Y, 520.0)]
        lane = Lane(2, 159.1, 1260.0, segs, left_id = 1, left_join = 800.0, left_sep = 1320.0)
        self.lanes.append(lane)


    def map_to_param_frame(self,
                           x                : float,        #X coordinate in the map frame, m
                           lane             : int           #lane ID (0-indexed)
                          ) -> float:                       #Returns P coordinate, m
        """Converts a point in the map coordinate frame (x, y) to a corresponding point in the parametric coordinate
            frame (p, q). Since the vehicles have no freedom of lateral movement other than whole-lane changes, Y
            coordinates are not important, only lane IDs. These will not change between the frames.
        """

        p = x
        if lane == 2:
            join_point = self.lanes[2].segments[0][2]
            if x < join_point:
                p = join_point - (join_point - x)/self.COS_LANE2_ANGLE

        return p


    def param_to_map_frame(self,
                           p                : float,        #P coordinate in the parametric frame, m
                           lane             : int           #lane ID (0-indexed)
                          ) -> float:                       #Returns X coordinate, m
        """Converts a point in the parametric coordinate frame (p, q) to a corresponding point in the map frame (x, y).
            Since the vehicles have no freedom of lateral movement other than whole-lane changes, Q and Y coordinates
            are not important, only lane IDs, which will not change between coordinate frames.
        """

        x = p
        if lane == 2:
            join_point = self.lanes[2].segments[0][2]
            if p < join_point:
                x = max(join_point - (join_point - p)*self.COS_LANE2_ANGLE, self.lanes[2].start_x)

        return x


    def get_current_lane_geom(self,
                                lane_id         : int   = 0,    #ID of the lane in question
                                p_loc           : float = 0.0   #ego vehicle's P coordinate in the parametric frame, m
                             ) -> Tuple[float, int, float, float, float, int, float, float, float]:
        """Determines all of the variable roadway geometry relative to the given vehicle location.
            Returns a tuple of (remaining dist in this lane, m,
                                ID of left neighbor ln (or -1 if none),
                                dist to left adjoin point A, m,
                                dist to left adjoin point B, m,
                                remaining dist in left ajoining lane, m,
                                ID of right neighbor lane (or -1 if none),
                                dist to right adjoin point A, m,
                                dist to right adjoin point B, m,
                                remaining dist in right adjoining lane, m).
            If either adjoining lane doesn't exist, its return values will be 0, inf, inf, inf.  All distances are in m.
        """

        # Ensure that the given location is not prior to beginning of the lane
        assert self.param_to_map_frame(p_loc, lane_id) >= self.lanes[lane_id].start_x, \
                "///// Roadway.get_current_lane_geom: p_loc of {} is prior to beginning of lane {}".format(p_loc, lane_id)

        if self.debug > 1:
            print("///// Entering Roadway.get_current_lane_geom for lane_id = ", lane_id, ", p_loc = ", p_loc)
        rem_this_lane = self.lanes[lane_id].length - (p_loc - self.map_to_param_frame(self.lanes[lane_id].start_x, lane_id))

        la = 0.0
        lb = SimpleHighwayRamp.SCENARIO_LENGTH + SimpleHighwayRamp.SCENARIO_BUFFER_LENGTH
        l_rem = SimpleHighwayRamp.SCENARIO_LENGTH + SimpleHighwayRamp.SCENARIO_BUFFER_LENGTH
        left_id = self.lanes[lane_id].left_id
        if left_id >= 0:
            la = self.lanes[lane_id].left_join - p_loc
            lb = self.lanes[lane_id].left_sep - p_loc
            l_rem = self.lanes[left_id].length - (p_loc - self.map_to_param_frame(self.lanes[left_id].start_x, left_id))

        ra = 0.0
        rb = SimpleHighwayRamp.SCENARIO_LENGTH + SimpleHighwayRamp.SCENARIO_BUFFER_LENGTH
        r_rem = SimpleHighwayRamp.SCENARIO_LENGTH + SimpleHighwayRamp.SCENARIO_BUFFER_LENGTH
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
        tgt = this_lane
        if direction == "left":
            tgt = this_lane.left_id
            if tgt >= 0:
                if p < this_lane.left_join  or  p > this_lane.left_sep:
                    tgt = -1

        else: #right
            tgt = this_lane.right_id
            if tgt >= 0:
                if p < this_lane.right_join  or  p > this_lane.right_sep:
                    tgt = -1

        if self.debug > 1:
            print("///// get_target_lane complete. Returning ", tgt)
        return tgt


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
