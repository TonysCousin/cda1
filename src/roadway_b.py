import math
from typing import Tuple, Dict, List
from constants import Constants
from roadway import Roadway, PavementType
from lane import Lane
from target_destination import TargetDestination


class RoadwayB(Roadway):
    """Defines a specific roadway model for the CDA agent operations.  It is 3 km long and contains 6 lanes with ramps."""


    def __init__(self,
                 debug      : int = 0   #debug printing level
                ):

        super().__init__(debug)
        self.name = "Roadway B"

        NORMAL_SL   = 29.1 #speed limit, m/s (65 mph)
        HIGH1_SL    = 31.3 #speed limit, m/s (70 mph)
        HIGH2_SL    = 33.5 #speed limit, m/s (75 mph)
        RAMP1_SL    = 20.1 #speed limit, m/s (45 mph)
        RAMP2_SL    = 22.4 #speed limit, m/s (50 mph)

        # NOTE: all values used in this geometry are lane centerlines

        # Segment values in the following columns:
        #       x0          y0              x1          y1              len     speed limit

        # Lane 0 - entrance ramp, short merge segment, then exit ramp
        L0_Y = 3*Roadway.LANE_WIDTH #the merge segment
        segs = [(1653.6,    L0_Y+0.5*400.0, 2000.0,     L0_Y,           400.0,  RAMP1_SL,   PavementType.ASPHALT),
                (2000.0,    L0_Y,           2400.0,     L0_Y,           400.0,  NORMAL_SL,  PavementType.ASPHALT),
                (2400.0,    L0_Y,           2573.2,     L0_Y+0.5*200.0, 200.0,  RAMP1_SL,   PavementType.ASPHALT)]
        lane = Lane(0, 1653.6, 1000.0, segs, right_id = 1, right_join = 2000.0, right_sep = 2400.0)
        self.lanes.append(lane)

        # Lane 1 - short full-speed ramp, then straight to the end, with high speed final segment
        L1_Y = L0_Y - Roadway.LANE_WIDTH
        segs = [(626.8,     L1_Y+0.5*200.0, 800.0,      L1_Y,           200.0,  NORMAL_SL,  PavementType.ASPHALT),
                (800.0,     L1_Y,           2400.0,     L1_Y,           1600.0, NORMAL_SL,  PavementType.ASPHALT),
                (2400.0,    L1_Y,           3000.0,     L1_Y,           600.0,  HIGH2_SL,   PavementType.ASPHALT)]
        lane = Lane(1, 626.8, 2400.0, segs, left_id = 0, left_join = 2000.0, left_sep = 2400.0,
                    right_id = 2, right_join = 800.0, right_sep = 3000.0)
        self.lanes.append(lane)

        # Lane 2 - spawned from lane 3, then straight to the end
        L2_Y = L1_Y - Roadway.LANE_WIDTH
        segs = [(500.0,     L2_Y,           3000.0,     L2_Y,           2500.0, NORMAL_SL,  PavementType.ASPHALT)]
        lane = Lane(2, 500.0, 2500.0, segs, left_id = 1, left_join = 800.0, left_sep = 3000.0,
                    right_id = 3, right_join = 500.0, right_sep = 2200.0)
        self.lanes.append(lane)

        # Lane 3 - high-speed single-lane road, then runs parallel to lane 2 for a while
        L3_Y = 0.0
        segs = [(0.0,       L3_Y,           500.0,      L3_Y,           500.0,  HIGH1_SL,   PavementType.ASPHALT),
                (500.0,     L3_Y,           2200.0,     L3_Y,           1700.0, NORMAL_SL,  PavementType.ASPHALT)]
        lane = Lane(3, 0.0, 2200.0, segs, left_id = 2, left_join = 500.0, left_sep = 2200.0,
                    right_id = 4, right_join = 1300.0, right_sep = 1500.0)
        self.lanes.append(lane)

        # Lane 4 - entrance ramp with short merge area then exit ramp
        L4_Y = L3_Y - Roadway.LANE_WIDTH #the merge segment
        segs = [(953.6,     L4_Y-0.5*400.0, 1300.0,     L4_Y,           400.0,  RAMP2_SL,   PavementType.ASPHALT),
                (1300.0,    L4_Y,           1500.0,     L4_Y,           200.0,  NORMAL_SL,  PavementType.ASPHALT),
                (1500.0,    L4_Y,           1673.2,     L4_Y-0.5*200.0, 200.0,  RAMP1_SL,   PavementType.ASPHALT)]
        origin_p = 900.0 #manually converted x0 to p0; can't use map_to_param_frame() yet, cuz it requires all lanes to be defined
        lane = Lane(4, 953.6, 800.0, segs, left_id = 3, left_join = 1300.0, left_sep = 1500.0,
                    right_id = 5, right_join = origin_p, right_sep = 1350.0)
        self.lanes.append(lane)

        # Lane 5 - secondary entrance ramp
        L5_Y = L4_Y - Roadway.LANE_WIDTH #the stubby merge segment
        segs = [(953.6,     L5_Y-0.5*400.0, 1300.0,     L5_Y,           400.0,  RAMP2_SL,   PavementType.ASPHALT),
                (1300.0,    L5_Y,           1350.0,     L5_Y,           50.0,   RAMP2_SL,   PavementType.ASPHALT)]
        lane = Lane(5, 953.6, 450.0, segs, left_id = 4, left_join = origin_p, left_sep = 1350.0)
        self.lanes.append(lane)
        self.NUM_LANES = len(self.lanes)

        # Define the possible target destinations
        self.targets.append(TargetDestination(self, 0, 2500.0))
        self.targets.append(TargetDestination(self, 1, 2900.0))
        self.targets.append(TargetDestination(self, 2, 2900.0))
        self.targets.append(TargetDestination(self, 4, 1600.0))
        self.NUM_TARGETS = len(self.targets)


    def map_to_param_frame(self,
                           x                : float,        #X coordinate in the map frame, m
                           lane             : int           #lane ID (0-indexed)
                          ) -> float:                       #Returns P coordinate, m
        """Converts a point in the map coordinate frame (x, y) to a corresponding point in the parametric coordinate
            frame (p, q). Since the vehicles have no freedom of lateral movement other than whole-lane changes, Y
            coordinates are not important, only lane IDs. These will not change between the frames.

            CAUTION: this logic is specific to the roadway geometry defined in __init__().
        """

        # Get the default conversion for all lanes that are completely straight
        p = super().map_to_param_frame(x, lane)

        # Override results for any lanes that involve angled sections
        if lane == 0:
            join_x = self.lanes[0].segments[0][2]
            sep_x = self.lanes[0].segments[2][0]
            if x < join_x:
                p = join_x - (join_x - x)/self.COS_RAMP_ANGLE
            elif x > sep_x:
                p = sep_x + (x - sep_x)/self.COS_RAMP_ANGLE

        elif lane == 1:
            join_x = self.lanes[1].segments[0][2]
            if x < join_x:
                p = join_x - (join_x - x)/self.COS_RAMP_ANGLE

        elif lane == 4:
            join_x = self.lanes[4].segments[0][2]
            sep_x = self.lanes[4].segments[2][0]
            if x < join_x:
                p = join_x - (join_x - x)/self.COS_RAMP_ANGLE
            elif x > sep_x:
                p = sep_x + (x - sep_x)/self.COS_RAMP_ANGLE

        elif lane == 5:
            join_x = self.lanes[5].segments[0][2]
            if x < join_x:
                p = join_x - (join_x - x)/self.COS_RAMP_ANGLE

        return p


    def param_to_map_frame(self,
                           p                : float,        #P coordinate in the parametric frame, m
                           lane             : int           #lane ID (0-indexed)
                          ) -> Tuple:                       #Returns (X, Y) coordinate, m
        """Converts a point in the parametric coordinate frame (p, q) to a corresponding point in the map frame (x, y).
            Since the vehicles have no freedom of lateral movement other than whole-lane changes, we us lane ID as a
            suitable proxy for the vertical location.

            CAUTION: this logic is specific to the roadway geometry defined in __init__().
        """

        # Get the default conversion for all lanes that are completely straight
        x, y = super().param_to_map_frame(p, lane)

        # Handle lanes with angled segments. We need to use max() here to compensate for small round-off errors.
        if lane == 0:
            join_x = self.lanes[0].segments[0][2]
            join_y = self.lanes[0].segments[0][3]
            sep_x = self.lanes[0].segments[2][0]
            sep_y = self.lanes[0].segments[2][1]
            y = join_y
            if p < join_x:
                x = max(join_x - (join_x - p)*self.COS_RAMP_ANGLE, self.lanes[0].start_x)
                y = join_y + (join_x - p)*self.SIN_RAMP_ANGLE
            elif p > sep_x:
                x = sep_x + (p - sep_x)*self.COS_RAMP_ANGLE
                y = sep_y + (p - sep_x)*self.SIN_RAMP_ANGLE

        elif lane == 1:
            join_x = self.lanes[1].segments[0][2]
            join_y = self.lanes[1].segments[0][3]
            y = join_y
            if p < join_x:
                x = max(join_x - (join_x - p)*self.COS_RAMP_ANGLE, self.lanes[1].start_x)
                y = join_y + (join_x - p)*self.SIN_RAMP_ANGLE

        elif lane == 4:
            join_x = self.lanes[4].segments[0][2]
            join_y = self.lanes[4].segments[0][3]
            sep_x = self.lanes[4].segments[2][0]
            sep_y = self.lanes[4].segments[2][1]
            y = join_y
            if p < join_x:
                x = max(join_x - (join_x - p)*self.COS_RAMP_ANGLE, self.lanes[4].start_x)
                y = join_y - (join_x - p)*self.SIN_RAMP_ANGLE #ramp is angling downward
            elif p > sep_x:
                x = sep_x + (p - sep_x)*self.COS_RAMP_ANGLE
                y = sep_y - (p - sep_x)*self.SIN_RAMP_ANGLE #ramp is angling downward

        elif lane == 5:
            join_x = self.lanes[5].segments[0][2]
            join_y = self.lanes[5].segments[0][3]
            y = join_y
            if p < join_x:
                x = max(join_x - (join_x - p)*self.COS_RAMP_ANGLE, self.lanes[5].start_x)
                y = join_y - (join_x - p)*self.SIN_RAMP_ANGLE

        return x, y
