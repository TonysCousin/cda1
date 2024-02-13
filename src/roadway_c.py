import math
from typing import Tuple, Dict, List
from constants import Constants
from roadway import Roadway, PavementType
from lane import Lane
from target_destination import TargetDestination


class RoadwayC(Roadway):
    """Defines a specific roadway model for the CDA agent operations.  It is 3 km long and contains 6 parallel lanes with
        several speed limit changes along the length of the lanes.
    """


    def __init__(self,
                 debug      : int = 0   #debug printing level
                ):

        super().__init__(debug)
        self.name = "C"

        # NOTE: all values used in this geometry are lane centerlines

        # Segment values in the following columns:
        #       x0          y0              x1          y1              len     speed limit

        # Lane 0
        L0_Y = 3*Roadway.LANE_WIDTH
        segs = [( 900.0,    L0_Y,           1200.0,     L0_Y,           300.0,  31.3,       PavementType.ASPHALT),
                (1200.0,    L0_Y,           1500.0,     L0_Y,           300.0,  29.1,       PavementType.ASPHALT),
                (1500.0,    L0_Y,           1800.0,     L0_Y,           300.0,  24.6,       PavementType.ASPHALT),
                (1800.0,    L0_Y,           2100.0,     L0_Y,           300.0,  20.1,       PavementType.ASPHALT),
                (2100.0,    L0_Y,           2400.0,     L0_Y,           300.0,  24.6,       PavementType.ASPHALT),
                (2400.0,    L0_Y,           2700.0,     L0_Y,           300.0,  29.1,       PavementType.ASPHALT),
                (2700.0,    L0_Y,           3000.0,     L0_Y,           300.0,  33.5,       PavementType.ASPHALT),
            ]
        lane = Lane(0, 900.0, 2100.0, segs, right_id = 1, right_join = 900.0, right_sep = 3000.0)
        self.lanes.append(lane)

        # Lane 1
        L1_Y = L0_Y - Roadway.LANE_WIDTH
        segs = [( 600.0,    L1_Y,            900.0,     L1_Y,           300.0,  26.9,       PavementType.ASPHALT),
                ( 900.0,    L1_Y,           1200.0,     L1_Y,           300.0,  31.3,       PavementType.ASPHALT),
                (1200.0,    L1_Y,           1500.0,     L1_Y,           300.0,  29.1,       PavementType.ASPHALT),
                (1500.0,    L1_Y,           1800.0,     L1_Y,           300.0,  24.6,       PavementType.ASPHALT),
                (1800.0,    L1_Y,           2100.0,     L1_Y,           300.0,  20.1,       PavementType.ASPHALT),
                (2100.0,    L1_Y,           2400.0,     L1_Y,           300.0,  24.6,       PavementType.ASPHALT),
                (2400.0,    L1_Y,           2700.0,     L1_Y,           300.0,  29.1,       PavementType.ASPHALT),
                (2700.0,    L1_Y,           3000.0,     L1_Y,           300.0,  33.5,       PavementType.ASPHALT),
            ]
        lane = Lane(1, 600.0, 2400.0, segs, left_id = 0, left_join = 900.0, left_sep = 3000.0,
                    right_id = 2, right_join = 600.0, right_sep = 2100.0)
        self.lanes.append(lane)

        # Lane 2
        L2_Y = L1_Y - Roadway.LANE_WIDTH
        segs = [(   0.0,    L2_Y,            300.0,     L2_Y,           300.0,  17.9,       PavementType.ASPHALT),
                ( 300.0,    L2_Y,            600.0,     L2_Y,           300.0,  22.4,       PavementType.ASPHALT),
                ( 600.0,    L2_Y,            900.0,     L2_Y,           300.0,  26.9,       PavementType.ASPHALT),
                ( 900.0,    L2_Y,           1200.0,     L2_Y,           300.0,  31.3,       PavementType.ASPHALT),
                (1200.0,    L2_Y,           1500.0,     L2_Y,           300.0,  29.1,       PavementType.ASPHALT),
                (1500.0,    L2_Y,           1800.0,     L2_Y,           300.0,  24.6,       PavementType.ASPHALT),
                (1800.0,    L2_Y,           2100.0,     L2_Y,           300.0,  20.1,       PavementType.ASPHALT),
            ]
        lane = Lane(2, 0.0, 2100.0, segs, left_id = 1, left_join = 600.0, left_sep = 2100.0,
                    right_id = 3, right_join = 0.0, right_sep = 2100.0)
        self.lanes.append(lane)

        # Lane 3
        L3_Y = 0.0
        segs = [(   0.0,    L3_Y,            300.0,     L3_Y,           300.0,  17.9,       PavementType.ASPHALT),
                ( 300.0,    L3_Y,            600.0,     L3_Y,           300.0,  22.4,       PavementType.ASPHALT),
                ( 600.0,    L3_Y,            900.0,     L3_Y,           300.0,  26.9,       PavementType.ASPHALT),
                ( 900.0,    L3_Y,           1200.0,     L3_Y,           300.0,  31.3,       PavementType.ASPHALT),
                (1200.0,    L3_Y,           1500.0,     L3_Y,           300.0,  29.1,       PavementType.ASPHALT),
                (1500.0,    L3_Y,           1800.0,     L3_Y,           300.0,  24.6,       PavementType.ASPHALT),
                (1800.0,    L3_Y,           2100.0,     L3_Y,           300.0,  20.1,       PavementType.ASPHALT),
            ]
        lane = Lane(3, 0.0, 2100.0, segs, left_id = 2, left_join = 0.0, left_sep = 2100.0,
                    right_id = 4, right_join = 600.0, right_sep = 2100.0)
        self.lanes.append(lane)

        # Lane 4
        L4_Y = L3_Y - Roadway.LANE_WIDTH
        segs = [( 600.0,    L4_Y,            900.0,     L4_Y,           300.0,  26.9,       PavementType.ASPHALT),
                ( 900.0,    L4_Y,           1200.0,     L4_Y,           300.0,  31.3,       PavementType.ASPHALT),
                (1200.0,    L4_Y,           1500.0,     L4_Y,           300.0,  29.1,       PavementType.ASPHALT),
                (1500.0,    L4_Y,           1800.0,     L4_Y,           300.0,  24.6,       PavementType.ASPHALT),
                (1800.0,    L4_Y,           2100.0,     L4_Y,           300.0,  20.1,       PavementType.ASPHALT),
                (2100.0,    L4_Y,           2400.0,     L4_Y,           300.0,  24.6,       PavementType.ASPHALT),
                (2400.0,    L4_Y,           2700.0,     L4_Y,           300.0,  29.1,       PavementType.ASPHALT),
                (2700.0,    L4_Y,           3000.0,     L4_Y,           300.0,  33.5,       PavementType.ASPHALT),
            ]
        lane = Lane(4, 600.0, 2400.0, segs, left_id = 3, left_join = 600.0, left_sep = 2100.0,
                    right_id = 5, right_join = 900.0, right_sep = 3000.0)
        self.lanes.append(lane)

        # Lane 5
        L5_Y = L4_Y - Roadway.LANE_WIDTH
        segs = [( 900.0,    L5_Y,           1200.0,     L5_Y,           300.0,  31.3,       PavementType.ASPHALT),
                (1200.0,    L5_Y,           1500.0,     L5_Y,           300.0,  29.1,       PavementType.ASPHALT),
                (1500.0,    L5_Y,           1800.0,     L5_Y,           300.0,  24.6,       PavementType.ASPHALT),
                (1800.0,    L5_Y,           2100.0,     L5_Y,           300.0,  20.1,       PavementType.ASPHALT),
                (2100.0,    L5_Y,           2400.0,     L5_Y,           300.0,  24.6,       PavementType.ASPHALT),
                (2400.0,    L5_Y,           2700.0,     L5_Y,           300.0,  29.1,       PavementType.ASPHALT),
                (2700.0,    L5_Y,           3000.0,     L5_Y,           300.0,  33.5,       PavementType.ASPHALT),
            ]
        lane = Lane(5, 900.0, 2100.0, segs, left_id = 4, left_join = 900.0, left_sep = 3000.0)
        self.lanes.append(lane)
        self.NUM_LANES = len(self.lanes)

        # Define the possible target destinations
        self.targets.append(TargetDestination(self, 0, 2900.0))
        self.targets.append(TargetDestination(self, 1, 2900.0))
        self.targets.append(TargetDestination(self, 4, 2900.0))
        self.targets.append(TargetDestination(self, 5, 2900.0))
        self.NUM_TARGETS = len(self.targets)


    """
        Use the base class methods for map_to_param_frame() and param_to_map_frame() since there are no angled segments.
    """
