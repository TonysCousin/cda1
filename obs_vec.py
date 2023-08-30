class ObsVec:
    """Defines enumeration names for key elements of the observation vector and other important obs info.
        There is exactly one allowable structure for an observation vector, and all agents must abide by it.
        Since only one agent is learning, the other bots need to suck it up and use this as best they can.
    """

    # CAUTION: ensure these match the OBS_SIZE defined in Constants
    EGO_DES_SPEED      =  0 #agent's most recent speed command, m/s (action feedback from this step)
    EGO_DES_SPEED_PREV =  1 #desired speed from previous time step, m/s
    LC_CMD             =  2 #agent's most recent lane change command, quantized (values map to the enum class LaneChange)
    LC_CMD_PREV        =  3 #lane change command from previous time step, quantized
    STEPS_SINCE_LN_CHG =  4 #num time steps since the previous lane change was initiated
    EGO_SPEED          =  5 #agent's actual forward speed, m/s
    EGO_SPEED_PREV     =  6 #agent's actual speed in previous time step, m/s
    FWD_DIST           =  7 #distance to nearest downtrack vehicle in same lane, m
    FWD_SPEED          =  8 #speed of the nearest downtrack vehicle in same lane, m/s

    OBS_SIZE            = 9 #number of elements in the vector

    """
    self.Z1_DRIVEABLE       = 11 #is all of this zone drivable pavement? (bool 0 or 1)
    self.Z1_REACHABLE       = 12 #is all of this zone reachable from ego's lane? (bool 0 or 1)
    self.Z1_OCCUPIED        = 13 #is this zone occupied by a neighbor vehicle? (bool 0 or 1)
    self.Z1_ZONE_P          = 14 #occupant's P location relative to this zone (0 = at rear edge of zone, 1 = at front edge)
    self.Z1_REL_SPEED       = 15 #occupant's speed relative to ego speed, fraction of speed limit (+ or -)
    """
