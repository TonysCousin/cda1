class ObsVec:

    """Defines enumeration names for key elements of the observation vector and other important obs info.
        There is exactly one allowable structure for an observation vector, and all agents must abide by it.
        Since only one agent is learning, the other bots need to suck it up and use this as best they can.

        There are two fundamental types of vehicles in use: one ego vehicle (at index 0) that will be in the
        learning program, and a bunch of bots that may be any level of sophistication, but are not learning.
        Since the observations are represented as a 2D ndarray, every vehicle model must use the same obs
        vector definition, even if they don't use all the elements of it. So the vector definition must be the
        union of the needs of all vehicle models.

        We will use the convention that all known common elements will appear first in the vector, followed by
        model-specific elements.  It may be possible that enhancements to bot models or new bot models will
        want to use some of the ego-specific data elements, which is fine, but they won't be re-ordered for it.

        The ego vehicle's "external sensor suite" provides magical way to detect details about the nearby
        roadway (i.e. it's physically possible but we don't want to worry about the details here). It divides
        the roadway into small zones of interest around the host vehicle, as follows.
            * Each zone is the width of a lane (assume all lanes are the same width), and 5 m long (approximately
              the length of a typical passenger car).
            * There is an unused zone centered on the host vehicle (no data will be provided for that zone).
            * A series of 20 contiguous zones stretch out in front of the host vehicle, in host's lane.
            * Two similar series of 20 zones appear contiguously to the left of the host's forward zones, and
              two appear to the right, so that there are 5 x 20 zones spread out in front of the host vehicle.
            * There are two contiguous zones immedately to the left and two immediately to the right of the host
              vehicle (immediately behind the forward 20 in those lateral offsets).
            * There are 4 contiguous zones stretched out behind the host vehicle, in host's lane.
            * There are two similar series of 4 zones to the left and right of these rear zones, so that there are
              a block of 5 x 4 zones in the region behind the host vehicle.
            * The below diagram represents the full set of observation zones (each '-' is a zone) surrounding the
              host vehicle "H".  In total there are 5x20 + 4 + 5x4 = 124 observation zones.

              -------------------------
              -------------------------
              ----H--------------------   >>>>> direction of travel
              -------------------------
              -------------------------

        We will represent the zones in longituudinal columns (parallel to the direction of travel).  The columns on
        each side of the vehicle will all be defined, then the center column (fore & aft of the vehicle) will appear
        last, since it has slightly different attributes. These zones move with the vehicle, and are a schematic
        representation of the nearby roadway situation. That is, they don't include any lane shape geometry,
        representing every lane as straight, and physically adjacent to its next lane. This is possible because we
        ASSUME that the vehicles are operating well inside their performance limits, so that road geometry won't
        affect their ability to change lanes, accelerate or decelerate at any desired rate. This assumption allows
        use of the parametric coordinate frame, which allows this zone construction.

        NOTE: this structure ignores possibley physical realities of having "adjacent" lanes separated (e.g. a ramp
        coming it at an angle to the mainline lane), which could degrade observations.

        Each zone in a side column and in the center column behind the host vehicle will include the following
        attributes (all are floats):
            * Drivable:     1.0 if that location covers a main lane or on-ramp that leads to a trained agent target,
                                0.0 if it covers an exit ramp, or -1.0 if it not a paved surface
            * Speed limit:  regulatory limit of the lane at the front of the zone, if drivable, normalized by MAX_SPEED,
                                else 0.0 if not driveable.
            * Occupied:     1.0 if any part of a vehicle is in this zone, 0 if empty
            * Speed:        relative to host vehicle's speed, (veh_spd - host_spd), normalizedd by MAX_SPEED, if the
                                zone is occupied, else 0.0.

        Each zone in the forward center column (representing the host's current lane) will have the above attributes, but
        also contain:
            * Left boundary:  -1.0 if cannot be crossed (e.g. solid line or grass on the other side), +1.0 if it
                                can be crossed (in either direction)
            * right boundary: -1.0 if cannot be crossed, +1.0 if it can be crossed (in either direction)

        If a neighbor vehicle is observed to be changing lanes, it will indicate occupancy in two adjacent zones,
        as it will have 2 wheels in each during the maneuver. Often, a vehicle of 5 m in length will occupy two
        longitudinally adjacent zones, since this simulation models continuous physics, not discrete grid motion.
        Also, it is allowed to have vehicles of any length - a tractor-trailer could occupy 4 or 5 zones
        longitudianlly.
    """

    ZONES_FORWARD       = 20 #num zones in front of the vehicle in a given lane
    ZONES_BEHIND        = 4  #num zones behind the vehicle in a given lane
    NORM_ELEMENTS       = 4  #num data elements in a normal zone (side column)
    CTR_ELEMENTS        = NORM_ELEMENTS + 2 #num data elements in a forward zone of the center column
    OBS_ZONE_LENGTH     = 5.0#longitudinal length of a single zone, m

    # Common elements
    SPEED_CMD           =  0 #agent's most recent speed command, m/s (action feedback from this step)
    SPEED_CMD_PREV      =  1 #desired speed from previous time step, m/s
    LC_CMD              =  2 #agent's most recent lane change command, quantized (values map to the enum class LaneChange)
    LC_CMD_PREV         =  3 #lane change command from previous time step, quantized
    STEPS_SINCE_LN_CHG  =  4 #num time steps since the previous lane change was initiated
    SPEED_CUR           =  5 #agent's actual forward speed, m/s
    SPEED_PREV          =  6 #agent's actual speed in previous time step, m/s

    # Elements specific to bots running ACC & changing lanes to reach a target destination
    FWD_DIST            =  7 #distance to nearest downtrack vehicle in same lane, m
    FWD_SPEED           =  8 #speed of the nearest downtrack vehicle in same lane, m/s
    TGT_LANE_OFFSET     =  9 #target dest is this many lanes left (negative) or right (positive) from current lane
    LEFT_OCCUPIED       = 10 #is there a vehicle immediately to the left (within +/- 1 zone longitudinally)? (0 = false, 1 = true)
    RIGHT_OCCUPIED      = 11 #is there a vehicle immediately to the right (within +/- 1 zone longitudinally)? (0 = false, 1 = true)

    #
    #..........Elements specific to the ego vehicle (Bridgit) is everything below here
    #
    FUTURE1             = 12 #reserved for future use
    FUTURE2             = 13
    FUTURE3             = 14

    # Bridgit controller lane change command outputs; relative desirability for each lane relative to the vehicle's current lane
    DESIRABILITY_LEFT   = 15
    DESIRABILITY_CTR    = 16
    DESIRABILITY_RIGHT  = 17

    # More elements specific to the Bridgit vehicle:
    # Zone columns are represented from rear to front. Each zone occupies a contiguous set or 3 or 5 vector elements,
    # depending on its purpose. Each column has a base reference, which points to the first element of the rear-most
    # zone in that column.
    BASE_LL             = 18 #first element in the far left column
    BASE_L              = BASE_LL + NORM_ELEMENTS*(ZONES_FORWARD + ZONES_BEHIND + 1)
    BASE_R              = BASE_L  + NORM_ELEMENTS*(ZONES_FORWARD + ZONES_BEHIND + 1)
    BASE_RR             = BASE_R  + NORM_ELEMENTS*(ZONES_FORWARD + ZONES_BEHIND + 1)
    BASE_CTR_REAR       = BASE_RR + NORM_ELEMENTS*(ZONES_FORWARD + ZONES_BEHIND + 1)
    BASE_CTR_FRONT      = BASE_CTR_REAR + NORM_ELEMENTS*ZONES_BEHIND
    FINAL_ELEMENT       = BASE_CTR_FRONT + CTR_ELEMENTS*ZONES_FORWARD - 1

    # Offsets for the individual data elements in each zone
    OFFSET_DRIVABLE     = 0
    OFFSET_SPD_LMT      = 1
    OFFSET_OCCUPIED     = 2
    OFFSET_SPEED        = 3
    OFFSET_LEFT_BDRY    = 4 #in forward center column only
    OFFSET_RIGHT_BDRY   = 5 #in forward center column only

    OBS_SIZE            = FINAL_ELEMENT + 1 #number of elements in the vector
