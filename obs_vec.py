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

        The ego vehicle's "external sensor suite" provides A magical way to detect details about the nearby
        roadway (i.e. it's physically possible but we don't want to worry about the details here). It divides
        the roadway into small zones of interest around the host vehicle, as follows.
            * Each zone is the width of a lane (assume all lanes are the same width), and 5 m long (approximately
              the length of a typical passenger car).
            * There is a zone centered on the host vehicle.
            * A series of 20 contiguous zones stretch out in front of the host vehicle, in host's lane.
            * Two similar series of 20 zones appear contiguously to the left of the host's forward zones, and
              two appear to the right, so that there are 5 x 20 zones spread out in front of the host vehicle.
            * There are two contiguous zones immedately to the left and two immediately to the right of the host
              vehicle (immediately behind the forward 20 in those lateral offsets).
            * There are 4 contiguous zones stretched out behind the host vehicle, in host's lane.
            * There are two similar series of 4 zones to the left and right of these rear zones, so that there are
              a block of 5 x 4 zones in the region behind the host vehicle.
            * The below diagram represents the full set of observation zones (each '-' is a zone) surrounding the
              host vehicle "H".  In total there are 5 x (20 + 1 + 4) = 125 observation zones.

              -------------------------
              -------------------------
              ----H--------------------   >>>>> direction of travel
              -------------------------
              -------------------------

        We will represent the zones in longitudinal columns (parallel to the direction of travel). The columns will
        be stored from left to right (relative to host vehicle). The center column looks exactly like the others,
        with a zone surrounding the host vehicle. The zones move with the vehicle, and are a schematic
        representation of the nearby roadway situation. That is, they don't include any lane shape geometry,
        representing every lane as straight, and physically adjacent to its next lane. This is possible because we
        ASSUME that the vehicles are operating well inside their performance limits, so that road geometry won't
        affect their ability to change lanes, accelerate or decelerate at any desired rate. This assumption allows
        use of the parametric coordinate frame, which allows this zone construction.

        NOTE: this structure ignores possibly physical realities of having "adjacent" lanes separated (e.g. a ramp
        coming it at an angle to the mainline lane), which could degrade observations with real world sensors.

        For each zone we record the following attributes (all are floats):
            * Drivable:     1.0 if that location covers a main lane or on-ramp that leads to a trained agent target,
                                0.0 if it covers an exit ramp, or -1.0 if it is not a paved surface
            * Speed limit:  regulatory limit of the lane at the front of the zone, if drivable, normalized by MAX_SPEED,
                                else 0.0 if not driveable.
            * Occupied:     1.0 if any part of a vehicle is in this zone, 0 if empty (always 1 in the host's zone)
            * Rel speed:    relative to host vehicle's speed, (veh_spd - host_spd), normalized by MAX_SPEED, if the
                                zone is occupied, else 0.0 (always 0 in host's zone)

        NOTE that data will not be stored in this way (all 4 variables describing a single zone together). The data
        are layered so that each data type over all zones is gathered together, then the next data type is stored.

        If a neighbor vehicle is observed to be changing lanes, it will indicate occupancy in two adjacent zones,
        as it will have 2 wheels in each during the maneuver. Often, a vehicle of 5 m in length will occupy two
        longitudinally adjacent zones, since this simulation models continuous physics, not discrete grid motion.
        Also, it is allowed to have vehicles of any length - a tractor-trailer could occupy 4 or 5 zones
        longitudinally.

        Because the center lane houses the host vehicle, it is important to represent its boundaries to allow planning
        for legal lane changes. However, we don't need to represent these on as fine a grid as we do the other sensor
        data. Using only a small number of data elements here avoids problems trying to mash this info into a raster-
        like grid that may require convolutional compression; rather, these elements can be exempt from that process.
        There are 3 data elements representing the left boundary through the length of the forward sensor region
        (100 m ahead of the host plus the length of host's zone). Then 3 elements representing the right boundary in
        the same way. For each boundary, element 0 is the host's zone plus the first two forward zones (0-1); element
        1 is forward zones 2-8; element 2 is zones 9-19 (the farthest forward). Values in these cells are:
            * -1 if a lane change is blocked (e.g. solid line or edge of pavement) in any part of its length
            * +1 if a lane change is legal over the entire length represented (i.e. a dashed line)
    """

    #TODO: constants beginning with D_ are now depricated. Names were changed to make them easy to find in code.

    ZONES_FORWARD       = 20 #num zones in front of the vehicle in a given lane
    ZONES_BEHIND        = 4  #num zones behind the vehicle in a given lane
    D_NORM_ELEMENTS       = 4  #num data elements in a normal zone
    OBS_ZONE_LENGTH     = 5.0#longitudinal length of a single zone, m

    # Common elements for all models
    SPEED_CMD           =  0 #agent's most recent speed command, m/s (action feedback from this step)
    SPEED_CMD_PREV      =  1 #desired speed from previous time step, m/s
    LC_CMD              =  2 #agent's most recent lane change command, quantized (values map to the enum class LaneChange)
    LC_CMD_PREV         =  3 #lane change command from previous time step, quantized
    STEPS_SINCE_LN_CHG  =  4 #num time steps since the previous lane change was initiated
    SPEED_CUR           =  5 #agent's actual forward speed, m/s
    SPEED_PREV          =  6 #agent's actual speed in previous time step, m/s
    LOCAL_SPD_LIMIT     =  7 #posted speed limit at the host's current location, m/s

    # Elements specific to bots running ACC & changing lanes to reach a target destination
    FWD_DIST            =  8 #distance to nearest downtrack vehicle in same lane, m
    FWD_SPEED           =  9 #relative speed of the nearest downtrack vehicle in same lane, m/s faster than ego vehicle
    TGT_LANE_OFFSET     = 10 #target dest is this many lanes left (negative) or right (positive) from current lane #TODO: unused???
    LEFT_OCCUPIED       = 11 #is there a vehicle immediately to the left (within +/- 1 zone longitudinally)? (0 = false, 1 = true)
    RIGHT_OCCUPIED      = 12 #is there a vehicle immediately to the right (within +/- 1 zone longitudinally)? (0 = false, 1 = true)

    #
    #..........Elements specific to the Bridgit model is everything below here
    #

    FUTURE1             = 13 #reserved for future use
    FUTURE2             = 14
    FUTURE3             = 15

    # Bridgit controller lane change command outputs; relative desirability for each lane that the vehicle can choose.
    # Values are floats from 0 (don't go there) to 1 (highly desirable).
    DESIRABILITY_LEFT   = 16 #to the left of host's current lane
    DESIRABILITY_CTR    = 17 #host's current lane
    DESIRABILITY_RIGHT  = 18 #to the right of host's current lane

    # Host lane edge markings/permeability
    NUM_BDRY_REGIONS    = 3 #number of data elements representing one side boundary of the lane forward of host
    BASE_LEFT_CTR_BDRY  = DESIRABILITY_RIGHT + 1 #left-hand boundaries of the forward center lane
    BASE_RIGHT_CTR_BDRY = BASE_LEFT_CTR_BDRY + NUM_BDRY_REGIONS #right-hand boundaries of the forward center lane

    # More elements specific to the Bridgit vehicle:
    # This section represents sensor zones surrounding the host vehicle. Each zone (a physical region in space)
    # has 4 sensor values. However, for efficient processing, each sensor data type is stored in a contiguous
    # rregion that covers all zones. This region is referred to as a layer (like layers of pixels in an image).
    # So there are 4 layers stored end-to-end. Each layer is stored by column, left-to-right, across the sensor
    # grid. Each column is represented from rear to front.

    NUM_COLUMNS         = 5 #num columns in the sensor grid
    NUM_ROWS            = ZONES_BEHIND + 1 + ZONES_FORWARD #num rows in the sensor grid, including host's row
    LAYER_SIZE          = NUM_COLUMNS * NUM_ROWS
    BASE_PVMT_TYPE      = BASE_RIGHT_CTR_BDRY + NUM_BDRY_REGIONS #first element of the first sensor layer
    BASE_SPD_LIMIT      = BASE_PVMT_TYPE + LAYER_SIZE
    BASE_OCCUPANCY      = BASE_SPD_LIMIT + LAYER_SIZE
    BASE_REL_SPEED      = BASE_OCCUPANCY + LAYER_SIZE
    FINAL_ELEMENT       = BASE_REL_SPEED + LAYER_SIZE - 1

    # This one is just a convenient alias to where the "sensor" data block begins
    BASE_SENSOR_DATA    = BASE_PVMT_TYPE
    SENSOR_DATA_SIZE    = FINAL_ELEMENT - BASE_SENSOR_DATA + 1
    assert SENSOR_DATA_SIZE % 2 == 0, "///// ObsVec: invalid SENSOR_DATA_SIZE = {}".format(SENSOR_DATA_SIZE)

    # Size references for the number of elements in certain groupings defined in this file.
    # CAUTION! these need to be maintained in sync with any changes to the index structures elsewhere in this file.
    NUM_COMMON_ELEMENTS =  8
    NUM_BOT_ELEMENTS    =  5
    NUM_BRIDGIT_NON_SENSOR = BASE_SENSOR_DATA - FUTURE1

    # Offsets for the individual data elements in each zone
    D_OFFSET_DRIVABLE     = 0
    D_OFFSET_SPD_LMT      = 1
    D_OFFSET_OCCUPIED     = 2
    D_OFFSET_SPEED        = 3

    OBS_SIZE            = FINAL_ELEMENT + 1 #number of elements in the vector
