"""Defines global constants that are used by multiple classes."""

class Constants:

    VEHICLE_LENGTH          = 20.0      #m
    NUM_LANES               = 3         #total number of unique lanes in the scenario
    MAX_SPEED               = 36.0      #vehicle's max achievable speed, m/s
    MAX_ACCEL               = 3.0       #vehicle's max achievable acceleration (fwd or backward), m/s^2
    MAX_JERK                = 4.0       #max desirable jerk for occupant comfort, m/s^3
    ROAD_SPEED_LIMIT        = 29.1      #Roadway's legal speed limit on all lanes, m/s (29.1 m/s = 65 mph)
    SCENARIO_LENGTH         = 3000.0    #total length of the roadway, m
    SCENARIO_BUFFER_LENGTH  = 200.0     #length of buffer added to the end of continuing lanes, m
    NUM_NEIGHBORS           = 6         #total number of neighbor vehicles in scenario (some or all may not be active)
    OBS_ZONE_LENGTH         = 2.0 * ROAD_SPEED_LIMIT #the length of a roadway observation zone, m
    #TODO: make this dependent upon time step size:
    HALF_LANE_CHANGE_STEPS  = 3.0 / 0.5 // 2    #num steps to get half way across the lane (equally straddling both)
    TOTAL_LANE_CHANGE_STEPS = 2 * HALF_LANE_CHANGE_STEPS
    MAX_STEPS_SINCE_LC      = 60        #largest num time steps we will track since previous lane change
    NUM_DIFFICULTY_LEVELS   = 6         #num levels of environment difficulty for the agent to learn; see descrip above
    # The following are for level neighbor adaptive cruise control (ACC) functionality
    DISTANCE_OF_CONCERN     = 8.0 * VEHICLE_LENGTH #following distance below which the vehicle needs to start slowing to avoid forward neighbor
    CRITICAL_DISTANCE       = 2.0 * VEHICLE_LENGTH #following distance below which the vehicle needs to be matching its forward neighbor's speed
