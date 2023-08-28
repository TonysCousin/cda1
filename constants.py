"""Defines global constants that are used by multiple classes."""

class Constants:

    VEHICLE_LENGTH          = 20.0      #m
    MAX_SPEED               = 36.0      #vehicle's max achievable speed, m/s
    MAX_ACCEL               = 3.0       #vehicle's max achievable acceleration (fwd or backward), m/s^2
    MAX_JERK                = 4.0       #max desirable jerk for occupant comfort, m/s^3
    SCENARIO_LENGTH         = 3000.0    #total length of the roadway, m
    #TODO: make this dependent upon time step size:
    HALF_LANE_CHANGE_STEPS  = 3.0 / 0.5 // 2    #num steps to get half way across the lane (equally straddling both)
    TOTAL_LANE_CHANGE_STEPS = 2 * HALF_LANE_CHANGE_STEPS
    MAX_STEPS_SINCE_LC      = 60        #largest num time steps we will track since previous lane change
    NUM_DIFFICULTY_LEVELS   = 6         #num levels of environment difficulty for the agent to learn; see descrip above
    # The following are for level neighbor adaptive cruise control (ACC) functionality
    DISTANCE_OF_CONCERN     = 8.0 * VEHICLE_LENGTH #following distance below which the vehicle needs to start slowing to avoid forward neighbor
    CRITICAL_DISTANCE       = 2.0 * VEHICLE_LENGTH #following distance below which the vehicle needs to be matching its forward neighbor's speed
