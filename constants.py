"""Defines global constants that are used by multiple classes."""

class Constants:

    MAX_SPEED               = 36.0      #vehicle's max achievable speed, m/s
    MAX_ACCEL               = 3.0       #vehicle's max achievable acceleration (fwd or backward), m/s^2
    MAX_JERK                = 4.0       #max desirable jerk for occupant comfort, m/s^3
    OBS_SIZE                = 7 #TODO bogus
    MAX_STEPS_SINCE_LC      = 60 #TODO bogus

    # The following are for level neighbor adaptive cruise control (ACC) functionality
    #DISTANCE_OF_CONCERN     = 8.0 * VEHICLE_LENGTH #following distance below which the vehicle needs to start slowing to avoid forward neighbor
    #CRITICAL_DISTANCE       = 2.0 * VEHICLE_LENGTH #following distance below which the vehicle needs to be matching its forward neighbor's speed
