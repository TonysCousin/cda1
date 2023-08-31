"""Defines global constants that are used by multiple classes."""

class Constants:

    #TODO: remove accel & jerk limits if possible
    MAX_SPEED               = 36.0      #max achievable speed, m/s, as a global physical limit
    MAX_ACCEL               = 3.0       #vehicle's max achievable acceleration (fwd or backward), m/s^2
    MAX_JERK                = 4.0       #max desirable jerk for occupant comfort, m/s^3
    MAX_STEPS_SINCE_LC      = 60 #TODO bogus
    REFERENCE_DIST          = 500.0     #an arbitrary length representative of distances we worry about, m
                                        # (may be used to normalize distances for NN i/o)
