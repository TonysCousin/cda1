"""Defines global constants that are used by multiple classes."""

class Constants:

    #TODO: remove accel & jerk limits if possible
    MAX_SPEED               = 36.0      #max achievable speed, m/s, as a global physical limit
    MAX_STEPS_SINCE_LC      = 60 #TODO bogus
    REFERENCE_DIST          = 500.0     #an arbitrary length representative of distances we worry about, m
                                        # (may be used to normalize distances for NN i/o)
    N_DISTRO_DIST_REAR      = 100.0     #distance to the rear of ego within which neighbors should be initially placed, m
    N_DISTRO_DIST_FRONT     = 180.0     #distance to the front of ego within which neighbors should be initially placed, m
