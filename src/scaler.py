from typing import List
import math
import torch
import numpy as np
from constants import Constants
from obs_vec import ObsVec

"""Provides functions for
    scaling observation vector
    unscaling action vector

    These live outside of any class so that they can be easily accessed from anywhere.
"""

def scale_obs(obs: np.array     #raw observation vector from the environment at world scale
             ) -> np.array:     #returns the obs vector scaled for NN input

    """Converts a raw observation vector to a scaled vector usable by a NN.
        CAUTION: code in this function is tightly coupled to HighwayEnv._init().
    """

    scaled = [0.0]*ObsVec.OBS_SIZE

    # Scale the initial items that require special scaling
    scaled[ObsVec.SPEED_CMD]            = obs[ObsVec.SPEED_CMD]             / Constants.MAX_SPEED           #range [0, 1]
    scaled[ObsVec.SPEED_CMD_PREV]       = obs[ObsVec.SPEED_CMD_PREV]        / Constants.MAX_SPEED           #range [0, 1]
    scaled[ObsVec.LC_CMD]               = obs[ObsVec.LC_CMD]
    scaled[ObsVec.LC_CMD_PREV]          = obs[ObsVec.LC_CMD_PREV]
    scaled[ObsVec.SPEED_CUR]            = obs[ObsVec.SPEED_CUR]             / Constants.MAX_SPEED           #range [0, 1]
    scaled[ObsVec.SPEED_PREV]           = obs[ObsVec.SPEED_PREV]            / Constants.MAX_SPEED           #range [0, 1]
    scaled[ObsVec.LOCAL_SPD_LIMIT]      = obs[ObsVec.LOCAL_SPD_LIMIT]       / Constants.MAX_SPEED           #range [0, 1]
    scaled[ObsVec.STEPS_SINCE_LN_CHG]   = obs[ObsVec.STEPS_SINCE_LN_CHG]    / Constants.MAX_STEPS_SINCE_LC  #range [0, 1]
    scaled[ObsVec.FWD_DIST]             = min(obs[ObsVec.FWD_DIST]          / Constants.REFERENCE_DIST, 1.0)#range [0, 1]
    scaled[ObsVec.FWD_DIST_PREV]        = min(obs[ObsVec.FWD_DIST_PREV]     / Constants.REFERENCE_DIST, 1.0)#range [0, 1]
    scaled[ObsVec.FWD_SPEED]            = obs[ObsVec.FWD_SPEED]             / Constants.MAX_SPEED           #range [0, 1]

    # Copy the remaining contents directly, as no scaling is needed on these
    scaled[ObsVec.FWD_SPEED + 1 : ObsVec.OBS_SIZE] = obs[ObsVec.FWD_SPEED + 1 : ObsVec.OBS_SIZE]

    # Return the obs as an ndarray
    return np.array(scaled, dtype = np.float32)


def unscale_actions(scaled_actions: torch.tensor
                   ) -> List:

    """Converts actions that are scaled in [-1, 1] to their unscaled world units."""

    actions = [None]*2
    actions[0] = (scaled_actions[0].item() + 1.0)/2.0 * Constants.MAX_SPEED
    raw_lc_cmd = min(max(scaled_actions[1].item(), -1.0), 1.0) #command threshold is +/- 0.5
    actions[1] = int(math.floor(raw_lc_cmd + 0.5))

    return actions
