from typing import List
import copy
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

    # Begin by copying the full vector as-is, to ensure that all elements that don't need scaling are covered
    scaled = copy.copy(obs)

    # Redo the initial items that require special scaling
    scaled[ObsVec.SPEED_CMD]            = obs[ObsVec.SPEED_CMD]             / Constants.MAX_SPEED           #scaled range [0, 1]
    scaled[ObsVec.SPEED_CMD_PREV]       = obs[ObsVec.SPEED_CMD_PREV]        / Constants.MAX_SPEED           #scaled range [0, 1]
    scaled[ObsVec.STEPS_SINCE_LN_CHG]   = obs[ObsVec.STEPS_SINCE_LN_CHG]    / Constants.MAX_STEPS_SINCE_LC  #scaled range [0, 1]
    scaled[ObsVec.SPEED_CUR]            = obs[ObsVec.SPEED_CUR]             / Constants.MAX_SPEED           #scaled range [0, 1]
    scaled[ObsVec.SPEED_PREV]           = obs[ObsVec.SPEED_PREV]            / Constants.MAX_SPEED           #scaled range [0, 1]
    scaled[ObsVec.LOCAL_SPD_LIMIT]      = obs[ObsVec.LOCAL_SPD_LIMIT]       / Constants.MAX_SPEED           #scaled range [0, 1]
    scaled[ObsVec.FWD_DIST]             = min(obs[ObsVec.FWD_DIST]          / Constants.REFERENCE_DIST, 1.0)#scaled range [0, 1]
    scaled[ObsVec.FWD_DIST_PREV]        = min(obs[ObsVec.FWD_DIST_PREV]     / Constants.REFERENCE_DIST, 1.0)#scaled range [0, 1]
    scaled[ObsVec.FWD_SPEED]            = obs[ObsVec.FWD_SPEED]             / Constants.MAX_SPEED           #scaled range [0, 1]

    # Return the obs as an ndarray
    return scaled


def unscale_actions(scaled_actions: torch.tensor
                   ) -> List:

    """Converts actions that are scaled in [-1, 1] to their unscaled world units."""

    actions = [None]*2
    actions[0] = (scaled_actions[0].item() + 1.0)/2.0 * Constants.MAX_SPEED
    raw_lc_cmd = min(max(scaled_actions[1].item(), -1.0), 1.0) #command threshold is +/- 0.5
    actions[1] = int(math.floor(raw_lc_cmd + 0.5))

    return actions
