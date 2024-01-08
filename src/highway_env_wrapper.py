print("***** Beginning imports for HighwayEnvWrapper")
import numpy as np
from typing import Tuple, Dict
print("***** HighwayEnvWrapper: ready to import rllib")
from ray.rllib.env.env_context import EnvContext
print("***** HighwayEnvWrapper: rllib import complete.")
from gymnasium.spaces import Box
print("***** HighwayEnvWrapper: gymnasium import complete.")

from constants import Constants
from obs_vec import ObsVec
from highway_env import HighwayEnv


#TODO - remove inheritance from the main env so that this class can specify its own obs space with proper boundaries
#       without constraining the boundaries in the env class (it can have its own or none at all). This should only
#       require instantiating a HighwayEnv object directly and call its methods instead of super().*, and it will
#       need a new method HighwayEnv.is_training().

class HighwayEnvWrapper(HighwayEnv):
    """Wraps the custom environment in order to properly convert observations and actions into usable structures for
        use by a torch NN.
    """

    def __init__(self,
                 config      : EnvContext
                ):

        super().__init__(config)


    def unscaled_reset(self, *,
                seed    :   int = None,     #seed value for the PRNG
                options :   object = None   #currently not recognized by the Env, but appears for syntactic compliance
             ) -> Tuple[np.array, dict]:    #returns a scaled vector of observations usable by a NN plus an info dict

        """Invokes the environment's reset method, but does not scale the resulting observations. This supports an
            inference engine using the output directly in real world units.
            CAUTION: intended to be used only for initial reset by an inference engine. The output will need to be
                     scaled externally if it is to be fed into an NN.
        """

        obs, info = super().reset(options = options)
        return obs, info


    def reset(self, *,
                seed    :   int = None,     #seed value for the PRNG
                options :   object = None   #currently not recognized by the Env, but appears for syntactic compliance
             ) -> Tuple[np.array, dict]:    #returns a scaled vector of observations usable by a NN plus an info dict

        """Invokes the environment's reset method, then scales the resulting observations to be usable by a NN."""

        obs, info = super().reset(options = options)
        return self.scale_obs(obs), info


    def step(self,
                action  :   list                            #list of actions output from an NN
            ) -> Tuple[np.array, float, bool, bool, Dict]:  #returns scaled obs, rewards, dones truncated flag, and infos,
                                                            # where obs are scaled for NN consumption

        """Passes the actions to the environment to advance it one step and to gather new observations and rewards.

            If the "training" config param is True, then the return obs needs the resulting observations scaled,
            such that it will be usable as input to a NN.  The rewards, dones and info structures are not modfied.
            However, if the "training" config does not exist or is not True, then the returned obs list is NOT scaled.
            This allows an inference engine to directly interpret the observations.  It will then be responsible for
            passing that unscaled obs structure into the scale_obs() method to transform it into values that can be
            sent back to the NN for the next time step, if that action is to be taken.
        """

        # Step the environment
        raw_obs, r, d, t, i = super().step(action)
        o = None
        if self.training:
            o = self.scale_obs(raw_obs)
        else:
            o = raw_obs

        #print("///// wrapper.step: scaled obs vector =")
        #for j in range(len(o)):
        #    print("      {:2d}:  {}".format(j, o[j]))

        return o, r, d, t, i


    def scale_obs(self,
                    obs     : np.array  #raw observation vector from the environment
                  ) -> np.array:        #returns obs vector scaled for use by NN

        """Converts a raw observation vector from the parent environment to a scaled vector usable by a NN."""

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
        scaled[ObsVec.FWD_SPEED]            = obs[ObsVec.FWD_SPEED]             / Constants.MAX_SPEED           #range [0, 1]

        # Copy the remaining contents directly, as no scaling is needed on these
        scaled[ObsVec.FWD_SPEED + 1 : ObsVec.OBS_SIZE] = obs[ObsVec.FWD_SPEED + 1 : ObsVec.OBS_SIZE]

        # Return the obs as an ndarray
        vec = np.array(scaled, dtype = np.float32)
        if self.debug > 1:
            print("scale_obs returning vec size = ", vec.shape)
            print(vec)

        return vec
