import numpy as np
from typing import Tuple, Dict
from ray.rllib.env.env_context import EnvContext
from gymnasium.spaces import Box

from constants import Constants
from obs_vec import ObsVec
from highway_env import HighwayEnv
from scaler import *


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
        return scale_obs(obs), info


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
            o = scale_obs(raw_obs)
        else:
            o = raw_obs

        #print("///// wrapper.step: scaled obs vector =")
        #for j in range(len(o)):
        #    print("      {:2d}:  {}".format(j, o[j]))

        return o, r, d, t, i
