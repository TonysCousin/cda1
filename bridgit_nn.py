import torch
import torch.nn as nn
import gymnasium
from ray.rllib.utils.typing import ModelConfigDict
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2

from obs_vec import ObsVec
from embed_support import Autoencoder


class BridgitNN(TorchModelV2, nn.Module):
    """Custom neural network definition that will learn desired behaviors for the Bridget agent."""

    def __init__(self,
                 obs_space      : gymnasium.spaces.Space,
                 action_space   : gymnasium.spaces.Space,
                 num_outputs    : int,
                 model_config   : ModelConfigDict,
                 name           : str,
                ):

        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        # Define constants for this implementation (better to put these into model_config once I figure out how to do that)
        COL_LENGTH = ObsVec.ZONES_BEHIND + 1 + ObsVec.ZONES_FORWARD     #num zones in a column of sensor data, longitudinally
        PAVEMENT_SENSOR_MODEL = "/home/starkj/projects/cda1/training/embedding_30p_231128-1655_99.pt"
        NUM_PAVEMENT_NEURONS = 30
        PAVEMENT_SENSOR_SIZE = 2 * COL_LENGTH * ObsVec.NUM_COLUMNS #number of data elements

        VEHICLES_SENSOR_MODEL = "/home/starkj/projects/cda1/training/embedding_150v_231128-0921_139.pt"
        NUM_VEHICLES_NEURONS = 150
        VEHICLES_SENSOR_SIZE = 2 * COL_LENGTH * ObsVec.NUM_COLUMNS

        NUM_MACRO_NEURONS = 30

        # Define the structure for early processing of the macroscopic data
        self.macro_encoder = nn.Linear(ObsVec.NUM_COMMON_ELEMENTS + ObsVec.NUM_BRIDGIT_NON_SENSOR, NUM_MACRO_NEURONS)

        # Define the structure for the early processing of pavement sensor data
        self.pavement_encoder = nn.Linear(PAVEMENT_SENSOR_SIZE, NUM_PAVEMENT_NEURONS)

        # Load the pavement encoder and load its first layer weights
        temp_pe = Autoencoder(encoding_size = NUM_PAVEMENT_NEURONS)
        temp_pe.load_state_dict(torch.load(PAVEMENT_SENSOR_MODEL))
        print("***** temp_pe = ")
        print(temp_pe)




        # Define the structure for the early processing of vehicle sensor data
        self.vehicles_encoder = nn.Linear(VEHICLES_SENSOR_SIZE, NUM_VEHICLES_NEURONS)

        # Load the vehicle encoder and load its first layer weights

        # The structure that brings all three data streams together

        print("///// BridgitNN model object instantiated.")


    def forward(self,
                ):

        pass #TODO
        # Be sure that output layer uses tanh
