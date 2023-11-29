import torch
import gymnasium
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2

class BridgitNN(TorchModelV2):
    """Custom neural network definition that will learn desired behaviors for the Bridget agent."""

    def __init__(self,
                 obs_space      : gymnasium.spaces.Space,
                 action_space   : gymnasium.spaces.Space,
                 num_outputs    : int,
                 model_config   : dict,
                 name           : str
                ):

        super().__init__(obs_space, action_space, num_outputs, model_config, name)

        # Define constants for this implementation (better to put these into model_config once I figure out how to do that)

        # Define the structure for early processing of the macroscopic data

        # Define the structure for the early processing of pavement sensor data

        # Load the pavement encoder and load its first layer weights

        # Define the structure for the early processing of vehicle sensor data

        # Load the vehicle encoder and load its first layer weights

        # The structure that brings all three data streams together

        print("///// BridgitNN model object instantiated.")


    def forward(self,
                ):

        pass #TODO
