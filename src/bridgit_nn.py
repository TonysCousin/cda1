from typing import Tuple, List
import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium
from ray.rllib.utils.typing import ModelConfigDict
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2

from obs_vec import ObsVec
from embed_support import Autoencoder


class BridgitNN(TorchModelV2, nn.Module):
    """Custom neural network definition that will learn desired behaviors for the Bridget tactical guidance
        agent. It uses two pre-trained networks for encoding pavement and neighbor vehicles sensory data.
        Therefore, these two will not be subjected to learning here.

        Built according to example at https://griddly.readthedocs.io/en/latest/rllib/intro/index.html
    """

    def __init__(self,
                 obs_space      : gymnasium.spaces.Space,
                 action_space   : gymnasium.spaces.Space,
                 num_outputs    : int,
                 model_config   : ModelConfigDict,
                 name           : str,
                ):

        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        self._num_actions = num_outputs
        assert num_outputs == 2*action_space.shape[0], \
                "///// ERROR in BridgitNN construction: num_outputs = {} but action_space is {}".format(num_outputs, action_space.shape)

        # Define constants for this implementation (better to put these into model_config once I figure out how to do that)
        PAVEMENT_SENSOR_MODEL = "/home/starkj/projects/cda1/models/embedding_30p_231218-2246_99.pt"
        self.NUM_PAVEMENT_NEURONS = 30
        self.PAVEMENT_DATA_SIZE = ObsVec.SENSOR_DATA_SIZE // 2

        VEHICLES_SENSOR_MODEL = "/home/starkj/projects/cda1/models/embedding_140v_231217-2225_99.pt"
        self.NUM_VEHICLES_NEURONS = 140
        self.VEHICLES_DATA_SIZE = ObsVec.SENSOR_DATA_SIZE // 2

        NUM_MACRO_NEURONS = 64
        NUM_FC2_NEURONS =   512
        NUM_FC3_NEURONS =   128

        # Define the structure for early processing of the macroscopic data (everything prior to sensor data) - this will be trainable
        self.macro_data_len = ObsVec.BASE_SENSOR_DATA
        self.macro_encoder = nn.Linear(self.macro_data_len, NUM_MACRO_NEURONS)

        # Define the structure for the early processing of pavement sensor data - not to be trained
        self.pavement_encoder = nn.Linear(self.PAVEMENT_DATA_SIZE, self.NUM_PAVEMENT_NEURONS)

        # Load the pavement encoder and get its first layer weights
        temp_pe = Autoencoder(encoding_size = self.NUM_PAVEMENT_NEURONS)
        temp_pe.load_state_dict(torch.load(PAVEMENT_SENSOR_MODEL))
        with torch.no_grad():
            self.pavement_encoder.weight.copy_(temp_pe.state_dict()["encoder.weight"])
            self.pavement_encoder.bias.copy_(temp_pe.state_dict()["encoder.bias"])

        # Define the structure for the early processing of vehicle sensor data - not to be trained
        self.vehicles_encoder = nn.Linear(self.VEHICLES_DATA_SIZE, self.NUM_VEHICLES_NEURONS)

        # Load the vehicle encoder and get its first layer weights
        temp_ve = Autoencoder(encoding_size = self.NUM_VEHICLES_NEURONS)
        temp_ve.load_state_dict(torch.load(VEHICLES_SENSOR_MODEL))
        with torch.no_grad():
            self.vehicles_encoder.weight.copy_(temp_ve.state_dict()["encoder.weight"])
            self.vehicles_encoder.bias.copy_(temp_ve.state_dict()["encoder.bias"])
        #self.fc1 = nn.Linear(525, NUM_MACRO_NEURONS + self.NUM_PAVEMENT_NEURONS + self.NUM_VEHICLES_NEURONS) #TODO debug only

        # The structure that brings all three data streams together
        self.fc2 = nn.Linear(NUM_MACRO_NEURONS + self.NUM_PAVEMENT_NEURONS + self.NUM_VEHICLES_NEURONS, NUM_FC2_NEURONS)
        self.fc3 = nn.Linear(NUM_FC2_NEURONS, NUM_FC3_NEURONS)

        self._actor_head = nn.Sequential(
            nn.Linear(NUM_FC3_NEURONS, self._num_actions)
        )

        self._critic_head = nn.Sequential(
            nn.Linear(NUM_FC3_NEURONS, 1)
        )


    def forward(self,
                input_dict,     #holds "obs" and other elements not used here
                state,          #not used here, probably an empty list
                seq_lens        #not used here, probably None
               ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Computes a forward pass on a batch of data through the NN. The input x represents the full observation vector from
            the environment model.
        """

        x = input_dict["obs"]

        # Pull out the macro observations, according to the ObsVec descriptions and compute its first linear layer
        macro = x[:, 0 : ObsVec.BASE_SENSOR_DATA]
        mo = F.tanh(self.macro_encoder(macro))
        #print("    Macro first layer complete: mo is {}, x is {}".format(mo.shape, x.shape))

        # Pull out the pavement sensor data and compute its embedding
        pavement = x[:, ObsVec.BASE_SENSOR_DATA : ObsVec.BASE_SENSOR_DATA + self.PAVEMENT_DATA_SIZE]
        po = None
        with torch.no_grad():
            po = F.tanh(self.pavement_encoder(pavement))

        # Pull out the vehicles sensor data and compute its embedding
        vehicles = x[:, ObsVec.BASE_OCCUPANCY : ObsVec.BASE_OCCUPANCY + self.VEHICLES_DATA_SIZE]
        vo = None
        with torch.no_grad():
            vo = F.tanh(self.vehicles_encoder(vehicles))

        # Assemble the first full layer input, which is an aggregate of the macro plus the two sensor embeddings outputs
        l1_out = torch.cat((mo, po, vo), dim = 1)
        #x = F.tanh(self.fc1(x)) #TODO debug only

        # Compute the remaining layers of the common network, bringing all of these parts together
        x = F.tanh(self.fc2(l1_out))
        x = F.tanh(self.fc3(x))

        # Final layer for the actor output (action values)
        actions = self._actor_head(x)

        # Final layer for the critic value
        self._value = self._critic_head(x).reshape(-1)

        return actions, state


    def value_function(self):
        """Provides the critic value for the training algorithm."""

        return self._value
