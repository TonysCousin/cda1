import torch
import torch.nn as nn
import torch.functional as F
import gymnasium
from ray.rllib.utils.typing import ModelConfigDict
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2

from obs_vec import ObsVec
from embed_support import Autoencoder


class BridgitNN(TorchModelV2, nn.Module):
    """Custom neural network definition that will learn desired behaviors for the Bridget tactical guidance
        agent. It uses two pre-trained networks for encoding pavement and neighbor vehicles sensory data.
        Therefore, these two will not be subjected to learning here.
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

        # Define constants for this implementation (better to put these into model_config once I figure out how to do that)
        PAVEMENT_SENSOR_MODEL = "/home/starkj/projects/cda1/training/embedding_30p_231128-1655_99.pt"
        self.NUM_PAVEMENT_NEURONS = 30
        self.PAVEMENT_DATA_SIZE = ObsVec.SENSOR_DATA_SIZE // 2

        VEHICLES_SENSOR_MODEL = "/home/starkj/projects/cda1/training/embedding_150v_231128-0921_139.pt"
        self.NUM_VEHICLES_NEURONS = 150
        self.VEHICLES_DATA_SIZE = ObsVec.SENSOR_DATA_SIZE // 2

        NUM_MACRO_NEURONS = 40
        NUM_FC2_NEURONS =   600
        NUM_FC3_NEURONS =   128

        # Define the structure for early processing of the macroscopic data - this will be trainable
        self.macro_data_len = ObsVec.NUM_COMMON_ELEMENTS + ObsVec.NUM_BRIDGIT_NON_SENSOR
        self.macro_encoder = nn.Linear(self.macro_data_len, NUM_MACRO_NEURONS)

        # Define the structure for the early processing of pavement sensor data - not to be trained
        self.pavement_encoder = nn.Linear(self.PAVEMENT_DATA_SIZE, self.NUM_PAVEMENT_NEURONS)

        # Load the pavement encoder and get its first layer weights
        temp_pe = Autoencoder(encoding_size = self.NUM_PAVEMENT_NEURONS)
        temp_pe.load_state_dict(torch.load(PAVEMENT_SENSOR_MODEL))
        with torch.no_grad:
            self.pavement_encoder.weight.copy_(temp_pe.state_dict["encoder.weight"])
            self.pavement_encoder.bias.copy_(temp_pe.state_dict["encoder.bias"])

        # Define the structure for the early processing of vehicle sensor data - not to be trained
        self.vehicles_encoder = nn.Linear(self.VEHICLES_DATA_SIZE, self.NUM_VEHICLES_NEURONS)

        # Load the vehicle encoder and get its first layer weights
        temp_ve = Autoencoder(encoding_size = self.NUM_VEHICLES_NEURONS)
        temp_ve.load_state_dict(torch.load(VEHICLES_SENSOR_MODEL))
        with torch.no_grad:
            self.vehicles_encoder.weight.copy_(temp_ve.state_dict["encoder.weight"])
            self.vehicles_encoder.bias.copy_(temp_ve.state_dict["encoder.bias"])

        # The structure that brings all three data streams together
        self.fc2 = nn.Linear(NUM_MACRO_NEURONS + self.NUM_PAVEMENT_NEURONS + self.NUM_VEHICLES_NEURONS, NUM_FC2_NEURONS)
        self.fc3 = nn.Linear(NUM_FC2_NEURONS, NUM_FC3_NEURONS)

        print("///// BridgitNN model object instantiated:")
        print(self)


    def forward(self, x):
        """Computes a forward pass on a batch of data through the NN. The input x represents the full observation vector from
            the environment model.
        """

        pass #TODO

        # Pull out the macro observations, according to the ObsVec descriptions and compute its first linear layer
        macro = torch.Tensor(x.shape[0], self.macro_data_len)
        macro[:, 0 : ObsVec.NUM_COMMON_ELEMENTS] = x[:, 0 : ObsVec.NUM_COMMON_ELEMENTS]
        macro[:, ObsVec.NUM_COMMON_ELEMENTS : self.macro_data_len] = x[:, ObsVec.FUTURE1 : ObsVec.FUTURE1 + ObsVec.NUM_BRIDGIT_NON_SENSOR]
        print("*** macro size is {}, macro_data_len = {}".format(macro.shape, self.macro_data_len))
        mo = F.tanh(self.macro_encoder(macro))
        print("    Macro first layer complete: mo is {}".format(mo.shape))

        # Pull out the pavement sensor data and compute its embedding
        pavement = torch.Tensor(x.shape[0], self.PAVEMENT_DATA_SIZE)
        pavement[:, 0: self.PAVEMENT_DATA_SIZE] = x[:, ObsVec.BASE_SENSOR_DATA : ObsVec.BASE_SENSOR_DATA + self.PAVEMENT_DATA_SIZE]
        print("*** pavement is {}, PAVEMENT_DATA_SIZE = {}".format(pavement.shape, self.PAVEMENT_DATA_SIZE))
        po = None
        with torch.no_grad:
            po = F.tanh(self.pavement_encoder(pavement))
        print("    Pavement encoded: po is {}".format(po.shape))

        # Pull out the vehicles sensor data and compute its embedding
        vehicles = torch.Tensor(x.shape[0], self.VEHICLES_DATA_SIZE)
        vehicles[:, 0: self.VEHICLES_DATA_SIZE] = x[:, ObsVec.BASE_OCCUPANCY : ObsVec.BASE_OCCUPANCY + self.VEHICLES_DATA_SIZE]
        print("*** vehicles is {}".format(vehicles.shape))
        vo = None
        with torch.no_grad:
            vo = F.tanh(self.vehicles_encoder(vehicles))
        print("    Vehicles encoded: vo is {}".format(vo.shape))

        # Assemble the first full layer input, which is an aggregate of the macro plus the two sensor embeddings outputs
        l1_out = torch.cat((mo, po, vo), dim = 1)
        print("*** l1_out is {}".format(l1_out.shape))

        # Compute the remaining layers, bringing all of these parts together
        x = F.tanh(self.f2(l1_out))
        x = F.tanh(self.fc3(x))

        return x
