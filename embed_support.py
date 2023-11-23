from typing import Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import pandas as pd

from obs_vec import ObsVec


class ObsDataset(Dataset):
    """Defines a custom dataset for the CDA1 sensor observations."""

    def __init__(self,
                 obs_datafile   : str,  #fully qualified pathname to the CSV file containing the data
                ):

        self.df = pd.read_csv(obs_datafile)


    def __len__(self):
        """Returns the number of items in the dataset."""

        return len(self.df)


    def __getitem__(self,
                    idx         : Union[int, torch.Tensor],  #index of the desired data record (or a batch of indices)
                   ) -> torch.Tensor:   #returns the data record as a 1D tensor
        """Retrieves a single data record."""

        # Handle the production of a batch if multiple indices have been provided
        if torch.is_tensor(idx):
            idx = idx.tolist()

        item = self.df.iloc[idx, :]
        npitem = item.to_numpy(dtype = np.float32)
        return npitem


class Autoencoder(nn.Module):
    """Defines an autoencoder NN that will compress and then decompress the observation grid data, attempting to
        recreate the original input data.
    """

    def __init__(self,
                 encoding_size  : int,  #number of data elements in the encoded data (number of first layer neurons)
                ):
        super(Autoencoder, self).__init__()

        self.encoder = nn.Linear(ObsVec.SENSOR_DATA_SIZE, encoding_size)
        self.decoder = nn.Linear(encoding_size, ObsVec.SENSOR_DATA_SIZE)


    def forward(self, x):
        """Computes a forward pass through the NN."""

        x = F.relu(self.encoder(x))
        x = F.sigmoid(self.decoder(x))

        return x
