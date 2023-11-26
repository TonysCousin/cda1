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

        self.encoder = nn.Linear(ObsVec.SENSOR_DATA_SIZE//2, encoding_size)
        self.decoder = nn.Linear(encoding_size, ObsVec.SENSOR_DATA_SIZE//2)


    def forward(self, x):
        """Computes a forward pass through the NN."""

        x = F.relu(self.encoder(x))
        x = F.tanh(self.decoder(x))

        return x


def reshape_batch(batch         : torch.Tensor, #a batch of 1D vectors (as defined in ObsVec)
                  batch_size    : int,          #num data records in the batch
                  layer_min     : int,          #lower layer number that will be transposed
                  layer_max     : int,          #upper lwyer num that will be transposed
                 ) -> torch.Tensor:
    """Copies a data batch into a new tensor with a different shape that only represents 2 of the 4 layers."""

    num_cols = 5
    col_len = ObsVec.BASE_L - ObsVec.BASE_LL
    num_rows = ObsVec.ZONES_BEHIND + 1 + ObsVec.ZONES_FORWARD

    #import time
    #begin = time.perf_counter()

    reshaped_batch = torch.Tensor(batch_size, 2, num_cols, num_rows) #always 2 layers
    for layer in range(layer_min, layer_max+1):
        for c in range(num_cols):
            for r in range(num_rows):
                try:
                    index = c*col_len + r*ObsVec.NORM_ELEMENTS + layer
                    reshaped_batch[:, layer - layer_min, c, r] = batch[:, index]
                except IndexError as e:
                    print("IndexError trapped in reshape_batch: layer = {}, c = {}, r = {}, index = {}, batch shape = {}"
                            .format(layer, c, r, index, batch.shape))
                    raise e

    #delta = time.perf_counter() - begin
    #print("reshaping took {:.2e} sec".format(delta))

    #TODO: testing only - the data rows only contain sensor data, not the full observation set.
    """
    try:
    #                 b  c  r   layer
        compare_cells(0, 0, 22, layer_min, batch, reshaped_batch)
        compare_cells(0, 1, 5,  layer_min, batch, reshaped_batch)
        compare_cells(0, 2, 1,  layer_min, batch, reshaped_batch)
        compare_cells(0, 3, 12, layer_min, batch, reshaped_batch)

        compare_cells(0, 4, 15, layer_min, batch, reshaped_batch)
        compare_cells(0, 0, 20, layer_max, batch, reshaped_batch)
        compare_cells(0, 1, 6,  layer_max, batch, reshaped_batch)
        compare_cells(0, 2, 17, layer_max, batch, reshaped_batch)
        compare_cells(0, 3, 10, layer_max, batch, reshaped_batch)
        compare_cells(0, 4, 3,  layer_max, batch, reshaped_batch)

        compare_cells(0, 0, 0,  layer_min, batch, reshaped_batch)
        compare_cells(0, 1, 6,  layer_min, batch, reshaped_batch)
        compare_cells(0, 2, 11, layer_min, batch, reshaped_batch)
        compare_cells(0, 3, 23, layer_min, batch, reshaped_batch)
        compare_cells(0, 4, 14, layer_min, batch, reshaped_batch)

        compare_cells(0, 0, 0,  layer_max, batch, reshaped_batch)
        compare_cells(0, 1, 6,  layer_max, batch, reshaped_batch)
        compare_cells(0, 2, 11, layer_max, batch, reshaped_batch)
        compare_cells(0, 3, 23, layer_max, batch, reshaped_batch)
        compare_cells(0, 4, 14, layer_max, batch, reshaped_batch)

    except AssertionError as e:
        print("\noriginal batch:")
        print(batch)
        print("\nReshaped:")
        print(reshaped_batch)
        raise e
    """

    return reshaped_batch.view(batch_size, -1)


def compare_cells(b, c, r, layer, batch, rb):
    """For testing only."""
    col_len = ObsVec.BASE_L - ObsVec.BASE_LL
    orig_cell = batch[b, c*col_len + r*ObsVec.NORM_ELEMENTS + layer]
    new_cell = rb[b, layer, c, r]
    assert orig_cell == new_cell, "Failed compare: b {}, c {}, r {}, layer {}: orig = {:.2f}, new = {:.2f}".format(b, c, r, layer, orig_cell, new_cell)
