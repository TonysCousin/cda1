from cmath import inf
import sys
import time
from typing import List, Union
import copy
import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from torch.utils.tensorboard import SummaryWriter


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


def main(argv):
    """This program trains an autoencoder to compress the host vehicle's sensor observations, then decompress them to form
        a reasonably accurate reproduction of the original sensor data. Once the training is satisfactory, the weights of
        the encoder layer are saved for future use in our CDA agent.
    """

    # Handle any args
    train_filename = "train.csv"
    test_filename = "test.csv"
    weights_filename = "embedding_weights.pt"
    max_epochs = 100
    lr = 0.001
    batch_size = 4
    enc_size = 50
    num_workers = 0
    program_desc = "Trains the auto-encoder for vector embedding in the cda1 project."
    parser = argparse.ArgumentParser(prog = argv[0], description = program_desc, epilog = "Will run until either max episodes or max timesteps is reached.")
    parser.add_argument("-b", type = int, default = batch_size, help = "Number of data rows in a training batch (default = {})".format(batch_size))
    parser.add_argument("-d", type = int, default = enc_size, help = "Number of neurons in the encoding layer (default = {})".format(enc_size))
    parser.add_argument("-e", type = int, default = max_epochs, help = "Max number of epochs to train (default = {})".format(max_epochs))
    parser.add_argument("-l", type = float, default = lr, help = "Learning rate (default = {})".format(lr))
    parser.add_argument("-n", type = int, default = num_workers, help = "Number of training worker processes (default = {})".format(num_workers))
    parser.add_argument("-r", type = str, default = train_filename, help = "Filename of the training observation dataset (default: {})".format(train_filename))
    parser.add_argument("-t", type = str, default = test_filename, help = "Filename of the test observation dataset (default: {})".format(test_filename))
    parser.add_argument("-w", type = str, default = weights_filename, help = "Name of the weights file produced (default: {})".format(weights_filename))
    args = parser.parse_args()

    batch_size = args.b
    enc_size = args.d
    max_epochs = args.e
    lr = args.l
    num_workers = args.n
    train_filename = args.r
    test_filename = args.t
    weights_filename = args.w
    print("///// Training for {} epochs with training data from {} and testing data from {}.".format(max_epochs, train_filename, test_filename))
    print("      Writing final weights to {}".format(weights_filename))
    DATA_PATH = "/home/starkj/projects/cda1/training"

    # Load the observation data
    print("///// Loading training dataset...")
    train_data = ObsDataset(train_filename)
    print("      {} data rows ready.".format(len(train_data)))
    print("///// Loading test dataset...")
    test_data = ObsDataset(test_filename)
    print("      {} data rows ready.".format(len(test_data)))

    # Verify GPU availability
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        device = torch.device("cuda:0")
        print("///// Beginning training on GPU")
    else:
        device = torch.device("cpu")
        print("///// Beginning training, but reverting to cpu.")

    # Set up the data loaders
    train_loader = torch.utils.data.DataLoader(train_data, batch_size = batch_size, shuffle = True, num_workers = num_workers)
    num_training_batches = len(train_loader)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size = batch_size, num_workers = num_workers)
    num_testing_batches = len(test_loader)
    print("      Batches per epoch = {} for training, {} for testing".format(num_training_batches, num_testing_batches))

    # Define model, loss function and optimizer
    model = Autoencoder(encoding_size = enc_size)
    model = model.to(device)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(params = model.parameters(), lr = lr)

    # Loop on epochs
    tensorboard = SummaryWriter(DATA_PATH)
    for ep in range(max_epochs):
        train_loss = 0.0
        model.train()

        # Loop on batches of data records
        for batch in train_loader:
            batch = batch.to(device)

             # Perform the learning step
            optimizer.zero_grad()
            output = model(batch)
            loss = loss_fn(output, batch)    # compare to the original input data
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Compute the avg loss over the epoch
        train_loss /= num_training_batches
        tensorboard.add_scalar("training_loss", train_loss)

        # Evaluate performance against the test dataset
        test_loss = 0.0
        model.eval()
        for batch in test_loader:
            batch = batch.to(device)
            output = model(batch)
            loss = loss_fn(output, batch)
            test_loss += loss.item()

        # Compute the avg test loss
        test_loss /= num_testing_batches
        tensorboard.add_scalar("test_loss", test_loss)
        print("Epoch {}: train loss = {:.7f}, test loss = {:.7f}".format(ep, train_loss, test_loss))

    # Summarize the run and store the encoder weights
    print("///// All data collected.  {} epochs complete.".format(ep+1))

    torch.save(model.state_dict(), weights_filename)
    print("      Model weights saved to {}".format(weights_filename))



######################################################################################################
######################################################################################################

if __name__ == "__main__":
   main(sys.argv)
