from cmath import inf
import sys
import argparse
import torch
import torch.nn as nn

from obs_vec import ObsVec
from embed_support import ObsDataset, Autoencoder


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
    model_vehicles = False
    program_desc = "Trains the auto-encoder for vector embedding in the cda1 project."
    parser = argparse.ArgumentParser(prog = argv[0], description = program_desc, epilog = "Will run until either max episodes or max timesteps is reached.")
    parser.add_argument("-b", type = int, default = batch_size, help = "Number of data rows in a training batch (default = {})".format(batch_size))
    parser.add_argument("-d", type = int, default = enc_size, help = "Number of neurons in the encoding layer (default = {})".format(enc_size))
    parser.add_argument("-e", type = int, default = max_epochs, help = "Max number of epochs to train (default = {})".format(max_epochs))
    parser.add_argument("-l", type = float, default = lr, help = "Learning rate (default = {})".format(lr))
    parser.add_argument("-n", type = int, default = num_workers, help = "Number of training worker processes (default = {})".format(num_workers))
    parser.add_argument("-r", type = str, default = train_filename, help = "Filename of the training observation dataset (default: {})".format(train_filename))
    parser.add_argument("-t", type = str, default = test_filename, help = "Filename of the test observation dataset (default: {})".format(test_filename))
    parser.add_argument("-v", action = "store_true", help = "Model vehicle layers? (if not used, roadway layers will be modeled)")
    parser.add_argument("-w", type = str, default = weights_filename, help = "Name of the weights file produced (default: {})".format(weights_filename))
    args = parser.parse_args()

    batch_size = args.b
    enc_size = args.d
    max_epochs = args.e
    lr = args.l
    num_workers = args.n
    train_filename = args.r
    test_filename = args.t
    model_vehicles = args.v
    weights_filename = args.w
    print("///// Training for {} epochs with training data from {} and testing data from {}. Modeling vehicle layers: {}"
          .format(max_epochs, train_filename, test_filename, model_vehicles))
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

    # Define which layers of sensor data we will be looking at
    layer_min = 0 #pavement data
    layer_max = 1 #roadway speed limits
    if model_vehicles:
        layer_min = 2 #neighbor vehicle presence
        layer_max = 3 #neighbor vehicle rel speed

    # Set up to track test loss performance for possible early stopping
    test_loss_min = inf

    # Loop on epochs
    #tensorboard = SummaryWriter(DATA_PATH)
    for ep in range(max_epochs):
        train_loss = 0.0
        model.train()

        # Loop on batches of data records
        for bn, batch in enumerate(train_loader):

            # Trim and reshape the batch so that we only get the desired data layers from each record
            reshaped_batch = reshape_batch(batch, batch_size, layer_min, layer_max)
            reshaped_batch = reshaped_batch.to(device)

             # Perform the learning step
            optimizer.zero_grad()
            output = model(reshaped_batch)
            loss = loss_fn(output, reshaped_batch)    # compare to the original input data
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Compute the avg loss over the epoch
        train_loss /= num_training_batches
        print("training of epoch {} complete".format(ep))
        #tensorboard.add_scalar("training_loss", train_loss)

        # Evaluate performance against the test dataset
        test_loss = 0.0
        model.eval()
        for batch in test_loader:
            reshaped_batch = reshape_batch(batch, batch_size, layer_min, layer_max)
            reshaped_batch = reshaped_batch.to(device)
            output = model(reshaped_batch)
            loss = loss_fn(output, reshaped_batch)
            test_loss += loss.item()

        # Compute the avg test loss
        test_loss /= num_testing_batches
        #tensorboard.add_scalar("test_loss", test_loss)
        regression = 0.0
        regression_msg = " "
        if test_loss < test_loss_min:
            test_loss_min = test_loss
        else:
            regression = 100.0*(test_loss - test_loss_min) / test_loss_min
            regression_msg = "{:.2f}% above min".format(regression)
        print("Epoch {:4d}: train loss = {:.7f}, test loss = {:.7f} {}".format(ep, train_loss, test_loss, regression_msg))

        # Early stopping if the test loss has increased significantly
        if regression >= 5.0:
            print("///// Early stopping due to test loss increase.")
            break

    # Summarize the run and store the encoder weights
    print("///// All data collected.  {} epochs complete.".format(ep+1))

    torch.save(model.state_dict(), weights_filename)
    print("      Model weights saved to {}".format(weights_filename))


def reshape_batch(batch         : torch.Tensor, #a batch of 1D vectors (as defined in ObsVec)
                  batch_size    : int,          #num data records in the batch
                  layer_min     : int,          #lower layer number that will be transposed
                  layer_max     : int,          #upper lwyer num that will be transposed
                 ) -> torch.Tensor:
    """Copies a data batch into a new tensor with a different shape that only represents 2 of the 4 layers."""

    num_cols = 5
    col_len = ObsVec.BASE_L - ObsVec.BASE_LL
    num_rows = ObsVec.ZONES_BEHIND + 1 + ObsVec.ZONES_FORWARD

    reshaped_batch = torch.Tensor(batch_size, 2, num_cols, num_rows) #always 2 layers
    for b in range(batch_size):
        for layer in range(layer_min, layer_max+1):
            for c in range(num_cols):
                for r in range(num_rows):
                    try:
                        index = c*col_len + r*ObsVec.NORM_ELEMENTS + layer
                        reshaped_batch[b, layer, c, r] = batch[b, index]
                    except IndexError as e:
                        print("IndexError trapped in reshape_batch: b = {}, layer = {}, c = {}, r = {}, index = {}, batch shape = {}"
                              .format(b, layer, c, r, index, batch.shape))
                        raise e

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



######################################################################################################
######################################################################################################

if __name__ == "__main__":
   main(sys.argv)
