from cmath import inf
import sys
import argparse
import torch
import torch.nn as nn
from datetime import datetime

from obs_vec import ObsVec
from hp_prng import HpPrng
from embed_support import ObsDataset, Autoencoder, reshape_batch


def main(argv):
    """This program evaluates an autoencoder by comparing its input and output graphically (they should be identical) on selected rows from
        the given dataset.
    """

    # Handle any args
    weights_filename = "embedding_weights.pt"
    data_filename = "test.csv"
    data_index = None
    enc_size = 50
    model_vehicles = False
    program_desc = "Evaluates an auto-encoder for vector embedding in the cda1 project."
    parser = argparse.ArgumentParser(prog = argv[0], description = program_desc, epilog = "Encoding dim (-d) must match the structure of the weight file.")
    parser.add_argument("-d", type = int, default = enc_size, help = "Number of neurons in the encoding layer (default = {})".format(enc_size))
    parser.add_argument("-i", type = int, default = data_index, help = "Index of data row to evaluate (default = {})".format(data_index))
    parser.add_argument("-s", type = str, default = data_filename, help = "Filename of the dataset (default = {})".format(data_filename))
    parser.add_argument("-v", action = "store_true", help = "Model vehicle layers? (if not used, roadway layers will be modeled)")
    parser.add_argument("-w", type = str, default = weights_filename, help = "Name of the weights file (default: {})".format(weights_filename))
    args = parser.parse_args()

    enc_size = args.d
    data_index = args.i
    data_filename = args.s
    model_vehicles = args.v
    weights_filename = args.w

    # Load the observation data
    print("///// Loading dataset...")
    dataset = ObsDataset(data_filename)
    print("      {} data rows ready.".format(len(dataset)))

    # Retrieve the desired record
    if data_index is None:
        seed = datetime.now().microsecond
        prng = HpPrng(seed = seed)
        data_index = int(prng.random() * len(dataset))
        if data_index == 0: #row 0 is the column header labels
            data_index = 1
    data_record = torch.from_numpy(dataset[data_index]).view(1, -1)

    # Define model and load its parameters
    model = Autoencoder(encoding_size = enc_size)
    model.load_state_dict(torch.load(weights_filename))
    loss_fn = nn.MSELoss()

    # Define the correct layers to look at
    layer_min = 0
    layer_max = 1
    if model_vehicles:
        layer_min = 2
        layer_max = 3

    # Pull the desired layer data out of the data row
    reshaped_batch = reshape_batch(data_record, 1, layer_min, layer_max)

    # Evaluate performance against the test dataset
    model.eval()
    output = model(reshaped_batch)
    eval_loss = loss_fn(output, reshaped_batch).item()

    # Display the comparison
    label = "Model: {}, on record {} of {}. Loss = {:.6f}".format(weights_filename, data_index, data_filename, eval_loss)
    print_obs(reshaped_batch[0], output[0], layer_min, layer_max, label)


def print_obs(input     : torch.Tensor, #the input observation record
              output    : torch.Tensor, #the output observation record
              layer_min : int,          #first layer index to be displayed
              layer_max : int,          #final layer index to be displayed
              label     : str,          #description of what is being displayed
             ) -> None:
    """Prints a character map of each layer of the observation grid, showing embedding input next to its output
        for visual comparison.
    """

    print("\n{}\n".format(label))

    # Visual layers
    #..................!!!!!!!!!!!!!.......................!!!!!!!!!!!!!!
    print("                Input                               Output")
    print("                -----                               ------\n")

    if 0 in [layer_min, layer_max]:
        print("Pavement type (-1 = no pavement, 0 = exit ramp, 1 = through lane):\n")
        display_layer(input, output, 0, True, -1.0)

    if 1 in [layer_min, layer_max]:
        print("\n\nSpeed limit (normalized by max_speed):\n")
        display_layer(input, output, 1, True, 0.0)

    if 2 in [layer_min, layer_max]:
        print("\n\nOccupied (1 = at least partially occupied, 0 = empty):\n")
        display_layer(input, output, 0, True, 0.0)

    if 3 in [layer_min, layer_max]:
        print("\n\nRelative speed ((neighbor speed - host speed)/max_speed):\n")
        display_layer(input, output, 1)


def display_layer(input:    torch.Tensor,   #input data record
                  output:   torch.Tensor,   #output data record
                  layer:    int,            #index of the layer to display
                  use_empty:bool = True,    #should the empty_val be used to indicate an empty cell?
                  empty_val:float = 0.0,    #if value < empty_val, the cell will show as empty rather than the numeric value
                 ) -> None:
    """Prints the content of a single pair of layers to compare the specified layer of the input record to that of the output record."""

    # Since the incoming tensors only represent half of the full data record (every other data element), we cannot use the
    # indexing constants from ObsVec here. The LL column starts at index 0, and each column is half as long as it is in ObsVec.
    # Also, the reshaping has reordered things so that each layer is clustered together.
    COL_LEN = (ObsVec.ZONES_BEHIND + 1 + ObsVec.ZONES_FORWARD) #a single layer represented in each column
    BASE_LL = layer * 5*COL_LEN

    # Build each row in the layer for both input and output
    EMPTY_THRESH = 0.03
    for row in range(25):
        z = 24 - row
        c0 = BASE_LL + z
        c1 = BASE_LL +   COL_LEN + z
        c2 = BASE_LL + 2*COL_LEN + z
        c3 = BASE_LL + 3*COL_LEN + z
        c4 = BASE_LL + 4*COL_LEN + z

        # Initialize the row display with all empties
        in_row = ["  .  ", "  .  ", "  .  ", "  .  ", "  .  "]
        out_row = ["  .  ", "  .  ", "  .  ", "  .  ", "  .  "]

        # Populate the row of the input layer
        if (use_empty and not (empty_val-EMPTY_THRESH <= input[c0] <= empty_val+EMPTY_THRESH))  or  not use_empty:
            in_row[0] = "{:5.2f}".format(input[c0])
        if (use_empty and not (empty_val-EMPTY_THRESH <= input[c1] <= empty_val+EMPTY_THRESH))  or  not use_empty:
            in_row[1] = "{:5.2f}".format(input[c1])
        if (use_empty and not (empty_val-EMPTY_THRESH <= input[c2] <= empty_val+EMPTY_THRESH))  or  not use_empty:
            in_row[2] = "{:5.2f}".format(input[c2])
        if (use_empty and not (empty_val-EMPTY_THRESH <= input[c3] <= empty_val+EMPTY_THRESH))  or  not use_empty:
            in_row[3] = "{:5.2f}".format(input[c3])
        if (use_empty and not (empty_val-EMPTY_THRESH <= input[c4] <= empty_val+EMPTY_THRESH))  or  not use_empty:
            in_row[4] = "{:5.2f}".format(input[c4])

        # Populate this row of the output layer
        if (use_empty and not (empty_val-EMPTY_THRESH <= output[c0] <= empty_val+EMPTY_THRESH))  or  not use_empty:
            out_row[0] = "{:5.2f}".format(output[c0])
        if (use_empty and not (empty_val-EMPTY_THRESH <= output[c1] <= empty_val+EMPTY_THRESH))  or  not use_empty:
            out_row[1] = "{:5.2f}".format(output[c1])
        if (use_empty and not (empty_val-EMPTY_THRESH <= output[c2] <= empty_val+EMPTY_THRESH))  or  not use_empty:
            out_row[2] = "{:5.2f}".format(output[c2])
        if (use_empty and not (empty_val-EMPTY_THRESH <= output[c3] <= empty_val+EMPTY_THRESH))  or  not use_empty:
            out_row[3] = "{:5.2f}".format(output[c3])
        if (use_empty and not (empty_val-EMPTY_THRESH <= output[c4] <= empty_val+EMPTY_THRESH))  or  not use_empty:
            out_row[4] = "{:5.2f}".format(output[c4])

        if row == 20:
            in_row[2] = "*Ego*"
            out_row[2] = "*Ego*"

        print("{:2d} [{} {} {} {} {}]     [{} {} {} {} {}]"
              .format(z, in_row[0], in_row[1], in_row[2], in_row[3], in_row[4], out_row[0], out_row[1], out_row[2], out_row[3], out_row[4]))


######################################################################################################
######################################################################################################

if __name__ == "__main__":
   main(sys.argv)
