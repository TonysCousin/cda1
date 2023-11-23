from cmath import inf
import sys
import argparse
import torch
import torch.nn as nn
from datetime import datetime

from obs_vec import ObsVec
from hp_prng import HpPrng
from embed_support import ObsDataset, Autoencoder


def main(argv):
    """This program evaluates an autoencoder by comparing its input and output graphically (they should be identical) on selected rows from
        the given dataset.
    """

    # Handle any args
    weights_filename = "embedding_weights.pt"
    data_filename = "test.csv"
    data_index = None
    enc_size = 50
    program_desc = "Evaluates an auto-encoder for vector embedding in the cda1 project."
    parser = argparse.ArgumentParser(prog = argv[0], description = program_desc, epilog = "Encoding dim (-d) must match the structure of the weight file.")
    parser.add_argument("-d", type = int, default = enc_size, help = "Number of neurons in the encoding layer (default = {})".format(enc_size))
    parser.add_argument("-i", type = int, default = data_index, help = "Index of data row to evaluate (default = {})".format(data_index))
    parser.add_argument("-s", type = str, default = data_filename, help = "Filename of the dataset (default = {})".format(data_filename))
    parser.add_argument("-w", type = str, default = weights_filename, help = "Name of the weights file (default: {})".format(weights_filename))
    args = parser.parse_args()

    enc_size = args.d
    data_index = args.i
    data_filename = args.s
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
    data_record = torch.from_numpy(dataset[data_index]) #returns np array

    # Define model and load its parameters
    model = Autoencoder(encoding_size = enc_size)
    model.load_state_dict(torch.load(weights_filename))
    print("///// NN model successfully loaded from {}".format(weights_filename))
    loss_fn = nn.MSELoss()

    # Evaluate performance against the test dataset
    model.eval()
    output = model(data_record)
    eval_loss = loss_fn(output, data_record).item()

    # Display the comparison
    label = "Model: {}, on record {} of {}. Loss = {:.6f}".format(weights_filename, data_index, data_filename, eval_loss)
    print_obs(data_record, output, label)


def print_obs(input     : torch.Tensor, #the input observation record
              output    : torch.Tensor, #the output observation record
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

    print("Pavement type (-1 = no pavement, 0 = exit ramp, 1 = through lane):\n")
    display_layer(input, output, 0, True, -0.99)

    print("\n\nSpeed limit (normalized by max_speed):\n")
    display_layer(input, output, 1, True, 0.01)

    print("\n\nOccupied (1 = at least partially occupied, 0 = empty):\n")
    display_layer(input, output, 2, True, 0.01)

    print("\n\nRelative speed ((neighbor speed - host speed)/max_speed):\n")
    display_layer(input, output, 3, False)


def display_layer(input:    torch.Tensor,   #input data record
                  output:   torch.Tensor,   #output data record
                  layer:    int,            #index of the layer to display
                  use_empty:bool = True,    #should the empty_val be used to indicate an empty cell?
                  empty_val:int = 0,        #if value < empty_val, the cell will show as empty rather than the numeric value
                 ) -> None:
    """Prints the content of a single pair of layers to compare the specified layer of the input record to that of the output record."""

    # Create an adjustment to the base index of the obs vector. The BASE constants from ObsVec assume we have the full observation
    # vector in place. For the embedding we are only looking at the sensor data, which is the last block in the observation vector.
    # Therefore, we need to subtract the offset of that block from everything.

    # Build each row in the layer for both input and output
    for row in range(25):
        z = 24 - row
        c0 = ObsVec.BASE_LL + z*ObsVec.NORM_ELEMENTS - ObsVec.BASE_SENSOR_DATA
        c1 = ObsVec.BASE_L + z*ObsVec.NORM_ELEMENTS - ObsVec.BASE_SENSOR_DATA
        c2 = ObsVec.BASE_CTR + z*ObsVec.NORM_ELEMENTS - ObsVec.BASE_SENSOR_DATA
        c3 = ObsVec.BASE_R + z*ObsVec.NORM_ELEMENTS - ObsVec.BASE_SENSOR_DATA
        c4 = ObsVec.BASE_RR + z*ObsVec.NORM_ELEMENTS - ObsVec.BASE_SENSOR_DATA

        # Initialize the row display with all empties
        in_row = ["  .  ", "  .  ", "  .  ", "  .  ", "  .  "]
        out_row = ["  .  ", "  .  ", "  .  ", "  .  ", "  .  "]

        # Populate the row of the input layer
        if (use_empty and input[c0+layer] > empty_val)  or  not use_empty:
            in_row[0] = "{:5.2f}".format(input[c0+layer])
        if (use_empty and input[c1+layer] > empty_val)  or  not use_empty:
            in_row[1] = "{:5.2f}".format(input[c1+layer])
        if (use_empty and input[c2+layer] > empty_val)  or  not use_empty:
            in_row[2] = "{:5.2f}".format(input[c2+layer])
        if (use_empty and input[c3+layer] > empty_val)  or  not use_empty:
            in_row[3] = "{:5.2f}".format(input[c3+layer])
        if (use_empty and input[c4+layer] > empty_val)  or  not use_empty:
            in_row[4] = "{:5.2f}".format(input[c4+layer])

        # Populate this row of the output layer
        if (use_empty and output[c0+layer] > empty_val)  or  not use_empty:
            out_row[0] = "{:5.2f}".format(output[c0+layer])
        if (use_empty and output[c1+layer] > empty_val)  or  not use_empty:
            out_row[1] = "{:5.2f}".format(output[c1+layer])
        if (use_empty and output[c2+layer] > empty_val)  or  not use_empty:
            out_row[2] = "{:5.2f}".format(output[c2+layer])
        if (use_empty and output[c3+layer] > empty_val)  or  not use_empty:
            out_row[3] = "{:5.2f}".format(output[c3+layer])
        if (use_empty and output[c4+layer] > empty_val)  or  not use_empty:
            out_row[4] = "{:5.2f}".format(output[c4+layer])

        if row == 20:
            in_row[2] = "*Ego*"
            out_row[2] = "*Ego*"

        print("{:2d} [{} {} {} {} {}]     [{} {} {} {} {}]"
              .format(z, in_row[0], in_row[1], in_row[2], in_row[3], in_row[4], out_row[0], out_row[1], out_row[2], out_row[3], out_row[4]))


######################################################################################################
######################################################################################################

if __name__ == "__main__":
   main(sys.argv)
