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

    # Define the correct layers to look at and extract them into a layer record
    two_layer_size = ObsVec.SENSOR_DATA_SIZE // 2
    layer_id = ObsVec.BASE_PVMT_TYPE - ObsVec.BASE_SENSOR_DATA
    if model_vehicles:
        layer_id = ObsVec.BASE_OCCUPANCY - ObsVec.BASE_SENSOR_DATA
    print("two_layer_size = {}, data_record shape = {}, layer_id = {}".format(two_layer_size, data_record.shape, layer_id))

    layer_record = data_record[:, layer_id : layer_id + two_layer_size]
    assert layer_record.shape[1] == two_layer_size, "///// ERROR: layer_record shape is {}".format(layer_record.shape)

    # Evaluate performance against the test dataset
    model.eval()
    output = model(layer_record)
    eval_loss = loss_fn(output, layer_record).item()

    # Display the comparison - the input & output are 2D tensors, with the first dim representing a batch of size 1
    label = "Model: {}, on record {} of {}. Loss = {:.6f}".format(weights_filename, data_index, data_filename, eval_loss)
    print_obs(layer_record[0], output[0], layer_id, label)


def print_obs(input     : torch.Tensor, #the input observation record
              output    : torch.Tensor, #the output observation record
              layer_id  : int,          #base index of the first layer to be displayed
              label     : str,          #description of what is being displayed
             ) -> None:
    """Prints a character map of 2 layers of the observation grid, showing embedding input next to its output
        for visual comparison.
    """

    print("\n{}\n".format(label))

    # Visual layers
    #..................!!!!!!!!!!!!!.......................!!!!!!!!!!!!!!
    print("                Input                               Output")
    print("                -----                               ------\n")

    print("print_obs: layer_id = {}".format(layer_id))
    if layer_id == 0: #print pavement layers
        offset = ObsVec.BASE_PVMT_TYPE #because only 2 layers of sensor data are present from the full obs vector
        print("Pavement type (-1 = no pavement, 0 = exit ramp, 1 = through lane):\n")
        display_layer(input, output, ObsVec.BASE_PVMT_TYPE - offset, True, -1.0)

        print("\n\nSpeed limit (normalized by max_speed):\n")
        display_layer(input, output, ObsVec.BASE_SPD_LIMIT - offset, True, 0.0)

    else: #print vehicles data
        offset = ObsVec.BASE_OCCUPANCY #because only the final 2 layers appear in a data record for vehicles
        print("offset = {}. input tensor = ".format(offset))
        print(input)
        print("\n\nOccupied (1 = at least partially occupied, 0 = empty):\n")
        display_layer(input, output, ObsVec.BASE_OCCUPANCY - offset, True, 0.0)

        print("\n\nRelative speed ((neighbor speed - host speed)/max_speed):\n")
        display_layer(input, output, ObsVec.BASE_REL_SPEED - offset)


def display_layer(input:        torch.Tensor,   #input data record
                  output:       torch.Tensor,   #output data record
                  layer_base:   int,            #base index of the layer to display
                  use_empty:    bool = True,    #should the empty_val be used to indicate an empty cell?
                  empty_val:    float = 0.0,    #if value < empty_val, the cell will show as empty rather than the numeric value
                 ) -> None:
    """Prints the content of the input & output values of a single layer for visual comparison."""

    # Build each row in the layer for both input and output
    EMPTY_THRESH = 0.03
    for row in range(ObsVec.NUM_ROWS):
        z = ObsVec.NUM_ROWS - 1 - row
        c0 = layer_base + z
        c1 = layer_base +   ObsVec.NUM_ROWS + z
        c2 = layer_base + 2*ObsVec.NUM_ROWS + z
        c3 = layer_base + 3*ObsVec.NUM_ROWS + z
        c4 = layer_base + 4*ObsVec.NUM_ROWS + z
        if row == 24:
            print("display_layer: layer_base = {}, input shape = {}, z = {}, c0 = {}, c2 = {}, c4 = {}".format(layer_base, input.shape, z, c0, c2, c4))

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
