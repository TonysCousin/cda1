# CDA1 (Bridgit)
**Multiple AI agents driving automated vehicles**

This project builds on the [CDA0 project](https://github.com/TonysCousin/cda0), which trained an AI automated vehicle to drive a simple highway ramp merge scenario in traffic.
In this project we extend the environment model to handle a generic vehicle model for each of the vehicles in the scenario.
Therefore, each vehicle is capable of running a different control algorithm, which could be an AI agent or something else.
This new environment could be used to duplicate the CDA0 project by using that AI agent in one of the vehicles and its "neighbor vehicle" algo in the others.
However, the goal here is to train an AI agent to drive all of the vehicles at once, and have it learn something about safe driving and anticipating others' actions in the process.
Again, as in CDA0, there is no communication between vehicles, and all sensors are treated as perfect.

This project is built on the [Ray](https://www.ray.io) platform using the [PyTorch](https://pytorch.org) framework on Ubuntu.


## Project Description
A statement of the detailed software requirements [can be found here](docs/cda1_rqmts.txt).

Training and testing is currently being performed with the _Roadway B_ test track, shown here. It is a 3 km long freeway segment with several speed limits, merges and lane drops, to exercise lane change maneuvering and, eventually, cooperative behavior.

![Roadway B](docs/images/RoadwayB_map.png)

## Installation & Running

#### To install this software, follow these steps:

0. It is recommended to create a virtual environment (venv or conda) in order to be assured the supporting libraries are exactly what you need, and do not interfere with othe projects. In this case, create your environment with Python 3.10 (Ray currently does not fully support Python 3.11). Then activate the environment you have created.
1. Clone this repo (e.g. `git clone https://github.com/TonysCousin/cda1 .`), then cd into the directory where it is located
2. `pip install -r requirements1.txt`
3. `pip install -r requirements2.txt`

My apologies for the two requirements files, but the global option used in the first prevents the Ray installation (in the second file), and I'm not enough of a pip expert to resolve that conflict in a single requirements file.

#### Training the Agent

For the time being, there is a single AI agent to be trained, as a deep neural network.
It provides the _tactical guidance_ for the ego vehicle.
We consider _strategic guidance_ to be the more abstract decision making.
Given a route, or desired destination on the map, strategic guidance decides what maneuvers are needed to get from the current location to that destination.
Given a chosen set of maneuvers, tactical guidance is the process of deciding how to execute the immediately needed maneuvers (both longitudinally and laterally).
For completeness, a still lower level of decision making is the _controller_, which, in a real vehicle, would translate maneuvering commands into hardware actuation.
In this simulation we can assume that a control layer is in place to make the vehicle move, but do not attempt to simulate those details.
Rather, simple code is in place (the `Vehicle` class) to model longitudinal motion that respects acceleration and jerk limits, and lateral motion that only moves from one lane to an adjacent lane at a reasonable rate (i.e. it takes several time steps to do a lane change).
There is also procedural code that preforms very simplistic strategic guidance, as that part is not our focus yet.
It simply compares the current ego vehicle location with the location of acceptable destination points and outputs a "desirability" value for each of the three immediate lane options: moving one lane to the left, staying in the same lane, or moving one lane to the right.
These desirability values are provided as input to the tactical guidance AI agent.

#### Perception Training

The Bridgit tactical guidance agent actually consists of 3 NNs.
Initial attempts to structure it as a single MLP resulted in unsatisfactory learning.
The agent's input (observation) vector contains 525 elements, 500 of which are simulated sensor data, in 4 conceptual layers.
It seems the sensor data was washing out the inputs from the other 25 elements, which represent basic vehicle state and some command history.
Therefore, the solution involves two additional NNs to provide a perception capability by pre-processing the sensor layers to effectively compress that data into a smaller number of data elements.
The first perception NN takes in two sensor layers (pavement type and speed limits, which can vary from lane to lane), and the second perception NN akes in the other two layers (neighbor vehicle locations and speeds).
Both of these provide vector embedding of their assigned sensor data.
That is, they are each the encoder half of an autoencoder, which needed to be trained to reproduce the sensor data with reasonable accuracy.

The first requirement to build vector embeddings is to have lots of labeled data that the autoencoder can attempt to reproduce.
This is accomplished by running
  `python embed_collect.py`
with appropriate command line options (explained with `--help`).
It generates a large `.csv` database of just sensor data (500 elements per row) by running a collection bot vehicle all over the test track (with lots of other bots to provide traffic).
Once this database is created, it needs to be split into a training and testing datasets.
This can easily be done with the `splitter.sh` script (note that it needs to be hand edited first, to specify the number of rows used for testing).

Training the vector embedding models is done by running the supervised learning algorithm:
  `python embed_train.py`
with the appropriate command line options (explained with `--help`).
Each run will train one of the two required embeddings.

To evaluate the performance of the new embedding models, run
  `python embed_eval.py`
with the appropriate command line options (explained with `--help`).
It shows a nice visual side-by-side comparison of the original sensor data (the labeled target) and the model's reproduction, to easily see its performance on any number of traffic situations.

Once the embeddings have been decided, their checkpoint files (written to .pt files) need to be used by the BridgitNN model.
To do this, edit the code in `bridgit_nn.py` to read those files in (one for the pavement embedding and one for the vehicles embedding).
The Bridgit tactical guidance agent splits the full observation vector (525 elements) into three parts, sending half the sensor elements to each of these two embedding NNs, and the remaining portion (macro data) through a separate linear layer.
It then combines the outputs of these 3 processes into an additional 3 linear layers, which ultimately output two maneuver actions (speed command, lane change command).

#### Training the Bridgit Agent

With the BridgitNN updated to use the trained perception embedding networks, the whole agent can now be trained with reinforcement learning to drive in (theoretically) any situation within the given roadway.
This RL training is accomplished by running
  `python train.py`
with the appropriate command line options (explained with `--help`).
This is a long process that can run for many hours, depending on the hardware.
It may be necessary to edit the `train.py` code, as it includes config hyperparameters for resource usage, which may need to be tuned to your particular system.
This training uses the **SAC** algorithm, and is quite robust to changes in the non-resource hyperparameters; the primary consideration is number of iterations required.

#### Inference Behavior

With a completely trained Bridgit agent, it can now be run in inference mode for some fun!
The provided inference program provides a 2D graphical display of the roadway and the motion of all vehicles in the scenario.
Running
  `python inference.py`
with the appropriate command line options (explained with `--help`) can be used to choose a variety of scenarios (initial conditions).
The number and types of vehicles available for any given run (both training and inference) can be specified with a vehicle config file.
Two config files are currently available, but any new file can be created and pointed to from within the config parameters section of both `train.py` and `inference.py`.

## License & Authorship
This software is published under the [Apache 2.0 open source license](LICENSE), and can be used and distributed according to the terms of that license.
The software was created by John Stark.

## Project Progress
Latest code ready for public use is on the _master_ branch.

Latest working code is on the _develop_ branch. It may be lacking complete functionality, contain experimental work, or be somewhat inconsistent, but is generally sound.

In-work features & fixes are on other branches off of _develop_.

This project was begun on 8/21/23, and is proceeding in the following direction:
- Part 1:  Train a single agent to drive the new track to suitable destinations with several bots attempting to do the same thing. **COMPLETE ON 12/29/23**
- Part 2:  Expand the training to include several instances of the agent (replacing some bots) as pseudo-multi-agent training (iterative).
- Part 3:  True multi-agent training, with several untrained agents (same policy) learning simultaneously using Ray multi-agent facilities.
- Part 4:  Multi-agent training for two policies in a single vehicle, one for planning and one for control.
