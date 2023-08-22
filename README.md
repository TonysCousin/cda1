# CDA1 (Bridgit)
**Multiple AI agents driving automated vehicles.**

This project builds on the [CDA0 project](https://github.com/TonysCousin/cda0) which trained an AI automated vehicle to drive a simple highway ramp merge scenario in traffic.
In this project we will extend the environment model to handle a generic vehicle model for each of the vehicles in the scenario.
Therefore, each vehicle is capable of running a different control algorithm, which could be an AI agent or something else.
This new environment could be used to duplicate the CDA0 project by using that AI agent in one of the vehicles and its "neighbor vehicle" algo in the others.
However, the goal here is to train an AI agent to drive all of the vehicles at once, and have it learn something about safe driving and anticipating others' actions in the process.
Again, as in CDA0, there is no communication between vehicles, and all sensors are perfect.

## Detailed Requirements
A statement of the detailed software requirements [can be found here](docs/cda1_rqmts.txt).

## Project Progress
**8/21/23** - Just getting started. New definition docs & skeleton code should appear shortly.

## License & Authorship
This software is published under the [Apache 2.0 open source license](LICENSE), and can be used and distributed according to the terms of that license.
It was written by John Stark.
