# Torchswitch: Distributed Deep Learning Training with Parallelism

## Project Overview
This project aims to explore and compare different parallelism techniques in deep learning training, specifically pipeline parallelism, data parallelism, model paralellism, and a mixture of these all, on a distributed system consisting of multiple CPU and/or GPU nodes. 

This project uses pytorch for deep learning model training operations, and GPipe for pipeline distribution.

## Structure
- `menu.py`: A command-line interface that allows users to select the parallelism mode and the number of nodes.
- `model.py`: Contains a simple convolutional neural network.
- `parapipe.py`: Contains the implementation of pipeline parallelism.
- `paradata.py`: Contains the implementation of data parallelism.
- `paramodel.py`: Contains the implementation of model parallelism.
- `paramix.py`: Manages a mix of techniques.
- `run_paradata_training.sh`: Bash script for distributing the running of `paradata.py` and experimenting with the distributed setup itself, it also outputs log files for reviewing performance on different nodes.

## Status
- `parapipe.py`: Very simple bare-bones implementation, not yet tested.
- `paradata.py`: Implemented, but not yet functioning distributedly (with the assistance of `run_paradata_training.sh`).
- `paramodel.py`: Not implemented yet.
- `paramix.py`: Not implemented yet.

## Goals
- To enable flexible use of distributed resources (CPU/GPU nodes) for deep learning training.
- To measure and compare the performance of different parallelism techniques.
- To find ways of combining different parallelism techniques.
- To provide insights through performance metrics and graphs, helping users understand the impact of various parallelism methods on training efficiency and speed.

## Usage
Ideally the user would run `menu.py` to start the program, but since this hasn't yet been fully implemented, customized scripts (such as `run_paradata_training.sh`) are being used for testing at the moment.

Through the menu, the user will be prompted to select the parallelism mode and the number of nodes. The program will then execute the training using the selected parameters and evaluate the performance.

## Future Enhancements and Work
- Including more alternatives for model selection, varying in complexity.
- Automating the process of running different training setups simultaneously for faster comparison.
- Enabling a mixture of simultaneous different parallelization paradigms for singular training sessions.
- Extending support for more complex models and datasets.
- Overall testing

