# Dynabench: A benchmark dataset for learning dynamical systems from data

This is the repository containing the data generation algorithms as well as all baseline models for the __Dynabench: A benchmark dataset for learning dynamical systems from data__ paper (not out yet!)

There are two main tasks for which the data can be used:
1. Forecasting - predicting the next state(s) of the system
2. Evolution - predicting the evolution rate of the system (first derivative with respect to time of the states)

The dataset consists of four different equations selected to be both sufficiently complex, as well as sufficiently variable to simulate different physical systems (first and second order, coupled equations, stationary and non-statinary).

An example (wave equation) of a simulated system is shown below:

![Wave example gif](figures/equation_example_wave.gif)

## Task description
ToDo

## Equationns
ToDo

## Setup
Right now the dependencies need to be installed manually from the `requirements.txt` file:

    pip install -r requirements.txt

It is recommended to first create a virtual environment with:

    python -m venv venv
    source venv/bin/activate

If you have troubles installing the pytorch geometric library, follow the instructions on [their website](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html).

> To setup the environment with all necessary requirements simply run
> python setup.py
> This creates a new 
### Currently not working!


## Generation
To generate data simply run

    python gen.py --num NUM_SIMULATIONS [OPTIONAL --seed SEED]

Where `NUM_SIMULATIONS` indicates how many times each equation is simulated. A reasonable value for testing and debugging should be 10. The full benchmark dataset should be used with the default seed and 100 simulations.

Warning! Depending on your CPU speed this can take a long time (~30 minutes for NUM_SIMULATIONS=10)

## Usage
The repository contains two dataset classes to handle the generated data.

1. A pytorch dataset class, where each sample has the form $X\in\mathbb{R}^{K\times N}$, where K is the number of target variables and N is the number of points. See documentation of the dataset for details

2. A graph dataset, specifically used for Message Passing Neural Networks implemented using the pytorch geometric module

