# Dynabench: A benchmark dataset for learning dynamical systems from mesh data

This is the repository containing the data generation algorithms as well as all baseline models for the __Dynabench: A benchmark dataset for learning dynamical systems from data__ paper (not out yet!)

There are two main tasks for which the data can be used:
1. Forecasting - predicting the next state(s) of the system
2. Evolution - predicting the evolution rate of the system (first derivative with respect to time of the states)

The dataset consists of four different equations selected to be both sufficiently complex, as well as sufficiently variable to simulate different physical systems (first and second order, coupled equations, stationary and non-statinary).

An example (wave equation) of a simulated system is shown below:

![Wave example gif](demos/equation_example_wave.gif)

## Task description
ToDo

## Equations
There are four different equations in the dataset, each with different characteristics summarized in the following table:

| Equation             | Components | Time Order | Spatial Order |
|----------------------|------------|------------|---------------|
| Gas Dynamics         | 4          | 1          | 2             |
| Wave                 | 1          | 2          | 2             |
| Brusselator          | 1          | 1          | 2             |
| Kuramoto Sivashinsky | 1          | 1          | 4             |

## Setup
### Automated setup
If needed create a virtual environment.
You can then install all dependencies by running 

    source scripts/install_requirements.sh

from the main project directory

### Manual installation
Alternatively you can manually install the dependencies from the `requirements.txt` file:

    pip install -r requirements.txt

It is recommended to first create a virtual environment with:

    python -m venv venv
    source venv/bin/activate

Additionally you need to install pytorch geometric, following the instructions on [their website](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html).



## Generation
To generate data simply run

    python gen.py --num NUM_SIMULATIONS [OPTIONAL --seed SEED]

Where `NUM_SIMULATIONS` indicates how many times each equation is simulated. A reasonable value for testing and debugging should be 10. The full benchmark dataset should be used with the default seed and 100 simulations.

Warning! Depending on your CPU speed this can take a long time (~30 minutes for NUM_SIMULATIONS=10)

## Usage
The repository contains two dataset classes to handle the generated data.

1. A pytorch dataset class, where each sample has the form $X\in\mathbb{R}^{K\times N}$, where K is the number of target variables and N is the number of points. See documentation of the dataset for details

2. A graph dataset, specifically used for Message Passing Neural Networks implemented using the pytorch geometric module

