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
To generate data  simply run

    python generate.py --num NUM_SIMULATIONS --equation EQUATION [OPTIONAL --seed SEED]

Where `NUM_SIMULATIONS` indicates how many times each equation is simulated and `EQUATION` is one of (brusselator, gas_dynamics, kuramoto_sivashinsky, wave). A reasonable number of simulations for testing and debugging should be 10. The full benchmark dataset should be used with the default seed and 30 simulations.

Warning! Depending on your CPU speed this can take a long time (~30 minutes for NUM_SIMULATIONS=10)

## Usage
The repository contains two dataset classes to handle the generated data.

1. A pytorch dataset class, where each sample has the form $X\in\mathbb{R}^{K\times N}$, where K is the number of target variables and N is the number of points. See documentation of the dataset for details

2. A graph dataset, specifically used for Message Passing Neural Networks implemented using the pytorch geometric module


## Benchmark Results (will be updated)
The following tables show the results of our experiments

- forecast task, high number of points (1-step MSE):

| model             |   brusselator |   gas_dynamics |   kuramoto_sivashinsky |        wave |
|:------------------|--------------:|---------------:|-----------------------:|------------:|
| feast             |    0.00242319 |     0.0021791  |             0.00128176 | 0.000484132 |
| gat               |    0.0281436  |     0.0367549  |             0.069632   | 0.0118712   |
| gcn               |    0.282337   |     0.140523   |             0.461607   | 0.0422273   |
| persistence       |    0.0332305  |     0.00561406 |             0.00129482 | 0.000264883 |
| point_gnn         |    0.00141922 |     0.00167125 |             0.00120295 | 0.000375454 |
| point_net         |    0.998704   |     0.0384901  |             0.99871    | 0.999058    |
| point_transformer |    0.0017799  |     0.00131779 |             0.00125645 | 0.000419147 |

- forecast task, high number of points (16-step rollout MSE):

| model             |   brusselator |   gas_dynamics |   kuramoto_sivashinsky |      wave |
|:------------------|--------------:|---------------:|-----------------------:|----------:|
| feast             |   0.227424    |       0.255952 |            0.134562    | 0.0339035 |
| gat               |   1.29767     |      43.9229   |            1.31319     | 1.7413    |
| gcn               |   2.46631e+08 |     326.2      |            2.78559e+15 | 8.68158   |
| persistence       |   2.29927     |       0.797735 |            0.28452     | 0.0670084 |
| point_gnn         |   0.133019    |       0.298848 |            0.110021    | 0.0503421 |
| point_net         |   0.999038    |       2.67773  |            0.998887    | 0.998901  |
| point_transformer |   0.166213    |       0.163016 |            0.117403    | 0.0253431 |

- evolution task, high number of points:


| model             |   brusselator |   gas_dynamics |   kuramoto_sivashinsky |     wave |
|:------------------|--------------:|---------------:|-----------------------:|---------:|
| difference        |     0.966056  |      0.937047  |              0.813676  | 0.967245 |
| feast             |     0.0209256 |      0.103211  |              0.132971  | 0.247774 |
| gat               |     0.335062  |      0.169115  |              0.255142  | 0.300955 |
| gcn               |     0.54936   |      0.553218  |              0.598326  | 0.309449 |
| point_gnn         |     0.0124441 |      0.0769728 |              0.110405  | 0.248534 |
| point_net         |     0.930219  |      0.993493  |              0.861911  | 0.994931 |
| point_transformer |     0.0136373 |      0.0649107 |              0.0959964 | 0.238254 |
