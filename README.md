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


## Benchmark Results (will be updated)

| equation\model    |   brusselator |   gas_dynamics |   kuramoto_sivashinsky |        wave |
|:------------------|--------------:|---------------:|-----------------------:|------------:|
| feast             |   0.000708671 |    0.000586686 |            0.000104922 | 3.05878e-05 |
| gat               |   0.0370567   |    0.0429969   |            0.0967137   | 0.0116138   |
| gcn               |   0.284355    |    0.181616    |            0.484141    | 0.0392228   |
| persistence       |   0.0404166   |    0.0089862   |            0.00136208  | 0.000278237 |
| point_gnn         |   0.000164287 |    0.000201561 |            3.13238e-05 | 8.14812e-06 |
| point_net         |   1.00087     |    0.171653    |            1.00321     | 0.988587    |
| point_transformer |   0.000556881 |    0.000220286 |            0.000213371 | 1.84354e-05 |
| zero              |   1.00087     |    0.993006    |            1.00318     | 0.988589    |