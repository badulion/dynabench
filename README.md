# Dynabench: A benchmark dataset for learning dynamical systems from mesh data

This is the repository containing the data generation algorithms as well as all baseline models for the __Dynabench: A benchmark dataset for learning dynamical systems from data__ paper (not out yet!)

DynaBench is a benchmark dataset for learning dynamical systems from data. Dynamical systems are physical systems that are typically modelled by partial differential equations (e.g. numerical weather prediction, climate models, fluid simulation, electromagnetic field simulation etc.). The main challenge of learning to predict the evolution of these systems from data is the chaotic behaviour that these systems show (small deviation from the initial conditions leads to highly different predictions) as well as data availability. In real world settings only low-resolution data is available, with measurements sparsly scattered in the simulation domain (see following figure illustrating the distribution of weather monitoring stations in europe).


![Weather stations europe gif](demos/weather_stations.gif)

In this benchmark we try to simulate this setting using synthetic data for easier evaluation and training of different machine learning models. To this end we generated simulation data by solving five different PDE systems which were then postprocessed to create low-resolution snapshots of the simulation.

There are two main tasks for which the data can be used:
1. Forecasting - predicting the next state(s) of the system
2. Evolution - predicting the evolution rate of the system (first derivative with respect to time of the states)

The five included different equations were selected to be both sufficiently complex, as well as sufficiently variable to simulate different physical systems (first and second order, coupled equations, stationary and non-statinary).

An example (wave equation) of a simulated system is shown below:

![Wave example gif](demos/equation_example_wave.gif)


## Equations
There are four different equations in the dataset, each with different characteristics summarized in the following table:

| Equation             | Components | Time Order | Spatial Order |
|----------------------|------------|------------|---------------|
| Advection            | 1          | 1          | 1             |
| Reaction-Diffusion   | 2          | 1          | 2             |
| Gas Dynamics         | 4          | 1          | 2             |
| Kuramoto-Sivashinsky | 1          | 1          | 4             |
| Wave                 | 1          | 2          | 2             |

## Setup
### Automated setup
If needed create a virtual environment and activate it
You can then install all dependencies by running 

    sh scripts/install_requirements.sh

from the main project directory

### Manual installation
Alternatively you can manually install the dependencies from the `requirements.txt` file:

    pip install -r requirements.txt

It is recommended to first create a virtual environment, for example:

    python -m venv venv
    source venv/bin/activate

Additionally you need to install pytorch geometric, following the instructions on [their website](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html).



## Generation
To generate the data used in our experiments (recommended) run
    sh scripts/generate_data.sh

Warning: this can take a long time (> 1h).

You can also generate specific parts of the data by running

    python generate.py num_simulations=NUM_SIMULATIONS equation=EQUATION

Where `NUM_SIMULATIONS` indicates how many times each equation is simulated and `EQUATION` is one of (brusselator, gas_dynamics, kuramoto_sivashinsky, wave). A minimal number simulations for testing and debugging should be 3. The full benchmark dataset should be used with the default seed and 30 simulations.

Warning! Depending on your CPU speed this can take a long time (~10 minutes for NUM_SIMULATIONS=10)

## Usage
To reproduce the experiments from our paper run:
    python main.py equation=EQUATION model=MODEL support=cloud num_points=NUM_POINTS task=TASK

This will start the training for a specific setting. The parameters specify which model, task, support structure, number of points etc. should be run. The available choices of parameters are:

    EQUATION = [brusselator, gas_dynamics, kuramoto_sivashinsky, wave, advection]

    MODEL = [persistence, zero, difference, point_gnn, point_net, point_transformer, gat, gcn, feast]

    NUM_POINTS = [high, low]

    TASK= [forecast, evolution]


Additionally, to use the benchmark for your own research use the included datasets.The repository contains two dataset classes to handle the generated data.

1. A pytorch dataset class, where each sample has the form $X\in\mathbb{R}^{L\times N\times D}$, where L is the lookback, N is the number of points and D is the number of target variables. See documentation of the dataset for details. To initialize the dataset class:

```python
DynaBenchBase(
    mode: str = 'train',
    equation: str = 'gas_dynamics',
    task: str = 'forecast',
    support: str = 'grid',
    num_points: str = 'high',
    base_path: str = 'data',
    lookback: int = 1,
    rollout: int = 1,
    test_ratio: float = 0.1,
    val_ratio: float = 0.1,
    merge_lookback: bool = True,
    *args,
    **kwargs
)
```

Initializes a pytorch dataset with selected parameters. The data is loaded lazily. 



**Args:**

- <b>`mode`</b> (str, optional):  the selection of data to use (train/val/test). Defaults to "train". 
- <b>`equation`</b> (str, optional):  the equation to use. Defaults to "gas_dynamics". 
- <b>`task`</b> (str, optional):  Which task to use as targets. Defaults to "forecast". 
- <b>`support`</b> (str, optional):  Structure of the points at which the measurements are recorded. Defaults to "grid". 
- <b>`num_points`</b> (str, optional):  Number of points at which measurements are available. Defaults to "high". 
- <b>`base_path`</b> (str, optional):  location where the data is stored. Defaults to "data". 
- <b>`lookback`</b> (int, optional):  How many past states are used to make the prediction. The additional states can be concatenated along the channel dimension if merge_lookback is set to True. Defaults to 1. 
- <b>`rollout`</b> (int, optional):  How many steps should be predicted in a closed loop setting. Only used for forecast task. Defaults to 1. 
- <b>`test_ratio`</b> (float, optional):  What fraction of simulations to set aside for testing. Defaults to 0.1. 
- <b>`val_ratio`</b> (float, optional):  What fraction of simulations to set aside for validation. Defaults to 0.1. 
- <b>`merge_lookback`</b> (bool, optional):  Whether to merge the additional lookback information into the channel dimension. Defaults to True. 


2. A graph dataset, specifically used for Message Passing Neural Networks implemented using the [Pytorch Geometric](https://pytorch-geometric.readthedocs.io/) module. It has a similar structure as the base DynaBench dataset.


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
