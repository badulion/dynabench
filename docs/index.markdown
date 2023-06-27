---
# Feel free to add content and custom Front Matter to this file.
# To modify the layout, see https://jekyllrb.com/docs/themes/#overriding-theme-defaults

subtitle: A benchmark dataset for learning dynamical systems from low-resolution data
---

# Abstract
Previous work on learning physical systems from data has focused on high-resolution grid-structured measurements. However, real-world knowledge of such systems (e.g. weather data) relies on sparsely scattered measuring stations. In this paper, we introduce a novel simulated benchmark dataset, DynaBench, for learning dynamical systems directly from sparsely scattered data without prior knowledge of the equations. The dataset focuses on predicting the evolution of a dynamical system from low-resolution, unstructured measurements. We simulate six different partial differential equations covering a variety of physical systems commonly used in the literature and evaluate several machine learning models, including traditional graph neural networks and point cloud processing models, with the task of predicting the evolution of the system. The proposed benchmark dataset is expected to advance the state of art as an out-of-the-box easy-to-use tool for evaluating models in a setting where only unstructured low-resolution observations are available. The benchmark is available at [https://professor-x.de/dynabench](https://professor-x.de/dynabench).



# Dynabench: A benchmark dataset for learning dynamical systems from low-resolution data

This is the repository containing the data generation algorithms as well as all baseline models for the __Dynabench: A benchmark dataset for learning dynamical systems from low-resolution data__ paper (under review)

DynaBench is a benchmark dataset for learning dynamical systems from data. Dynamical systems are physical systems that are typically modelled by partial differential equations (e.g. numerical weather prediction, climate models, fluid simulation, electromagnetic field simulation etc.). The main challenge of learning to predict the evolution of these systems from data is the chaotic behaviour that these systems show (small deviation from the initial conditions leads to highly different predictions) as well as data availability. In real world settings only low-resolution data is available, with measurements sparsly scattered in the simulation domain (see following figure illustrating the distribution of weather monitoring stations in europe).


![Weather stations europe gif](images/demos/weather_stations.gif)

In this benchmark we try to simulate this setting using synthetic data for easier evaluation and training of different machine learning models. To this end we generated simulation data by solving five different PDE systems which were then postprocessed to create low-resolution snapshots of the simulation.

There main tasks for which the dataset has been generated is forecasting - predicting the next state(s) of the system

The six included different equations were selected to be both sufficiently complex, as well as sufficiently variable to simulate different physical systems (first and second order, coupled equations, stationary and non-statinary).

An example (wave equation) of a simulated system is shown below:

![Wave example gif](images/demos/equation_example_wave.gif)


## Equations
There are four different equations in the dataset, each with different characteristics summarized in the following table:

| Equation             | Components | Time Order | Spatial Order |
|----------------------|------------|------------|---------------|
| Advection            | 1          | 1          | 1             |
| Burgers'             | 2          | 1          | 2             |
| Gas Dynamics         | 4          | 1          | 2             |
| Kuramoto-Sivashinsky | 1          | 1          | 4             |
| Reaction-Diffusion   | 2          | 1          | 2             |
| Wave                 | 1          | 2          | 2             |