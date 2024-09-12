##########################
Papers using the benchmark
##########################

Here we give an overview of all the papers that have used the DynaBench benchmark in their research.


*********
DynaBench
*********

**Full title**: DynaBench: A benchmark dataset for learning dynamical systems from low-resolution data.

The original DynaBench paper can be found `here <https://arxiv.org/abs/2306.05805>`_.

**Abstract** Previous work on learning physical systems from data has focused on high-resolution 
grid-structured measurements. However, real-world k
nowledge of such systems (e.g. weather data) relies on sparsely
scattered measuring stations. In this paper, we introduce a novel simulated benchmark dataset, 
DynaBench, for learning dynamical systems
directly from sparsely scattered data without prior knowledge of the
equations. The dataset focuses on predicting the evolution of a dynamical system from low-resolution, 
unstructured measurements. We simulate
six different partial differential equations covering a variety of physical
systems commonly used in the literature and evaluate several machine
learning models, including traditional graph neural networks and point
cloud processing models, with the task of predicting the evolution of
the system. The proposed benchmark dataset is expected to advance the
state of art as an out-of-the-box easy-to-use tool for evaluating models in
a setting where only unstructured low-resolution observations are available. 

*****
GrINd
*****

**Full title**: GrINd: GrINd: Grid Interpolation Network for Scattered Observations

The original GrINd paper can be found on `Arxiv <https://arxiv.org/abs/2403.19570>`_.
The original code is available on `GitHub <https://github.com/badulion/grind>`_.

**Abstract** Predicting the evolution of spatiotemporal physical systems from sparse and scattered observational 
data poses a significant challenge in various scientific domains. 
Traditional methods rely on dense grid-structured data, limiting their 
applicability in scenarios with sparse observations. To address this challenge, we introduce 
GrINd (Grid Interpolation Network for Scattered Observations), a novel network architecture 
that leverages the high-performance of grid-based models by mapping scattered observations 
onto a high-resolution grid using a Fourier Interpolation Layer. In the high-resolution space, 
a NeuralPDE-class model predicts the system's state at future timepoints using differentiable 
ODE solvers and fully convolutional neural networks parametrizing the system's dynamics. 
We empirically evaluate GrINd on the DynaBench benchmark dataset, comprising six different 
physical systems observed at scattered locations, demonstrating its state-of-the-art
performance compared to existing models. GrINd offers a promising approach for 
forecasting physical systems from sparse, scattered observational data, extending 
the applicability of deep learning methods to real-world scenarios with limited data availability.

******************
Your paper missing
******************

Have you used the DynaBench benchmark in your paper and cannot find it here?
Please let us know by creating a pull request or an issue on our `GitHub repository <https://github.com/badulion/dynabench>`_.