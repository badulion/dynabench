Dataset and Dataloader Selection
=================================

Use the following decision flowchart to choose the appropriate equation class and data iterator for your workflow.

.. code-block:: text

    1. Do you have precomputed simulation data (.h5 files)?
       |
       +-- Yes --> Choose between:
       |            |
       |            +-- `DynabenchIterator`
       |            |      Use for forecasting tasks (predict R steps given L steps).
       |            |      Data format: (data_input, data_target, points)
       |            |
       |            +-- `DynabenchSimulationIterator`
       |                   Use for full trajectory modeling or training on complete simulations.
       |                   Data format: (data, points)
       |
       +-- No  --> You need to generate data on-the-fly. Choose an Equation class first:
                    |
                    +-- Custom PDE: `BaseEquation`
                    +-- Burgers:    `SimpleBurgersEquation`
                    +-- Wave:       `WaveEquation`
                    +-- Cahn-Hilliard: `CahnHilliardEquation`
                    +-- Diffusion:  `DiffusionEquation`
                    +-- Fitzhugh-Nagumo: `FitzhughNagumoEquation`
                    +-- K-S:        `KuramotoSivashinskyEquation`
                    +-- Advection:  `AdvectionEquation`

                    Then choose the matching iterator:
                    |
                    +-- `EquationMovingWindowIterator`
                    |      Generates forecasting samples on-the-fly.
                    |
                    +-- `EquationSimulationIterator`
                           Generates full simulation trajectories on-the-fly.


Grid vs. Point Cloud Data
-------------------------

All iterators support both structured grid and scattered point cloud data.
- **Grid Data**: Structured 2D arrays. Best for CNNs, ResNets, FNOs.
- **Point Cloud**: Unstructured coordinates. Best for Point Transformers, Geo-FNOs.

Configure your grid via ``UnitGrid`` or custom ``Grid``, and set your initial conditions using classes like ``WrappedGaussians`` or ``RandomUniform``.
