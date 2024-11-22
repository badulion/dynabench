====================================
Example 2: Cahn-Hilliard Equation
====================================

In this example we will show how to generate data for the Cahn-Hilliard equation using the DynaBench library. 
The Cahn-Hilliard equation is an equation of mathematical physics which describes the process of phase separation, 
spinodal decomposition, by which the two components of a binary fluid spontaneously separate and form domains pure in each component. 
The equation is given by:

.. math::
    \frac{\partial c}{\partial t} = D\nabla^2\left(c^3-c-\gamma\nabla^2 c\right)

where :math:`c` is the concentration of the fluid with :math:`c=\mp 1` being the different domains, 
:math:`D` is the diffusion coefficient and :math:`\sqrt{\gamma}` represents the length of the transition regions between 
the two domains.

To generate data for the Cahn-Hilliard equation using the DynaBench library, 
we need to specify the parameters of the equation, the grid on which the data will be generated,
and the initial condition of the system. We will use the PyPDESolver to solve the Cahn-Hilliard equation.

We will generate data for the Cahn-Hilliard equation using the following steps:

1. :ref:`Define the Cahn-Hilliard equation. <define_equation>`
2. :ref:`Define the grid. <define_grid>`
3. :ref:`Define the initial condition. <define_initial>`
4. :ref:`Solve the Cahn-Hilliard equation. <solve>`
5. :ref:`Summary. <summary_cahnhiliard>`

.. _define_equation:

************************************
Define the Cahn-Hilliard equation
************************************
Let's start by defining the Cahn-Hilliard equation. We will use the :class:`dynabench.equation.CahnHilliardEquation` class to define the Cahn-Hilliard equation.

.. code-block::

    from dynabench.equation import CahnHilliardEquation

    # Create an instance of the CahnHilliardEquation class with default parameters
    pde_equation = CahnHilliardEquation()

In this case the default parameters of the equation are used. 
The parameters of the Cahn-Hilliard equation can be modified by passing the desired parameters to the constructor of the CahnHilliardEquation class.

The default parameters of the Cahn-Hilliard equation are as follows:

- :math:`D = 1.0`
- :math:`\gamma = 1.0`

.. _define_grid:

**************************************
Define the grid
**************************************

Next, we will define the grid on which the data for the Cahn-Hilliard equation will be generated.
We will use the :class:`dynabench.grid.Grid` class to define the grid.

.. code-block::

    from dynabench.grid import Grid

    # Create an instance of grid with default parameters
    grid = Grid(grid_limits=((0, 64), (0, 64)), grid_size=(64, 64))

In this case spatial domain is defined by the grid_limits parameter, 
which specifies the limits of the grid in each dimension.
The grid_size parameter specifies the number of grid points in each dimension.

In this case, as dx = dy = 1, the :class:`dynabench.grid.UnitGrid` class can be used to define the same grid:

.. code-block::

    from dynabench.grid import UnitGrid

    # Create an instance of a unit grid
    grid = UnitGrid(grid_size=(64, 64))

.. _define_initial:

**************************************
Define the initial condition
**************************************

Next, we will define the initial condition of the system. 
We will use the :class:`dynabench.initial.RandomUniform` class to generate 
an initial condition with random fluctuations drawn from a uniform distribution.

.. code-block::

    from dynabench.initial import RandomUniform

    # generate an initial condition as a sum of 5 gaussians
    intitial = RandomUniform()

.. _solve:

**************************************
Solve the Cahn-Hilliard equation
**************************************

Finally, we will solve the Cahn-Hilliard equation with the initial condition on the grid using the PyPDESolver.

.. code-block::

    from dynabench.solver import PyPDESolver

    # Solve the Cahn-Hilliard equation with the initial condition
    solver = PyPDESolver(equation=pde_equation, grid=grid, initial_generator=intitial, parameters={'method': "RK23"})
    solver.solve(t_span=[0, 100], dt_eval=1, random_state=42, out_dir="data/raw")

The parameters of the solver can be modified by passing the desired parameters as a dictionary to the constructor of the PyPDESolver class.
The t_span parameter specifies the time span over which the equation will be solved and the dt_eval parameter specifies the time step at which the solution will be evaluated.
Additionally, the random_state parameter can be used to set the random seed with which the initial condition is generated.

The solution of the Cahn-Hilliard equation is saved in h5 format specified by the optional `out_dir` parameter.

.. _summary_cahnhiliard:

**************************************
Summary
**************************************

Overall the code for generating data for the Cahn-Hilliard equation is as follows:

.. code-block::

    from dynabench.equation import CahnHilliardEquation
    from dynabench.initial import RandomUniform
    from dynabench.grid import Grid, UnitGrid
    from dynabench.solver import PyPDESolver

    # Create an instance of the CahnHilliardEquation class with default parameters
    pde_equation = CahnHilliardEquation()

    # Create an instance of a unit grid
    grid = UnitGrid(grid_size=(64, 64))

    # generate an initial condition as a sum of 5 gaussians
    intitial = RandomUniform()


    # Solve the Cahn-Hilliard equation with the initial condition
    solver = PyPDESolver(equation=pde_equation, grid=grid, initial_generator=intitial, parameters={'method': "RK23"})
    solver.solve(t_span=[0, 100], dt_eval=1)