===================
Getting Started
===================

This section guides you on how to start using the DynaBench package.

--------------------
Installing DynaBench
--------------------

^^^^^^^^^^^^^^^^^^^^
Install using pip
^^^^^^^^^^^^^^^^^^^^

The easiest way to install DynaBench is to use pip:

.. code-block::

    pip install dynabench

Also when using pip, it’s good practice to use a virtual environment - see `this guide <https://dev.to/bowmanjd/python-tools-for-managing-virtual-environments-3bko#howto>`_ for details on using virtual environments.

^^^^^^^^^^^^^^^^^^^^
Install from source
^^^^^^^^^^^^^^^^^^^^

To get the latest version of the code, you can install from source. 
This can be done by providing the git repository with the pip command:

.. code-block::

    pip install git+https://github.com/badulion/dynabench.git

Alternatively, if you want to help develop the package you can clone the repository and install it manually:

.. code-block::

    git clone https://github.com/badulion/dynabench.git
    cd dynabench
    pip install .

--------------------
Downloading the data
--------------------

The DynaBench data is available for download from the `WueData repository <https://wuedata.uni-wuerzburg.de/radar/de/dataset/sSEeRraAYDgQCgBP>`_. 
The data can be easily downloaded using the :py:func:`dynabench.dataset.download_equation` function as shown below:

.. code-block::

    from dynabench.dataset import download_equation

    download_equation('advection', structure='cloud', resolution='low')

This will download download the advection dataset with observation points scattered (cloud) and low resolution that will be saved in the `data/` directory.

--------------------
Using the data
--------------------

To easily load the data the dynabench package provides the :py:class:`dynabench.dataset.DynabenchIterator` iterator:

.. code-block::

    from dynabench.dataset import DynabenchIterator

    advection_iterator = DynabenchIterator(equation='advection', 
                                           structure='cloud', 
                                           resolution='low',
                                           lookback=4,
                                           rollout=16)

This will iterate through all downloaded simulation of the advection dataset with observation points scattered (cloud) and low resolution. 
Each sample will be a tuple containing a snapshot of the simulation at the past 4 time steps, the future 16 time steps as target as well as the coordinates of the observation points:


.. code-block::

    for sample in advection_iterator:
        x, y, points = sample

        # x is the input data with shape (lookback, n_points, n_features)
        # y is the target data with shape (rollout, n_points, n_features)
        # points are the observation points with shape (n_points, dim)
        # for the advection equation n_features=1 and dim=2