import numpy as np
import matplotlib.pyplot as plt
import h5py
from sklearn.neighbors import KNeighborsRegressor
import matplotlib


f = h5py.File("data/gas_dynamics/data_0.hdf5")
print(f.keys())
print(f['indices_full'].shape)