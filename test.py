import numpy as np
import matplotlib.pyplot as plt
import h5py


f = h5py.File('data.hdf5', 'r')
dset = f["data"]
plt.imshow(dset[500][1])
plt.show()