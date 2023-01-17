import numpy as np
from scipy.interpolate import RBFInterpolator
import matplotlib.pyplot as plt
import h5py

f = h5py.File("data/wave/0.hdf5")
ds_full = f['data_grid_full'][:]
ds_high = f['data_cloud_high'][:]
print(ds_high.shape)

fig, ax = plt.subplots(1,2)
ax[0].imshow(ds_full[4,1,:,:])
ax[1].imshow(ds_high[1,1,:,:])
plt.show()