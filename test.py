import numpy as np
from scipy.interpolate import RBFInterpolator
import matplotlib.pyplot as plt
import h5py

ds = h5py.File("data/wave/0.hdf5")['data'][:]
points_ds = h5py.File("data/wave/0.hdf5")['points'][:]

def generate_grid(num_points=30):
    points = np.linspace(0.5/num_points, 1-0.5/num_points, num_points, endpoint=True)
    y, x = np.meshgrid(points, points)
    return np.stack([x, y], axis=-1)

def generate_pointcloud(num_points=900):
    points = np.random.uniform(0, 1, size=(num_points, 2))
    return points


points_low = generate_grid(15)
points_mid = generate_grid(23)
points_high = generate_grid(30)
points_full = generate_grid(64)

#np.transpose(ds, )
X_full = ds.transpose((2,3,0,1))

rbfint = RBFInterpolator(points_full.reshape((-1, 2)), X_full.reshape((64*64, 1001, 4)), neighbors=16)

X_low = rbfint(points_low.reshape((-1, 2))).reshape((15,15,1001,4))


fig, ax = plt.subplots(1,2)
ax[0].imshow(X_low[:,:,1000,1])
ax[1].imshow(X_full[:, :, 1000,1])
plt.show()