import numpy as np
import matplotlib.pyplot as plt
import h5py
from sklearn.neighbors import KNeighborsRegressor
from scipy.interpolate import RBFInterpolator
import matplotlib
matplotlib.use('webagg')


f = h5py.File('data/gas_dynamics/data_16.hdf5', 'a')

idx = 999
var = 1

# ground truth
f_gt = f['data']
f_points = f['points']

# interpolation
f_graph = f['data_low']
f_graph_points = f['points_low']

X = f_graph_points
y = f_graph[idx, var, :]

interpolator = KNeighborsRegressor(n_neighbors=3, weights="distance")
interpolator.fit(X, y) 
interpolator = RBFInterpolator(X, y)
 
X_eval = f_points[:].reshape(-1, 2)
#y_eval = interpolator.predict(X_eval).reshape(64, 64)
y_eval = interpolator(X_eval).reshape(64, 64)

fig, ax = plt.subplots(1,2)
ax[0].imshow(f_gt[idx, var])
ax[1].imshow(y_eval)
plt.show()