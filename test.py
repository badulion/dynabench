import numpy as np
from scipy.interpolate import RBFInterpolator
import matplotlib.pyplot as plt
import h5py
from src.dataset import DynaBenchBase
from tqdm import tqdm

# f = h5py.File("data/wave/0.hdf5")
# ds_low = f['data_grid_low'][:]
# ds_full = f['data_grid_full'][:]
# ds_cloud_low = f['data_cloud_low'][:]


def plot_comparison(ds_low, ds_high):
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(ds_low[100, 0])
    ax[1].imshow(ds_high[100,0])
    plt.show()


ds1 = DynaBenchBase(equation="wave", task="evolution",support="grid", num_points="high", lookback=8)

for _ in tqdm(ds1):
    _
