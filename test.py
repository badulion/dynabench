import numpy as np
import matplotlib.pyplot as plt
import h5py
from sklearn.neighbors import KNeighborsRegressor
import matplotlib
import zipfile
from src.dataset.dataset_base import DynaBenchBase
from tqdm import tqdm
import torch

x = torch.rand(10, 10, 1)
print(x)