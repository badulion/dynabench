from torch.utils.data import Dataset
import os
import h5py
import numpy as np

from src.dataset.dataset_base import DynaBenchBase

class DynaBench(DynaBenchBase):
    def __init__(self, name="dyna-benchmark", mode="train", equation="gas_dynamics", support="high", task="forecast", base_path="data", lookback=0, rollout=1):
        super().__init__(name=name, 
                         mode=mode,
                         equation=equation, 
                         support=support,
                         task=task, 
                         base_path=base_path,
                         lookback=lookback, 
                         rollout=rollout)