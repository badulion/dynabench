from torch.utils.data import Dataset
import os
import numpy as np
from math import ceil
from typing import Literal

class DynaBenchBase(Dataset):
    def __init__(
        self,
        mode: str="train",
        equation: str="gas_dynamics",
        task: str="forecast",
        support: str="grid",
        num_points: str="high",
        base_path: str="data",
        lookback: int=1,
        rollout: int=1,
        test_ratio: float=0.1,
        val_ratio:  float=0.1,
        merge_lookback: bool=True,
        *args,
        **kwargs,
    ):
        """Initializes a pytorch dataset with selected parameters. The data is loaded lazily.

        Args:
            mode (str, optional): the selection of data to use (train/val/test). Defaults to "train".
            equation (str, optional): the equation to use. Defaults to "gas_dynamics".
            task (str, optional): Which task to use as targets. Defaults to "forecast".
            support (str, optional): Structure of the points at which the measurements are recorded. Defaults to "grid".
            num_points (str, optional): Number of points at which measurements are available. Defaults to "high".
            base_path (str, optional): location where the data is stored. Defaults to "data".
            lookback (int, optional): How many past states are used to make the prediction. The additional states can be concatenated along the channel dimension if merge_lookback is set to True. Defaults to 1.
            rollout (int, optional): How many steps should be predicted in a closed loop setting. Only used for forecast task. Defaults to 1.
            test_ratio (float, optional): What fraction of simulations to set aside for testing. Defaults to 0.1.
            val_ratio (float, optional): What fraction of simulations to set aside for validation. Defaults to 0.1.
            merge_lookback (bool, optional): Whether to merge the additional lookback information into the channel dimension. Defaults to True.

        Raises:
            RuntimeError: Data not found. Generate the data first
            RuntimeError: Data not found. Generate the data first
            RuntimeError: At least 3 simulations need to be run in order to use the dataset (1 each for train/val/test)
        """
        

        super(DynaBenchBase, self).__init__()
        self.equation = equation
        self.base_path = base_path
        self.support = support
        self.num_points = num_points
        self.task = task
        self.mode = mode
        self.merge_lookback = merge_lookback


        self.lookback = lookback
        self.rollout = rollout

        # support selector
        self.data_selector = f"data_{support}_{num_points}"
        self.points_selector = f"points_{support}_{num_points}"

        # read files
        self.path = os.path.join(self.base_path, self.equation)

        # do it nicer (perhaps not here?)
        if not os.path.exists(self.path):
            raise RuntimeError(f"Data not found. Did you generate the data?")
        if len(os.listdir(self.path)) == 0:
            raise RuntimeError(f"Data not found. Did you generate the data?")


        file_paths_all = os.listdir(self.path)
        file_numbers = [file[:-5] for file in file_paths_all]

        # filter by train/val/test
        num_files = len(file_numbers)
        if num_files < 3 and mode == "train":
            raise RuntimeError(f"Not enough data to split")
        else:
            num_files_test = ceil(num_files*test_ratio)
            num_files_val = ceil(num_files*val_ratio)

        self.file_numbers_train = range(num_files-num_files_test-num_files_val)
        self.file_numbers_val = range(num_files-num_files_test-num_files_val,num_files-num_files_test)
        self.file_numbers_test = range(num_files-num_files_test,num_files)

        if mode == "train":
            self.file_paths = [f"{i}.hdf5" for i in file_numbers if int(i) in self.file_numbers_train]
        elif mode == "val":
            self.file_paths = [f"{i}.hdf5" for i in file_numbers if int(i) in self.file_numbers_val]
        elif mode == "test":
            self.file_paths = [f"{i}.hdf5" for i in file_numbers if int(i) in self.file_numbers_test]

        self.files = [h5py.File(os.path.join(self.path, file_path)) for file_path in self.file_paths]
        self.raw_lengths = [len(file['times']) for file in self.files]
        self.real_lengths = [length - self.lookback - self.rollout+1 for length in self.raw_lengths]
        self.indices_end = np.cumsum(self.real_lengths)
        self.indices_start = self.indices_end - self.real_lengths[0]


    def __getitem__(self, index):
        if index < 0:
            index += len(self)
        if index > len(self) or index < 0:
            raise IndexError("Index out of bounds")
        
        # select appropriate file and indices
        file_selector = next(idx for idx, x in enumerate(self.indices_end) if index < x)
        file_idx = index - self.indices_start[file_selector]
        file = self.files[file_selector]

        
        # select data
        if self.task == "forecast":
            x, y = self._get_item_forecast(file, file_idx)
        else:
            x, y = self._get_item_evolution(file, file_idx)

        # permute axis for cloud data
        x, y = self._permute_axis(x, y)

        # get points
        points = file[self.points_selector][:]

        # merge lookback if necessary
        if self.merge_lookback:
            x = x.transpose((1, 0, 2))
            x = x.reshape((x.shape[0], -1))
            

        # additional transforms
        x, y, points = self.additional_transforms(x, y, points)
        """
        # join lookback as channels
        """
        return x, y, points

    
    def _get_item_forecast(self, file, file_idx):
        data_x = np.split(file[self.data_selector][file_idx:file_idx+self.lookback], 2, axis=1)[0]
        data_y = np.split(file[self.data_selector][file_idx+self.lookback:file_idx+self.lookback+self.rollout], 2, axis=1)[0]

        if self.equation == "wave":
            data_x = data_x[:, [0]]
            data_y = data_y[:, [0]]

        return data_x, data_y

    def _get_item_evolution(self, file, file_idx):
        data_x = np.split(file[self.data_selector][file_idx:file_idx+self.lookback], 2, axis=1)[0]
        data_y = np.split(file[self.data_selector][file_idx], 2, axis=0)[1]

        if self.equation == "wave":
            data_x = data_x[:, [0]]
            data_y = data_y[[0]]

        return data_x, data_y

    def _permute_axis(self, x, y):
        # before permutation dimensions are lookback x num_channels x num_points
        # should be lookback x num_points x num_channels for cloud
        if self.support == "cloud":
            x = x.transpose((0, 2, 1))
            y = y.transpose((0, 2, 1)) if self.task == "forecast" else y.transpose((1,0))
        return x, y

    
    def __len__(self):
        return sum(self.real_lengths)

    def additional_transforms(self, x, y, points):
        return x, y, points