from torch.utils.data import Dataset
import os
import h5py
import numpy as np

from src.generator.solver import PDESolver
from src.generator.equations import equation_selector

class DynaBenchBase(Dataset):

    available_equations = ["gas_dynamics", "wave", "kuramoto_sivashinsky", "brusselator"]
    available_supports = ["low", "high", "full", "grid"]
    available_tasks = ["forecast", "evolution"]
    available_modes = ["train", "val", "test"]
    num_fields_dict = {
        "gas_dynamics": 4, 
        "wave": 1, 
        "kuramoto_sivashinsky": 1, 
        "brusselator": 1
    }

    def __init__(
        self,
        name="dyna-benchmark",
        mode="train",
        equation="gas_dynamics",
        task="forecast",
        support="high",
        base_path="data",
        lookback=1,
        rollout=1,
    ):
        super(DynaBenchBase, self).__init__()
        self.name = name
        self.equation = equation
        self.base_path = base_path
        self.support = support
        self.task = task
        self.mode = mode

        if not mode in self.available_modes:
            raise KeyError(f"Mode not available. Select from {self.available_mode}")



        self.lookback = lookback
        self.rollout = rollout

        # support selector
        if not support in self.available_supports:
            raise KeyError(f"Support not available. Select from {self.available_supports}")
        self.data_support, self.points_support = self._support_selector(self.support)

        # read files
        if not equation in self.available_equations:
            raise KeyError(f"Selected equation not available. Select from {self.available_equations}")
        self.path = os.path.join(self.base_path, self.equation)


        self.num_fields = self.num_fields_dict[self.equation]

        # do it nicer (perhaps not here?)
        if not os.path.exists(self.path):
            os.makedirs(self.path, exist_ok=True)
        if len(os.listdir(self.path)) == 0:
            EQ_MODULE = equation_selector(self.equation)
            eq = EQ_MODULE()
            solver = PDESolver(eq, self.path)
            solver.solve()
            solver.postprocess()


        file_paths_all = os.listdir(self.path)
        file_numbers = [file[5:-5] for file in file_paths_all]

        # filter by train/val/test
        num_files = len(file_numbers)
        self.file_numbers_train = range(8)
        self.file_numbers_val = [8]
        self.file_numbers_test = [9]
        if mode == "train":
            self.file_paths = [f"data_{i}.hdf5" for i in file_numbers if int(i) in self.file_numbers_train]
        elif mode == "val":
            self.file_paths = [f"data_{i}.hdf5" for i in file_numbers if int(i) in self.file_numbers_val]
        elif mode == "test":
            self.file_paths = [f"data_{i}.hdf5" for i in file_numbers if int(i) in self.file_numbers_test]

        self.files = [h5py.File(os.path.join(self.path, file_path)) for file_path in self.file_paths]
        self.raw_lengths = [len(file['data']) for file in self.files]
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
            x, y, points = self.get_item_forecast(file, file_idx)
        else:
            x, y, points = self.get_item_evolution(file, file_idx)

        # join lookback as channels
        new_shape = (-1, ) + x.shape[2:]
        x = x.reshape(new_shape)

        # permute axes
        x = np.transpose(x, axes=(1, 0))
        y = np.transpose(y, axes=(-1, -2))


        # additional transforms
        x, y, points = self.additional_transforms(x, y, points)

        return x, y, points

    
    def get_item_forecast(self, file, file_idx):
        data_x = np.split(file['data'][file_idx:file_idx+self.lookback], 2, axis=1)[0]
        data_y = np.split(file['data'][file_idx+self.lookback:file_idx+self.lookback+self.rollout], 2, axis=1)[0]
        points = file['points'][:]

        if self.equation == "wave":
            data_x = data_x[:, 0]
            data_y = data_y[:, 0]

        if self.support != "grid":
            points = points.reshape(-1, 2)
            data_x = data_x.reshape(self.lookback, self.num_fields, -1)
            data_y = data_y.reshape(self.rollout, self.num_fields, -1)
            indices = file[f"indices_{self.support}"]

            # select points
            points = points[indices]
            data_x = data_x[:,:, indices]
            data_y = data_y[:,:, indices]

        return data_x, data_y, points

    def get_item_evolution(self, file, file_idx):
        data_x = np.split(file['data'][file_idx:file_idx+self.lookback], 2, axis=1)[0]
        data_y = np.split(file['data'][file_idx], 2, axis=1)[1]
        points = file['points'][:]

        if self.equation == "wave":
            data_x = data_x[:, 0]
            data_y = data_y[:, 0]

        if self.support != "grid":
            points = points.reshape(-1, 2)
            data_x = data_x.reshape(self.lookback, self.num_fields, -1)
            data_y = data_y.reshape(self.num_fields, -1)
            indices = file[f"indices_{self.support}"]

            # select points
            points = points[indices]
            data_x = data_x[:,:, indices]
            data_y = data_y[:, indices]

        return data_x, data_y, points
    
    def __len__(self):
        return sum(self.real_lengths)

    def _support_selector(self, support):
        if support == "high":
            suffix = "_high"
        elif support == "low":
            suffix = "_low"
        else:
            suffix = ""
        return "data"+suffix, "points"+suffix

    def additional_transforms(self, x, y, points):
        return x, y, points