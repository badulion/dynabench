from .interpolate import generate_data_at_different_resolutions
from dynabench.utils.archive import write_np_array_to_tar
from dataclasses import dataclass
import numpy as np
import os

from typing import Union



@dataclass
class Postprocessor:
    cloud_resolutions: dict[str, int]
    grid_resolutions: dict[str, int]
    save_location: str = "data/"
    save_prefix: str = ""
    save_postfix: str = ""
    normalize: bool = True

    def process(self, u, p, simulation_name: Union[str, int]=1, random_state: int = None):

        # create output dir if not existent
        if not os.path.exists(self.save_location):
            os.makedirs(self.save_location)
        
        # standardise the data
        if self.normalize:
            mean = np.mean(u, axis=(-1, -2), keepdims=True)
            std = np.std(u, axis=(-1, -2), keepdims=True)
            std[std < 1e-10] = 1
            u = (u - mean) / std

        # interpolate
        interpolations_grid, interpolations_cloud = generate_data_at_different_resolutions(u, p, self.cloud_resolutions.values(), self.grid_resolutions.values(), random_state=random_state)

        # check if path exists
        # save data cloud interpolations
        for key, value in self.cloud_resolutions.items():
            u_int, p_int = interpolations_cloud[value]
            tar_archive_name = f"{self.save_prefix}_cloud_{key}_{self.save_postfix}.tar"
            tar_location = os.path.join(self.save_location, tar_archive_name)
            write_np_array_to_tar(u_int, f"{simulation_name}.data", tar_location)
            write_np_array_to_tar(p_int, f"{simulation_name}.points", tar_location)

        # save data grid interpolations
        for key, value in self.grid_resolutions.items():
            u_int, p_int = interpolations_grid[value]
            tar_archive_name = f"{self.save_prefix}_grid_{key}_{self.save_postfix}.tar"
            tar_location = os.path.join(self.save_location, tar_archive_name)
            write_np_array_to_tar(u_int, f"{simulation_name}.data", tar_location)
            write_np_array_to_tar(p_int, f"{simulation_name}.points", tar_location)

        # save data full simulation
        tar_archive_name = f"{self.save_prefix}_grid_full_{self.save_postfix}.tar"
        tar_location = os.path.join(self.save_location, tar_archive_name)
        write_np_array_to_tar(u, f"{simulation_name}.data", tar_location)
        write_np_array_to_tar(p, f"{simulation_name}.points", tar_location)

        return 0