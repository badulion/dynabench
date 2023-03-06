from pde import UnitGrid, FieldCollection, FileStorage
from pde import CallbackTracker, ProgressTracker
from pde.trackers.base import TrackerCollection

from pde.visualization.plotting import ScalarFieldPlot
from pde.visualization.movies import Movie

from scipy.interpolate import RBFInterpolator
import os
import h5py
import numpy as np

import matplotlib.pyplot as plt
from tqdm import tqdm

from .equations import WavePDE, GasDynamicsPDE, BrusselatorPDE, KuramotoSivashinskyPDE
from ..utils import write_np_array_to_tar

class PDESolver:
    available_equations = ['wave', 'gas_dynamics', 'brusselator', 'kuramoto_sivashinsky']
    def __init__(self,
        equation,
        save_dir,
        save_name,
        grid_size= 64,
        t_range=100,
        dt=1e-3,
        save_interval=0.1,
        num_points=[15, 22, 30],
        num_points_names=["low", "med", "high"],
        save_full_grid=True,
        save_full_cloud=True,
        save_cloud_points=True,
        save_grid_points=True,
        ) -> None:


        # attributes
        self.equation = equation
        self.grid_size = grid_size
        self.t_range = t_range
        self.dt = dt
        self.save_interval = save_interval
        self.num_points = num_points
        self.num_points_names = num_points_names

        # save settings
        self.save_full_grid = save_full_grid
        self.save_full_cloud = save_full_cloud
        self.save_cloud_points = save_cloud_points
        self.save_grid_points = save_grid_points
        if not save_full_cloud*save_cloud_points*save_full_grid*save_grid_points:
            UserWarning("No data will be saved, are you sure?")

        self.grid = UnitGrid([grid_size, grid_size], periodic=[True, True])
        self.state = self.equation.get_initial_state(self.grid)
        self.state_with_dyn = FieldCollection([*self.state, *self.equation.evolution_rate(self.state)])
        
        # create storage
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
            
        sample_n = len(os.listdir(save_dir))
        self.path = os.path.join(save_dir, f"{sample_n}.hdf5")
        self.save_dir = save_dir
        self.save_name = save_name
        self.data = []
        self.times = []
        
        
    def solve(self):
        self.data = []
        self.times = []
        def save_state(state, time):
            state_with_dyn = FieldCollection([*state.copy(), *self.equation.evolution_rate(state)])
            self.data.append(state.copy().data)
            self.times.append(time)

        # setup
        tracker_callback = CallbackTracker(save_state, interval=self.save_interval)
        tracker_progress = ProgressTracker()
        tracker = TrackerCollection([tracker_callback, tracker_progress])

        # solve
        sol = self.equation.solve(self.state, t_range=self.t_range, tracker=tracker)
        self.data = np.stack(self.data)
        self.times = np.stack(self.times)

        # sample data and save indices
        self._postprocess()

    def make_movie(self, path):
        sfp = ScalarFieldPlot(self.state)
        sfp.make_movie(self.storage, os.path.join("figures", path))
        self.storage.close()

    def make_movie_frame_by_frame(self, filename):
        path = os.path.join('figures', filename)
        movie = Movie(path)
        for frame, t in zip(tqdm(self.storage.data), self.storage.times):
            fig, ax = plt.subplots(1,4, dpi=150, figsize=(12.8, 4.8))
            fig.set_layout_engine("tight")
            fig.suptitle(f"Time {int(t)}")
            fig.set_figwidth(0.75*fig.get_figwidth())
            fig.set_figheight(0.75*fig.get_figheight())

            # first subplot
            hm = ax[0].imshow(frame[0,:,:], interpolation="bilinear", vmin=-2.5, vmax=2.5)
            fig.colorbar(hm, ax=ax[0], location="bottom", shrink=1, pad=0.05)
            ax[0].set_xticks([])
            ax[0].set_yticks([])
            ax[0].set_title("Density")

            # second subplot
            hm = ax[1].imshow(frame[1,:,:], interpolation="bilinear", vmin=-2.5, vmax=2.5)
            fig.colorbar(hm, ax=ax[1], location="bottom", shrink=1, pad=0.05)
            ax[1].set_xticks([])
            ax[1].set_yticks([])
            ax[1].set_title("Temperature")

            # first subplot
            hm = ax[2].imshow(frame[2,:,:], interpolation="bilinear", vmin=-2.5, vmax=2.5)
            fig.colorbar(hm, ax=ax[2], location="bottom", shrink=1, pad=0.05)
            ax[2].set_xticks([])
            ax[2].set_yticks([])
            ax[2].set_title("Velocity x")

            # second subplot
            hm = ax[3].imshow(frame[3,:,:], interpolation="bilinear", vmin=-2.5, vmax=2.5)
            fig.colorbar(hm, ax=ax[3], location="bottom", shrink=1, pad=0.05)
            ax[3].set_xticks([])
            ax[3].set_yticks([])
            ax[3].set_title("Velocity y")

            movie.add_figure(fig)
            plt.close(fig)
        movie.save()

    def _postprocess(self):
        def generate_grid(num_points=30):
            points = np.linspace(0.5/num_points, 1-0.5/num_points, num_points, endpoint=True)
            y, x = np.meshgrid(points, points)
            return np.stack([x, y], axis=-1)

        def generate_pointcloud(num_points=900):
            points = np.random.uniform(0, 1, size=(num_points, 2))
            return points

        def interpolate_cloud(interpolator, points):
            interpolated_values = interpolator(points)
            interpolated_values = interpolated_values.transpose((1, 2, 0))
            return interpolated_values

        def interpolate_grid(interpolator, points):
            num_grid_points = points.shape[0]
            points_reshaped = points.reshape((-1, 2))
            interpolated_values = interpolator(points_reshaped)
            interpolated_values = interpolated_values.reshape((num_grid_points, num_grid_points)+interpolated_values.shape[1:])
            interpolated_values = interpolated_values.transpose((2,3,0,1))
            return interpolated_values
                        
                        
        # select points for grid
        points_grid = [generate_grid(sup) for sup in self.num_points]
        points_grid_full = self.grid.cell_coords/self.grid_size

        # select points for cloud
        points_cloud = [generate_pointcloud(sup**2) for sup in self.num_points]
        points_cloud_full = points_grid_full.reshape(-1, 2)

        # write times
        write_np_array_to_tar(self.times, f"{self.save_name}.times", os.path.join(self.save_dir, f"times.tar"))
        num_samples = len(self.times)


        # standardise the data
        mean = np.mean(self.data, axis=(-1, -2), keepdims=True)
        std = np.std(self.data, axis=(-1, -2), keepdims=True)
        std[std < 1e-10] = 1
        data_grid_full = (self.data - mean) / std

        # interpolate the data at selected points
        interpolation_values = data_grid_full.transpose((2,3,0,1)).reshape((self.grid_size**2,)+data_grid_full.shape[:2])
        interpolator = RBFInterpolator(points_cloud_full, interpolation_values, neighbors=16)

        # grid part of the dataset
        data_grid = [interpolate_grid(interpolator, points) for points in points_grid]
        # save data
        if self.save_full_grid:
            write_np_array_to_tar(data_grid_full, f"{self.save_name}.data", os.path.join(self.save_dir, f"grid_full.tar"))
            write_np_array_to_tar(points_grid_full, f"{self.save_name}.points", os.path.join(self.save_dir, f"grid_full.tar"))
            
        if self.save_grid_points:
            # save sampled subgrids:
            for points, name in zip(points_grid, self.num_points_names):
                write_np_array_to_tar(points, f"{self.save_name}.points", os.path.join(self.save_dir, f"grid_{name}.tar"))
            for data, name in zip(data_grid, self.num_points_names):
                write_np_array_to_tar(data, f"{self.save_name}.data", os.path.join(self.save_dir, f"grid_{name}.tar"))

        # point cloud part of the dataset
        data_cloud_full = data_grid_full.reshape(data_grid_full.shape[:2]+(-1,))
        data_cloud = [interpolate_cloud(interpolator, points) for points in points_cloud]
    
        if self.save_full_cloud:
            write_np_array_to_tar(data_cloud_full, f"{self.save_name}.data", os.path.join(self.save_dir, f"cloud_full.tar"))
            write_np_array_to_tar(points_cloud_full, f"{self.save_name}.points", os.path.join(self.save_dir, f"cloud_full.tar"))

            
        if self.save_cloud_points:
            # save sampled points
            for points, name in zip(points_cloud, self.num_points_names):
                write_np_array_to_tar(points, f"{self.save_name}.points", os.path.join(self.save_dir, f"cloud_{name}.tar"))
            for data, name in zip(data_cloud, self.num_points_names):
                write_np_array_to_tar(data, f"{self.save_name}.data", os.path.join(self.save_dir, f"cloud_{name}.tar"))



