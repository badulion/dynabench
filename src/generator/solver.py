from pde import UnitGrid, FieldCollection, FileStorage
from pde import CallbackTracker, ProgressTracker
from pde.trackers.base import TrackerCollection

from pde.visualization.plotting import ScalarFieldPlot
from pde.visualization.movies import Movie

from scipy.interpolate import RBFInterpolator

import gin
import os
import h5py
import numpy as np

import matplotlib.pyplot as plt
from tqdm import tqdm

@gin.configurable
class PDESolver:
    def __init__(self,
        equation,
        save_dir,
        grid_size= 64,
        t_range=100,
        dt=1e-3,
        save_interval=0.1,
        grid_dim_low=15,
        grid_dim_mid=23,
        grid_dim_high=30,
        ) -> None:

        # attributes
        self.equation = equation
        self.grid_size = grid_size
        self.t_range = t_range
        self.dt = dt
        self.save_interval = save_interval
        self.grid_dim_low = grid_dim_low
        self.grid_dim_mid = grid_dim_mid
        self.grid_dim_high = grid_dim_high

        self.grid = UnitGrid([grid_size, grid_size], periodic=[True, True])
        self.state = equation.get_initial_state(self.grid)
        self.state_with_dyn = FieldCollection([*self.state, *equation.evolution_rate(self.state)])
        
        # create storage
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
            
        sample_n = len(os.listdir(save_dir))
        self.path = os.path.join(save_dir, f"{sample_n}.hdf5")
        self.storage = FileStorage(self.path)
        
        
    def solve(self):
        def save_state(state, time):
            state_with_dyn = FieldCollection([*state.copy(), *self.equation.evolution_rate(state)])
            self.storage.append(state_with_dyn)

        # setup
        tracker_callback = CallbackTracker(save_state, interval=self.save_interval)
        tracker_progress = ProgressTracker()
        tracker = TrackerCollection([tracker_callback, tracker_progress])

        # solve
        self.storage.start_writing(self.state_with_dyn)
        sol = self.equation.solve(self.state, t_range=self.t_range, tracker=tracker)
        self.storage.close()

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
            fig, ax = plt.subplots(1,2, dpi=150)
            fig.set_layout_engine("tight")
            fig.suptitle(f"Time {int(t)}")
            fig.set_figwidth(0.75*fig.get_figwidth())
            fig.set_figheight(0.75*fig.get_figheight())


            # first subplot
            hm = ax[0].imshow(frame[0,:,:], interpolation="bilinear", vmin=-2.5, vmax=2.5)
            fig.colorbar(hm, ax=ax[0], location="bottom", shrink=1, pad=0.05)
            ax[0].set_xticks([])
            ax[0].set_yticks([])
            ax[0].set_title("Brusselator")

            # second subplot
            hm = ax[1].imshow(frame[1,:,:], interpolation="bilinear", vmin=-2.5, vmax=2.5)
            fig.colorbar(hm, ax=ax[1], location="bottom", shrink=1, pad=0.05)
            ax[1].set_xticks([])
            ax[1].set_yticks([])
            ax[1].set_title("First Derivative")

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
            num_points = points.shape[0]
            return interpolator(points)

        def interpolate_grid(interpolator, points):
            num_grid_points = points.shape[0]
            points_reshaped = points.reshape((-1, 2))
            interpolated_values = interpolator(points_reshaped)
            return interpolated_values.reshape((num_grid_points, num_grid_points)+interpolated_values.shape[1:])

        # select points for grid
        points_grid_low = generate_grid(self.grid_dim_low)
        points_grid_mid = generate_grid(self.grid_dim_mid)
        points_grid_high = generate_grid(self.grid_dim_high)
        points_grid_full = self.grid.cell_coords/self.grid_size

        # select points for cloud
        points_cloud_low = generate_pointcloud(self.grid_dim_low**2)
        points_cloud_mid = generate_pointcloud(self.grid_dim_mid**2)
        points_cloud_high = generate_pointcloud(self.grid_dim_high**2)
        points_cloud_full = points_grid_full.reshape(-1, 2)


        # extract graph points
        f = h5py.File(self.path, 'a')
        data_grid = f['data'][:]
        num_samples = data_grid.shape[2:]

        # standardise the data
        mean = np.mean(data_grid, axis=(-1, -2), keepdims=True)
        std = np.std(data_grid, axis=(-1, -2), keepdims=True)
        std[std < 1e-10] = 1
        data_grid_full = (data_grid - mean) / std

        # interpolate the data at selected points
        interpolation_values = data_grid_full.transpose((2,3,0,1)).reshape((self.grid_size**2, -1))
        interpolator = RBFInterpolator(points_cloud_full, interpolation_values, neighbors=16)

        data_grid_low = interpolate_grid(interpolator, points_grid_low)
        data_grid_mid = interpolate_grid(interpolator, points_grid_mid)
        data_grid_high = interpolate_grid(interpolator, points_grid_high)

        data_cloud_full = data_grid_full.reshape((-1,)+num_samples)
        data_cloud_low = interpolate_cloud(interpolator, points_cloud_low)
        data_cloud_mid = interpolate_cloud(interpolator, points_cloud_mid)
        data_cloud_high = interpolate_cloud(interpolator, points_cloud_high)
        


        # save graph sampled points
        f.create_dataset('points_grid_low', data=points_grid_low)
        f.create_dataset('points_grid_mid', data=points_grid_mid)
        f.create_dataset('points_grid_high', data=points_grid_high)
        f.create_dataset('points_grid_full', data=points_grid_full)

        f.create_dataset('data_grid_low', data=data_grid_low)
        f.create_dataset('data_grid_mid', data=data_grid_mid)
        f.create_dataset('data_grid_high', data=data_grid_high)
        f.create_dataset('data_grid_full', data=data_grid_full)

        f.create_dataset('points_cloud_low', data=points_cloud_low)
        f.create_dataset('points_cloud_mid', data=points_cloud_mid)
        f.create_dataset('points_cloud_high', data=points_cloud_high)
        f.create_dataset('points_cloud_full', data=points_cloud_full)

        f.create_dataset('data_cloud_low', data=data_cloud_low)
        f.create_dataset('data_cloud_mid', data=data_cloud_mid)
        f.create_dataset('data_cloud_high', data=data_cloud_high)
        f.create_dataset('data_cloud_full', data=data_cloud_full)

        del f["data"]

        f.close()



