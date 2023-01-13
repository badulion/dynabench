from pde import UnitGrid, FieldCollection, FileStorage
from pde import CallbackTracker, ProgressTracker
from pde.trackers.base import TrackerCollection

from pde.visualization.plotting import ScalarFieldPlot
from pde.visualization.movies import Movie

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
        graph_points_low=100,
        graph_points_mid=500,
        graph_points_high=900,
        ) -> None:

        # attributes
        self.equation = equation
        self.grid_size = grid_size
        self.t_range = t_range
        self.dt = dt
        self.save_interval = save_interval
        self.graph_points_low = graph_points_low
        self.graph_points_mid = graph_points_mid
        self.graph_points_high = graph_points_high

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

    def make_movie_frame_by_frame(self, path):
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
            ax[0].set_title("Wave")

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
        # select points
        indices_low = np.random.choice(self.grid_size*self.grid_size, size=self.graph_points_low, replace=False)
        indices_mid = np.random.choice(self.grid_size*self.grid_size, size=self.graph_points_mid, replace=False)
        indices_high = np.random.choice(self.grid_size*self.grid_size, size=self.graph_points_high, replace=False)
        indices_full = np.arange(self.grid_size*self.grid_size)

        points = self.grid.cell_coords/self.grid_size
        points_full = points.reshape(-1, 2)


        # extract graph points
        f = h5py.File(self.path, 'a')
        data_grid = f['data'][:]

        # standardise the data
        mean = np.mean(data_grid, axis=(-1, -2), keepdims=True)
        std = np.std(data_grid, axis=(-1, -2), keepdims=True)
        std[std < 1e-10] = 1
        data_grid = (data_grid - mean) / std

        # save graph sampled points
        f.create_dataset('indices_low', data=indices_low)
        f.create_dataset('indices_mid', data=indices_mid)
        f.create_dataset('indices_high', data=indices_high)
        f.create_dataset('indices_full', data=indices_full)
        f.create_dataset('points', data=points)


        f['data'][...] = data_grid

        f.close()



