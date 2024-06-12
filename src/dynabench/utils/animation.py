import imageio.v2 as iio
import os
import matplotlib.pyplot as plt
import numpy as np

from typing import List

from .colors import NEURALPDE_COLORMAP

def animate_simulation(u_array: np.ndarray, output_path: str = "equation.gif"):

    os.makedirs("tmp", exist_ok=True)
    for i in range(len(u_array)):
        plt.imshow(u_array[i], cmap=NEURALPDE_COLORMAP)
        plt.colorbar()
        plt.savefig(f"tmp/{i}.png")
        plt.close()

    with iio.get_writer('equation.gif', mode='I') as writer:
        file_list = [f"tmp/{i}.png" for i in range(len(u_array))]
        for filename in file_list:
            image = iio.imread(filename)
            writer.append_data(image)

    # cleanup
    tmp_files = os.listdir("tmp")
    for file in tmp_files:
        os.remove(os.path.join("tmp", file))
    os.removedirs("tmp")

def animate_multiple_simulations(u_array_list: List[np.ndarray], output_path: str = "equation.gif"):
    num_steps = len(u_array_list[0])
    def plot_one_step(step_num: int):
        fig, ax = plt.subplots(1, len(u_array_list))
        for i, u_array in enumerate(u_array_list):
            im = ax[i].imshow(u_array[step_num], vmin=-1, vmax=1, cmap=NEURALPDE_COLORMAP)
            #fig.colorbar(im)
        plt.tight_layout()
        fig.savefig(f"tmp/{step_num}.png", bbox_inches='tight')
        plt.close()

    os.makedirs("tmp", exist_ok=True)
    for i in range(num_steps):
        plot_one_step(i)

    with iio.get_writer('equation.gif', mode='I') as writer:
        file_list = [f"tmp/{i}.png" for i in range(num_steps)]
        for filename in file_list:
            image = iio.imread(filename)
            writer.append_data(image)

    # cleanup
    tmp_files = os.listdir("tmp")
    for file in tmp_files:
        os.remove(os.path.join("tmp", file))
    os.removedirs("tmp")