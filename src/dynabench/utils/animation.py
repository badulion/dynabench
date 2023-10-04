import imageio.v2 as iio
import os
import matplotlib.pyplot as plt
import numpy as np

from typing import List, Optional

from .colors import NEURALPDE_COLORMAP

def animate_simulation(u_array: np.ndarray, output_path: str = "equation.gif"):
    v_min = np.min(u_array[0])
    v_max = np.max(u_array[0])
    os.makedirs("tmp", exist_ok=True)
    for i in range(len(u_array)):
        plt.imshow(u_array[i], vmin=v_min, vmax=v_max, cmap=NEURALPDE_COLORMAP, interpolation="spline36")
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

def animate_multiple_simulations(u_array_list: List[np.ndarray], output_path: str = "equation.gif", u_labels: Optional[List[str]] = None, target_fps=20):
    num_steps = len(u_array_list[0])
    def plot_one_step(step_num: int):
        fig, ax = plt.subplots(1, len(u_array_list), squeeze=False)
        for i, u_array in enumerate(u_array_list):
            im = ax[0, i].imshow(u_array[step_num], vmin=-1, vmax=1, cmap=NEURALPDE_COLORMAP)
            ax[0, i].tick_params(left = False, right = False , labelleft = False,
                              labelbottom = False, bottom = False)
            if u_labels:
                ax[0, i].set_title(u_labels[i], fontsize=16)
        plt.tight_layout()
        fig.savefig(f"tmp/{step_num}.png", bbox_inches='tight')
        plt.close()

    os.makedirs("tmp", exist_ok=True)
    for i in range(num_steps):
        plot_one_step(i)

    with iio.get_writer(output_path, mode='I') as writer:
        file_list = [f"tmp/{i}.png" for i in range(num_steps)]
        for filename in file_list:
            image = iio.imread(filename)
            writer.append_data(image)

    gif = iio.mimread(output_path)
    iio.mimsave(f"{output_path}", gif, duration=(1000/target_fps))

    # cleanup
    tmp_files = os.listdir("tmp")
    for file in tmp_files:
        os.remove(os.path.join("tmp", file))
    os.removedirs("tmp")