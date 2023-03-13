import io
import numpy as np
import tarfile

EQUATION = "gas_dynamics"

with tarfile.open(f"data/{EQUATION}/train/grid_full.tar") as f:
    files = f.getnames()
    file = f.extractfile(files[0])
    file_bytes = io.BytesIO(file.read())
    X = np.load(file_bytes, encoding="bytes")


import matplotlib.pyplot as plt
import matplotlib.animation as animation
fps = 30
# First set up the figure, the axis, and the plot element we want to animate

fig, ax = plt.subplots(1,2)

ax[0].imshow(X[50, 0], interpolation='none', aspect='auto', vmin=-2, vmax=3)
ax[1].imshow(X[51, 0], interpolation='none', aspect='auto', vmin=-2, vmax=3)
plt.savefig("output/figures/init_cond.png")



mse = np.mean((X[1:, :]-X[:-1, :])**2)
print(mse)