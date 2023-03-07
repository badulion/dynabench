import io
import numpy as np
import tarfile


EQUATION = "advection"

with tarfile.open(f"data/{EQUATION}/train/grid_full.tar") as f:
    files = f.getnames()
    file = f.extractfile(files[0])
    file_bytes = io.BytesIO(file.read())
    X = np.load(file_bytes, encoding="bytes")


import matplotlib.pyplot as plt
import matplotlib.animation as animation
fps = 30
# First set up the figure, the axis, and the plot element we want to animate
fig = plt.figure( figsize=(8,8) )

a = X[0, 0]
im = plt.imshow(a, interpolation='none', aspect='auto', vmin=-2, vmax=3)
plt.colorbar()

def animate_func(i):
    if i % fps == 0:
        print( '.', end ='' )

    im.set_array(X[i, 0])
    return [im]

anim = animation.FuncAnimation(
                               fig, 
                               animate_func, 
                               frames = 100
                               )

anim.save(f'output/figures/{EQUATION}.gif', fps=fps)

print('Done!')

#plt.show()  # Not required, it seems!