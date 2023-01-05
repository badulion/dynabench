import numpy as np
from tqdm import tqdm, trange
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('webagg')


def smootherstep(x):
    x = np.clip(x, -1, 1)

    x = np.abs(x)
    return -x * x * (3 - 2 * x)

x = np.linspace(-2, 2, 1000)
y = smootherstep(x)
plt.plot(x, y)
plt.show()