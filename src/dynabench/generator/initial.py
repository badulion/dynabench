import numpy as np
from scipy.signal import convolve2d
import matplotlib.pyplot as plt

def sum_of_gaussians(grid_size=(64,64), components=1, zero_level = 0.0, random_state=None):
    np.random.seed(random_state)
    x, y = np.meshgrid(np.linspace(0,1,grid_size[0], endpoint=False), np.linspace(0,1,grid_size[1], endpoint=False))

    mx = [np.random.choice(grid_size[0]) for i in range(components)]
    my = [np.random.choice(grid_size[1]) for i in range(components)]

    squared_distance_to_center = (x-0.5)**2 + (y-0.5)**2
    gaussian = np.exp(-40*squared_distance_to_center)

    u = zero_level+np.zeros_like(x)
    for i in range(components):
        
        component = np.roll(gaussian, (mx[i],my[i]), axis=(0,1))

        u = u + np.random.uniform(-1, 1) * component
        
    #smoothing filter
    a = 5
    smoothing_filter = np.ones((2*a+1, 2*a+1))
    smoothing_filter/= np.sum(smoothing_filter)
    u = convolve2d(u, smoothing_filter, mode='same', boundary='wrap')

    return u

def random_uniform(grid_size=(64,64), smooth=False, random_state=None):
    np.random.seed(random_state)
    u = np.random.uniform(0, 1, size=grid_size)

    #smoothing filter
    if smooth:
        a = 1
        smoothing_filter = np.ones((2*a+1, 2*a+1))
        smoothing_filter/= np.sum(smoothing_filter)
        u = convolve2d(u, smoothing_filter, mode='same', boundary='wrap')

    return u

def random_normal(grid_size=(64,64), smooth=False, random_state=None):
    np.random.seed(random_state)
    u = np.random.normal(0, 1, size=grid_size)

    #smoothing filter
    if smooth:
        a = 1
        smoothing_filter = np.ones((2*a+1, 2*a+1))
        smoothing_filter/= np.sum(smoothing_filter)
        u = convolve2d(u, smoothing_filter, mode='same', boundary='wrap')

    return u