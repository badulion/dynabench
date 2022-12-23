import numpy as np
from scipy.signal import convolve2d
import matplotlib.pyplot as plt

def sum_of_gaussians(x, y, components=1, zero_level = 0.0):

    mx = [np.random.uniform(0, 1) for i in range(components)]
    my = [np.random.uniform(0, 1) for i in range(components)]

    u = zero_level+np.zeros_like(x)
    for i in range(components):
        component = np.exp(-(30*(x-mx[i])**2 + 30*(y-my[i])**2))

        u = u + np.random.uniform(-1, 1) * component
        
    #smoothing filter
    a = 5
    smoothing_filter = np.ones((2*a+1, 2*a+1))
    smoothing_filter/= np.sum(smoothing_filter)
    u = convolve2d(u, smoothing_filter, mode='same', boundary='symm')

    return u


def sum_of_gaussians_periodic(x, y, components=1, zero_level = 0.0):

    mx = [np.random.uniform(0, 1) for i in range(components)]
    my = [np.random.uniform(0, 1) for i in range(components)]

    gaussian = np.exp(-(20*(x-0.5)**2 + 20*(y-0.5)**2))

    u = zero_level+np.zeros_like(x)
    for i in range(components):
        shift_x = np.argmin(np.abs(x[:,0]-mx[i]))-np.argmin(np.abs(x[:,0]-np.mean(x)))
        shift_y = np.argmin(np.abs(y[0,:]-my[i]))-np.argmin(np.abs(y[0,:]-np.mean(y)))
        
        component = np.roll(gaussian, (shift_y,shift_x), axis=(0,1))

        u = u + np.random.uniform(-1, 1) * component
        
    #smoothing filter
    a = 5
    smoothing_filter = np.ones((2*a+1, 2*a+1))
    smoothing_filter/= np.sum(smoothing_filter)
    u = convolve2d(u, smoothing_filter, mode='same', boundary='wrap')

    return u