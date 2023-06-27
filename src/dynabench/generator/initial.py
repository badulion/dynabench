import numpy as np
from scipy.signal import convolve2d
import matplotlib.pyplot as plt

def sum_of_gaussians(x, y, components=1, zero_level = 0.0):

    mx = [np.random.choice(x.shape[1]) for i in range(components)]
    my = [np.random.choice(y.shape[0]) for i in range(components)]

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

def sum_of_sines(x, y, components=1, zero_level = 0.0):

    mx = [np.random.choice(x.shape[1]) for i in range(components)]
    my = [np.random.choice(y.shape[0]) for i in range(components)]

    distance_to_center = np.sqrt((x-0.5)**2 + (y-0.5)**2)
    threshold = 0.2

    wave = 0.5 + 0.5*np.cos((1/threshold)*np.pi*distance_to_center)
    wave[distance_to_center>threshold] = zero_level

    u = zero_level+np.zeros_like(x)
    for i in range(components):
        component = np.roll(wave, (mx[i],my[i]), axis=(0,1))

        u = u + np.random.uniform(-1, 1) * component
        
    #smoothing filter
    a = 50
    smoothing_filter = np.ones((2*a+1, 2*a+1))
    smoothing_filter/= np.sum(smoothing_filter)
    u = convolve2d(u, smoothing_filter, mode='same', boundary='wrap')

    return u

def sum_of_smoothsteps(x, y, components=1, zero_level=0.0):

    mx = [np.random.choice(x.shape[1]) for i in range(components)]
    my = [np.random.choice(y.shape[0]) for i in range(components)]

    distance_to_center = np.sqrt((x-0.5)**2 + (y-0.5)**2)
    threshold = 0.2

    bell_curve = smootherstep((1/threshold)*distance_to_center)

    u = zero_level+np.zeros_like(x)
    for i in range(components):
        component = np.roll(bell_curve, (mx[i],my[i]), axis=(0,1))

        u = u + np.random.uniform(-1, 1) * component
        
    #smoothing filter
    a = 50
    smoothing_filter = np.ones((2*a+1, 2*a+1))
    smoothing_filter/= np.sum(smoothing_filter)
    #u = convolve2d(u, smoothing_filter, mode='same', boundary='wrap')

    return u

def random_uniform(x, y, smooth=False):
    u = np.random.uniform(0, 1, size=x.shape)

    #smoothing filter
    if smooth:
        a = 1
        smoothing_filter = np.ones((2*a+1, 2*a+1))
        smoothing_filter/= np.sum(smoothing_filter)
        u = convolve2d(u, smoothing_filter, mode='same', boundary='wrap')

    return u

def random_normal(x, y, smooth=False):
    u = np.random.normal(0, 1, size=x.shape)

    #smoothing filter
    if smooth:
        a = 1
        smoothing_filter = np.ones((2*a+1, 2*a+1))
        smoothing_filter/= np.sum(smoothing_filter)
        u = convolve2d(u, smoothing_filter, mode='same', boundary='wrap')

    return u


def smootherstep(x):
    x = np.clip(x, -1, 1)

    x = np.abs(x)
    return -x * x * x * (x * (x * 6 - 15) + 10)