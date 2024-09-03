import numpy as np


class GridDownsamplerFFT:
    """
        Downsample a grid using FFT. First transforms an image to the frequency domain, then removes the high frequencies up to a 
        given target size, and finally transforms the image back to the spatial domain.
        
        Parameters
        ----------
        target_size : tuple
            The target size of the downsampled grid.
    """
    def __init__(self, target_size: tuple):
        self.target_size = target_size
        
    def __call__(self, data):
        frequency = np.fft.fftshift(np.fft.fft2(data, norm='forward'), axes=(-2,-1))
        shift_x = (data.shape[-2]-self.target_size[0])//2
        shift_y = (data.shape[-1]-self.target_size[1])//2
        frequency = frequency[..., shift_x:-shift_x, shift_y:-shift_y]
        return np.fft.ifft2(np.fft.ifftshift(frequency, axes=(-2,-1)), norm='forward').real