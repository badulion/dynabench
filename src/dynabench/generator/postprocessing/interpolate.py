from scipy.interpolate import RBFInterpolator
import numpy as np

from typing import Union, Tuple, Optional, List

def create_interpolator(u_grid, p_grid):
    # create padding
    p_padded_x = np.concatenate([p_grid-[1,0], p_grid, p_grid+[1,0]], axis=0)
    p_padded = np.concatenate([p_padded_x-[0,1], p_padded_x, p_padded_x+[0,1]], axis=1)

    u_padded_x = np.concatenate([u_grid]*3, axis=-2)
    u_padded = np.concatenate([u_padded_x]*3, axis=-1)
    

    p_reshaped = p_padded.reshape(-1,2)
    u_reshaped = u_padded.transpose(2,3,0,1).reshape((-1,)+u_padded.shape[:2])

    interpolator = RBFInterpolator(p_reshaped, u_reshaped, neighbors=16)
    return interpolator

def interpolate_grid(interpolator, grid_size: Union[Tuple[int], int] = 30):
    if type(grid_size) == int:
        grid_size = (grid_size, grid_size)
    
    x,y = np.meshgrid(
        np.linspace(0,1,grid_size[0], endpoint=False), 
        np.linspace(0,1,grid_size[1], endpoint=False)
    )

    p_grid = np.stack([x,y], axis=-1)
    p_grid_reshaped = p_grid.reshape(-1,2)

    u_interpol_reshaped = interpolator(p_grid_reshaped)
    u_interpol = u_interpol_reshaped.reshape(grid_size+interpolator.d_shape).transpose(2,3,0,1)
    return u_interpol, p_grid

def interpolate_cloud(interpolator, num_points: int = 900, random_state: Optional[int] = None):
    np.random.seed(random_state)
    p_cloud = np.random.random((num_points, 2))
    
    u_cloud_reshaped = interpolator(p_cloud)
    u_cloud = u_cloud_reshaped.transpose(1,0,2)
    return u_cloud, p_cloud


def generate_data_at_different_resolutions(u_grid, p_grid, resolutions_cloud: List[int] = [225, 484, 900], resolutions_grid: List[int] = [15, 22, 30], random_state: Optional[int] = None):
    interpolator = create_interpolator(u_grid, p_grid)

    interpolations_cloud = {}
    for res in resolutions_cloud:
        u, p = interpolate_cloud(interpolator, num_points=res, random_state=random_state)
        interpolations_cloud[res] = (u, p)

    interpolations_grid = {}
    for res in resolutions_grid:
        u, p = interpolate_grid(interpolator, grid_size=res)
        interpolations_grid[res] = (u, p)

    return interpolations_grid, interpolations_cloud

