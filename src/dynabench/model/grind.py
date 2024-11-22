import torch.nn as nn
import torch    
    
class FourierInterpolator(nn.Module):
    """
        Fourier Interpolation Layer. Interpolates a function using Fourier coefficients. Given a set of points and values of a function,
        it computes the Fourier coefficients and then evaluates the function at a different set of points.
        
        Parameters
        ----------
        num_ks : int, default 5
            The number of Fourier modes to use for the interpolation.
        spatial_dim : int, default 2
            The spatial dimension of the PDE.
    """
    def __init__(self, num_ks = 5, spatial_dim: int = 2, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.num_ks = num_ks
        self.spatial_dim = spatial_dim
        
    def forward(self, points_source, values_source, points_target):
        """ approximates the function at the given points using the fourier coefficients """
        fourier_coefficients = self.solve_for_fourier_coefficients(points_source, values_source)
        basis = self.generate_fourier_basis(points_target)
        reconstruction = (basis @ fourier_coefficients).real
        return reconstruction
    
    def generate_fourier_basis(self, points):
        points = 2*torch.pi*(points-0.5)
        ks = self.generate_fourier_ks(points)
        return torch.exp(1j * (points @ ks.T))
    
    def generate_fourier_ks(self, points):
        if self.num_ks % 2 == 0:
            ks = torch.arange(-self.num_ks//2, self.num_ks//2, dtype=points.dtype, device=points.device)
        else:
            ks = torch.arange(-(self.num_ks-1)//2, (self.num_ks-1)//2+1, dtype=points.dtype, device=points.device)
            
        ks_ = torch.meshgrid(*[ks]*self.spatial_dim)
        ks = torch.stack([k.flatten() for k in ks_], axis=1)
        return ks
    
    def solve_for_fourier_coefficients(self, points, values):
        basis = self.generate_fourier_basis(points)
        coeffs = torch.linalg.lstsq(basis, values+0j)[0]
        return coeffs
    
class GrIND(nn.Module):
    """
        GrIND model for predicting the evolution of PDEs by first interpolating onto a high-resolution grid, 
        solving the PDE and interpolating back to the original space.
        
        Parameters
        ----------
        prediction_net : nn.Module
            The neural network that predicts the evolution of the PDE in the high resolution space.
        num_ks : int, default 21
            The number of Fourier modes to use for the interpolation.
        grid_resolution : int, default 64
            The resolution of the high-grid to interpolate onto.
        spatial_dim : int, default 2
            The spatial dimension of the PDE.
    """
    def __init__(self, 
                 prediction_net: nn.Module,
                 num_ks: int = 21,
                 grid_resolution: int = 64,
                 spatial_dim: int = 2,
                 *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.grid_resolution = grid_resolution
        
        self.fourier_interpolator = FourierInterpolator(num_ks=num_ks, spatial_dim=spatial_dim)
        self.interpolation_points = self.generate_interpolation_points(grid_resolution)
        
        self.prediction_net = prediction_net
        
        
    def generate_interpolation_points(self, grid_resolution):
        x_grid, y_grid = torch.meshgrid(torch.linspace(0, 1, grid_resolution), torch.linspace(0, 1, grid_resolution))
        p_grid = torch.stack([y_grid, x_grid], dim=-1).reshape(-1, 2)
        return p_grid
        
    def forward(self, x, p, t_eval=[0.0, 1.0]):
        # check devices
        if self.interpolation_points.device != p.device:
            self.interpolation_points = self.interpolation_points.to(p.device)
            
        # interpolate on a grid
        x_grid = self.fourier_interpolator(p, x, self.interpolation_points)
        x_grid = x_grid.view(x.shape[0], self.grid_resolution, self.grid_resolution, x.shape[-1])
        x_grid = x_grid.permute(0, 3, 1, 2)
        
        # resnet smoother
        if hasattr(self, "smoother"):
            x_grid = x_grid + self.smoother(x_grid)
        
        # run solver
        x_pred = self.prediction_net(x_grid, t_eval=t_eval)
        
        # interpolate back to the original points
        x_pred = x_pred.permute(1, 0, 3, 4, 2)
        x_pred = x_pred.reshape(x.shape[0], len(t_eval[1:]), self.grid_resolution**2, x.shape[-1])
        x_pred = self.fourier_interpolator(self.interpolation_points.view(1,1,*self.interpolation_points.shape), x_pred, p.unsqueeze(1))
        return x_pred