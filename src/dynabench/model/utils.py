import torch

from typing import List, Optional

import einops

class RolloutWrapper(torch.nn.Module):
    """
    Wrapper class for iterative model evaluation.
    This class is designed to perform iterative evaluation of models by calling the model multiple times at different time points.
    It can be used for both point-based and grid-based models.

    Parameters
    ----------
    model : torch.nn.Module
        The model to be wrapped and iteratively evaluated.
    batch_first : bool, default True
        If True, the first dimension of the input tensor is considered as the batch dimension. If False, the first dimension is the rollout dimension.
    feature_dim: int, default -1
        The id of the feature dimension. 
    lookback_dim: int, default 1
        The id of the lookback dimension. 


    Methods
    -------
    forward(x: torch.Tensor, p: torch.Tensor, t_eval: List[float] = [1]) -> torch.Tensor
        Perform iterative evaluation of the model at specified time points.
    """
    def __init__(self, 
                 model,
                 structure: str = 'grid',
                 batch_first: bool = True,
                 lookback_dim: int = 1):
        super().__init__()
        if structure not in ['grid', 'cloud']:
            raise ValueError("Structure must be either 'grid' or 'cloud'")
        self.structure = structure
        self.model = model
        self.batch_first = batch_first
        
        self.feature_dim = 2 if structure == 'grid' else -1
        
        self.lookback_dim = lookback_dim
        self.alphabet = 'abcdefghijklmnopqrstuvwxyz'
        
    def forward(self, 
                x: torch.Tensor, # features
                p: Optional[torch.Tensor] = None, # point coordinates
                t_eval: List[float] = [1]):
        
        rollout = []
        for t in t_eval:
            x_stacked_lookback = einops.rearrange(x, self._einops_stack_lookback_expr()) # Merge lookback with the feature dimension
            
            args = (x_stacked_lookback,) if self.structure=="grid" else (x_stacked_lookback, p)
            x_single = self.model(*args)
            
            x_single_unstacked_loockback = einops.rearrange(x_single, "batch ... -> batch () ...") # add dummy dim for lookback in pred
            x = torch.cat([x[:, 1:], x_single_unstacked_loockback], dim=self.lookback_dim)
            
            rollout.append(x_single)
            
            
        rollout_dim = 1 if self.batch_first else 0
        return torch.stack(rollout, dim=rollout_dim)
            
    def _einops_stack_lookback_expr(self):
        if self.structure == "grid":
            expr = 'batch lookback feature ... -> batch (lookback feature) ...'
        elif self.structure == "cloud":
            # Generate einops expression for cloud structure
            expr = 'batch lookback points feature -> batch points (lookback feature)'
        else:
            raise ValueError("Structure must be either 'grid' or 'cloud'")    
        return expr

    def _einops_unstack_loockback_expr(self):
        if self.structure == "grid":
            expr = 'batch (lookback feature) ... -> batch lookback feature ...'
        elif self.structure == "cloud":
            # Generate einops expression for cloud structure
            expr = 'batch points (lookback feature) -> batch lookback points feature'
        else:
            raise ValueError("Structure must be either 'grid' or 'cloud'")    
        
        return expr
        

class CloudRolloutWrapper(RolloutWrapper):
    """
        Alias for `dynabench.model.utils.RolloutWrapper with structure="cloud"
    """
    def __init__(self,
                 model,
                 batch_first: bool = True,
                 lookback_dim: int = 1):
        super().__init__(model=model, 
                         structure="cloud", 
                         batch_first=batch_first,
                         lookback_dim=lookback_dim)
        
class GridRolloutWrapper(RolloutWrapper):
    """
        Alias for `dynabench.model.utils.RolloutWrapper with structure="grid"
    """
    def __init__(self,
                 model,
                 batch_first: bool = True,
                 lookback_dim: int = 1):
        super().__init__(model=model, 
                         structure="grid", 
                         batch_first=batch_first,
                         lookback_dim=lookback_dim)