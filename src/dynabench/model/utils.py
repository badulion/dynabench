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
    structure : str, default 'grid'
        The structure of the input data. Can be either 'grid' or 'cloud'.
    is_lookback_squeezed : bool, default False
        If True, the lookback dimension is squeezed. If True, the lookback dimension parameter is ignored.
    """
    def __init__(self, 
                 model,
                 structure: str = 'grid',
                 batch_first: bool = True,
                 lookback_dim: int = 1,
                 is_lookback_squeezed: bool = False):
        super().__init__()
        if structure not in ['grid', 'cloud']:
            raise ValueError("Structure must be either 'grid' or 'cloud'")
        self.structure = structure
        self.model = model
        self.batch_first = batch_first
        
        self.feature_dim = 2 if structure == 'grid' else -1
        
        self.lookback_dim = lookback_dim
        self.is_lookback_squeezed = is_lookback_squeezed
        self.alphabet = 'abcdefghijklmnopqrstuvwxyz'
        
    def forward(self, 
                x: torch.Tensor, # features
                p: Optional[torch.Tensor] = None, # point coordinates
                t_eval: List[float] = [1]):
        
        rollout = []
        for t in t_eval:
            x_stacked_lookback = self._stack_lookback(x) # Merge lookback with the feature dimension
            
            x_single = self._single_step(x_stacked_lookback, p) # Call the model once
            
            x = self._wrap_input_with_lookback(x, x_single) # Wrap the input with the new prediction
            
            rollout.append(x_single)
            
            
        rollout_dim = 1 if self.batch_first else 0
        return torch.stack(rollout, dim=rollout_dim)
            
    def _stack_lookback(self, x):
        if self.structure == "grid":
            expr = 'batch lookback feature ... -> batch (lookback feature) ...'
        elif self.structure == "cloud":
            # Generate einops expression for cloud structure
            expr = 'batch lookback points feature -> batch points (lookback feature)'
        else:
            raise ValueError("Structure must be either 'grid' or 'cloud'")    
        
        if not self.is_lookback_squeezed:
            return einops.rearrange(x, expr)
        else:
            return x 
    
    def _single_step(self, x, p):
        if p is not None:
            x_single = self.model(x, p)
        else:
            x_single = self.model(x)
        return x_single
                

    def _wrap_input_with_lookback(self, x_previous, x_pred_single):
        if not self.is_lookback_squeezed:
            x_single_unstacked_loockback = einops.rearrange(x_pred_single, "batch ... -> batch () ...") # add dummy dim for lookback in pred
            x_next = torch.cat([x_previous[:, 1:], x_single_unstacked_loockback], dim=self.lookback_dim)
        else:
            x_next = x_pred_single
        return x_next

class CloudRolloutWrapper(RolloutWrapper):
    """
        Alias for `dynabench.model.utils.RolloutWrapper` with structure="cloud"
    """
    def __init__(self,
                 model,
                 batch_first: bool = True,
                 lookback_dim: int = 1,
                 is_lookback_squeezed: bool = False):
        super().__init__(model=model, 
                         structure="cloud", 
                         batch_first=batch_first,
                         lookback_dim=lookback_dim,
                         is_lookback_squeezed=is_lookback_squeezed)
        
class GridRolloutWrapper(RolloutWrapper):
    """
        Alias for `dynabench.model.utils.RolloutWrapper` with structure="grid"
    """
    def __init__(self,
                 model,
                 batch_first: bool = True,
                 lookback_dim: int = 1,
                 is_lookback_squeezed: bool = False):
        super().__init__(model=model, 
                         structure="grid", 
                         batch_first=batch_first,
                         lookback_dim=lookback_dim,
                         is_lookback_squeezed=is_lookback_squeezed)