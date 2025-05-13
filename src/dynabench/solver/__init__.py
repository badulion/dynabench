"""
This module contains all different backends for the DynaBench solver.
"""

from ._base import BaseSolver
from ._pypde import PyPDESolver

__all__ = ["BaseSolver", "PyPDESolver"]


try :
    from ._dedalus import DedalusSolver
    __all__.append("DedalusSolver")
except ImportError:
    import warnings
    # Dedalus is not installed, so we won't be able to use it
    warnings.warn("Dedalus is not installed. DedalusSolver will not be available.")

