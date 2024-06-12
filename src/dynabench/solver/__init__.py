"""
This module contains all different backends for the DynaBench solver.
"""

from ._base import BaseSolver
from ._pypde import PyPDESolver

__all__ = ["BaseSolver", "PyPDESolver"]