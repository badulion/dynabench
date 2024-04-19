"""
Module for representing partial differential equations.
"""

from ._base import BaseEquation, Term
from ._advection import AdvectionEquation

__all__ = ["BaseEquation", "Term", "AdvectionEquation"]