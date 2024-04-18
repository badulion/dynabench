"""
This module contains the classes for representing equations.
"""

from ._base import BaseEquation, Term
from ._advection import AdvectionEquation

__all__ = ["BaseEquation", "Term", "AdvectionEquation"]