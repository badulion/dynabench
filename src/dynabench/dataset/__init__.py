"""
Module for loading the data.
"""

from ._download import download_equation
from ._dynabench import DynabenchIterator, DynabenchSimulationIterator
from ._equation import EquationMovingWindowIterator

__all__ = ["download_equation", "DynabenchIterator", "DynabenchSimulationIterator", "EquationMovingWindowIterator"]