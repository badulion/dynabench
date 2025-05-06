"""
Module for loading the data.
"""

from ._download import download_equation
from ._dynabench import DynabenchIterator, DynabenchSimulationIterator
from ._equation import EquationMovingWindowIterator, EquationSimulationIterator
from ._base import BaseListMovingWindowIterator, BaseListSimulationIterator
from ._transforms import MakeKNNGraph, Grid2Cloud, Compose

__all__ = ["download_equation", 
           "DynabenchIterator", 
           "DynabenchSimulationIterator", 
           "EquationMovingWindowIterator",
           "EquationSimulationIterator",
           "BaseListMovingWindowIterator",
           "BaseListSimulationIterator",
           "MakeKNNGraph",
           "Grid2Cloud",
           "Compose"]
