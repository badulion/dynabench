"""
Module for loading the data.
"""

from ._download import download_equation
from ._iterator import DynabenchIterator
from ._simulation_iterator import DynabenchSimulationIterator

__all__ = ["download_equation", "DynabenchIterator", "DynabenchSimulationIterator"]