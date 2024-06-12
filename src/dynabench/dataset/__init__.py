"""
Module for loading the data.
"""

from ._download import download_equation
from ._dynabench import DynabenchIterator, DynabenchSimulationIterator

__all__ = ["download_equation", "DynabenchIterator", "DynabenchSimulationIterator"]