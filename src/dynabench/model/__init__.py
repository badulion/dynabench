"""
Module containing the models used in the DynaBench benchmark.
"""

from ._grid.cnn import CNN
from ._grid.resnet import ResNet


__all__ = ['CNN', 'ResNet']