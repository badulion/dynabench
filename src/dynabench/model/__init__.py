"""
Module containing the models used in the DynaBench benchmark.
"""

from ._grid.cnn import CNN
from ._grid.resnet import ResNet
from ._grid.neuralpde import NeuralPDE

from .point.point_transformer import PointTransformerV1


__all__ = ['CNN', 'ResNet', 'NeuralPDE', 'PointTransformerV1']