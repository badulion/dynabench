from ._v1 import PointTransformerV1

__all__ = ["PointTransformerV1"]

try:
    from ._v3 import PointTransformerV3
    __all__ = ["PointTransformerV1", "PointTransformerV3"]
except ImportError:
    pass

