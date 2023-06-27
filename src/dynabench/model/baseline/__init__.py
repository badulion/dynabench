from .baseline_persistence import BaselinePersistence
from .baseline_difference import BaselineDifference
from .baseline_zero import BaselineZero


def is_baseline(model):
    if isinstance(model, BaselinePersistence):
        return True
    elif isinstance(model, BaselinePersistence):
        return True
    else:
        return False