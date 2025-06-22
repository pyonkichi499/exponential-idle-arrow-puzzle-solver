"""15 puzzle (sliding puzzle) implementation."""

from .board import FifteenBoard
from .solver import FifteenSolver

__all__ = ["FifteenBoard", "FifteenSolver"]

# Import vision only if opencv is available
try:
    from .vision import FifteenVision
    __all__.append("FifteenVision")
except ImportError:
    FifteenVision = None