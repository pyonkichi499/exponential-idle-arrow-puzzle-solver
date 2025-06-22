"""Core interfaces for puzzle solver framework."""

from .base_board import BaseBoard
from .base_solver import BaseSolver
from .base_vision import BaseVision

__all__ = ["BaseBoard", "BaseSolver", "BaseVision"]