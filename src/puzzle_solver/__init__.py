"""Puzzle Solver - A framework for solving various puzzle types."""

__version__ = "0.2.0"

# Import core interfaces
from .core import BaseBoard, BaseSolver, BaseVision

# Import puzzle implementations
from .puzzles import AVAILABLE_PUZZLES, get_puzzle_board, get_puzzle_solver, get_puzzle_vision

__all__ = [
    "BaseBoard",
    "BaseSolver", 
    "BaseVision",
    "AVAILABLE_PUZZLES",
    "get_puzzle_board",
    "get_puzzle_solver",
    "get_puzzle_vision",
]