"""Registry for available puzzle types."""

from typing import Dict, Type, Tuple
from ..core import BaseBoard, BaseSolver, BaseVision


# Registry for puzzle implementations
_PUZZLE_REGISTRY: Dict[str, Tuple[Type[BaseBoard], Type[BaseSolver], Type[BaseVision]]] = {}


def register_puzzle(name: str, board_class: Type[BaseBoard], 
                   solver_class: Type[BaseSolver], 
                   vision_class: Type[BaseVision]) -> None:
    """Register a new puzzle type.
    
    Args:
        name: Name of the puzzle type
        board_class: Board implementation class
        solver_class: Solver implementation class
        vision_class: Vision implementation class
    """
    _PUZZLE_REGISTRY[name] = (board_class, solver_class, vision_class)


def get_puzzle_board(puzzle_type: str) -> Type[BaseBoard]:
    """Get board class for a puzzle type.
    
    Args:
        puzzle_type: Name of the puzzle type
        
    Returns:
        Board class for the puzzle
        
    Raises:
        ValueError: If puzzle type not found
    """
    if puzzle_type not in _PUZZLE_REGISTRY:
        raise ValueError(f"Unknown puzzle type: {puzzle_type}")
    return _PUZZLE_REGISTRY[puzzle_type][0]


def get_puzzle_solver(puzzle_type: str) -> Type[BaseSolver]:
    """Get solver class for a puzzle type.
    
    Args:
        puzzle_type: Name of the puzzle type
        
    Returns:
        Solver class for the puzzle
        
    Raises:
        ValueError: If puzzle type not found
    """
    if puzzle_type not in _PUZZLE_REGISTRY:
        raise ValueError(f"Unknown puzzle type: {puzzle_type}")
    return _PUZZLE_REGISTRY[puzzle_type][1]


def get_puzzle_vision(puzzle_type: str) -> Type[BaseVision]:
    """Get vision class for a puzzle type.
    
    Args:
        puzzle_type: Name of the puzzle type
        
    Returns:
        Vision class for the puzzle
        
    Raises:
        ValueError: If puzzle type not found
    """
    if puzzle_type not in _PUZZLE_REGISTRY:
        raise ValueError(f"Unknown puzzle type: {puzzle_type}")
    return _PUZZLE_REGISTRY[puzzle_type][2]


@property
def AVAILABLE_PUZZLES() -> list:
    """Get list of available puzzle types."""
    return list(_PUZZLE_REGISTRY.keys())


# Import and register puzzle implementations
try:
    from .arrow import ArrowBoard, ArrowSolver, ArrowVision
    register_puzzle("arrow", ArrowBoard, ArrowSolver, ArrowVision)
except ImportError:
    pass

try:
    from .fifteen import FifteenBoard, FifteenSolver, FifteenVision
    register_puzzle("fifteen", FifteenBoard, FifteenSolver, FifteenVision)
except ImportError:
    pass