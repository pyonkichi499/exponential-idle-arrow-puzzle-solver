"""Base solver interface for all puzzle types."""

from abc import ABC, abstractmethod
from typing import List, Optional, Any, Dict
from .base_board import BaseBoard


class BaseSolver(ABC):
    """Abstract base class for puzzle solvers."""

    def __init__(self, board: BaseBoard):
        """Initialize solver with a board.
        
        Args:
            board: The puzzle board to solve
        """
        self.board = board.copy()
        self.original_board = board.copy()
        self.solution: List[Any] = []
        self.stats: Dict[str, Any] = {}

    @abstractmethod
    def solve(self, **kwargs) -> bool:
        """Solve the puzzle.
        
        Args:
            **kwargs: Puzzle-specific solving parameters
            
        Returns:
            True if solution found, False otherwise
        """
        pass

    @abstractmethod
    def get_solution(self) -> List[Any]:
        """Get the solution moves.
        
        Returns:
            List of moves to solve the puzzle
        """
        pass

    @abstractmethod
    def get_solution_steps(self) -> List[str]:
        """Get human-readable solution steps.
        
        Returns:
            List of strings describing each step
        """
        pass

    @abstractmethod
    def verify_solution(self) -> bool:
        """Verify that the solution is correct.
        
        Returns:
            True if solution is valid, False otherwise
        """
        pass

    def get_stats(self) -> Dict[str, Any]:
        """Get solving statistics.
        
        Returns:
            Dictionary containing solving statistics
        """
        return self.stats

    def reset(self) -> None:
        """Reset the solver to initial state."""
        self.board = self.original_board.copy()
        self.solution = []
        self.stats = {}

    @abstractmethod
    def get_algorithm_name(self) -> str:
        """Get the name of the solving algorithm.
        
        Returns:
            Name of the algorithm used
        """
        pass

    @abstractmethod
    def estimate_difficulty(self) -> str:
        """Estimate the difficulty of the puzzle.
        
        Returns:
            Difficulty level (e.g., "easy", "medium", "hard")
        """
        pass