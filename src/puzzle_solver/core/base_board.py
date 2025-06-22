"""Base board interface for all puzzle types."""

from abc import ABC, abstractmethod
from typing import Any, List, Tuple, Optional


class BaseBoard(ABC):
    """Abstract base class for puzzle boards."""

    @abstractmethod
    def __init__(self, *args, **kwargs):
        """Initialize the board with puzzle-specific parameters."""
        pass

    @abstractmethod
    def is_solved(self) -> bool:
        """Check if the puzzle is in a solved state.
        
        Returns:
            True if the puzzle is solved, False otherwise
        """
        pass

    @abstractmethod
    def get_legal_moves(self) -> List[Any]:
        """Get all legal moves from the current state.
        
        Returns:
            List of legal moves (format depends on puzzle type)
        """
        pass

    @abstractmethod
    def apply_move(self, move: Any) -> None:
        """Apply a move to the board.
        
        Args:
            move: The move to apply (format depends on puzzle type)
        """
        pass

    @abstractmethod
    def copy(self) -> "BaseBoard":
        """Create a deep copy of the board.
        
        Returns:
            A new board instance with the same state
        """
        pass

    @abstractmethod
    def __str__(self) -> str:
        """Get string representation of the board.
        
        Returns:
            Human-readable string representation
        """
        pass

    @abstractmethod
    def __repr__(self) -> str:
        """Get detailed representation of the board.
        
        Returns:
            Detailed string representation
        """
        pass

    @abstractmethod
    def to_dict(self) -> dict:
        """Convert board state to dictionary.
        
        Returns:
            Dictionary representation of the board state
        """
        pass

    @abstractmethod
    def from_dict(self, data: dict) -> None:
        """Load board state from dictionary.
        
        Args:
            data: Dictionary containing board state
        """
        pass

    @abstractmethod
    def get_state_hash(self) -> str:
        """Get a hash representing the current board state.
        
        Returns:
            Hash string that uniquely identifies the state
        """
        pass