"""15 puzzle board implementation."""

import numpy as np
import hashlib
from typing import List, Tuple, Optional, Any
from ...core import BaseBoard


class FifteenBoard(BaseBoard):
    """Represents a 15 puzzle (sliding puzzle) board."""

    def __init__(self, size: int = 4):
        """Initialize a 15 puzzle board.

        Args:
            size: Size of the board (default is 4x4)
        """
        self.size = size
        self.max_num = size * size - 1  # 15 for 4x4
        self.grid = np.zeros((size, size), dtype=int)
        self.empty_pos = (size - 1, size - 1)  # Empty position starts at bottom-right
        
        # Initialize with solved state
        num = 1
        for i in range(size):
            for j in range(size):
                if i == size - 1 and j == size - 1:
                    self.grid[i, j] = 0  # 0 represents empty space
                else:
                    self.grid[i, j] = num
                    num += 1

    def copy(self) -> "FifteenBoard":
        """Create a deep copy of the board."""
        new_board = FifteenBoard(self.size)
        new_board.grid = self.grid.copy()
        new_board.empty_pos = self.empty_pos
        return new_board

    def is_solved(self) -> bool:
        """Check if the puzzle is solved."""
        num = 1
        for i in range(self.size):
            for j in range(self.size):
                if i == self.size - 1 and j == self.size - 1:
                    # Last position should be empty (0)
                    if self.grid[i, j] != 0:
                        return False
                else:
                    if self.grid[i, j] != num:
                        return False
                    num += 1
        return True

    def get_legal_moves(self) -> List[str]:
        """Get all legal moves from current state.
        
        Returns:
            List of legal moves: ["up", "down", "left", "right"]
        """
        moves = []
        empty_row, empty_col = self.empty_pos
        
        # Check each direction
        if empty_row > 0:  # Can move tile down (empty space moves up)
            moves.append("up")
        if empty_row < self.size - 1:  # Can move tile up (empty space moves down)
            moves.append("down")
        if empty_col > 0:  # Can move tile right (empty space moves left)
            moves.append("left")
        if empty_col < self.size - 1:  # Can move tile left (empty space moves right)
            moves.append("right")
            
        return moves

    def apply_move(self, move: str) -> None:
        """Apply a move to the board.
        
        Args:
            move: Direction to move the empty space ("up", "down", "left", "right")
        """
        empty_row, empty_col = self.empty_pos
        
        # Calculate new position for empty space
        if move == "up" and empty_row > 0:
            new_row, new_col = empty_row - 1, empty_col
        elif move == "down" and empty_row < self.size - 1:
            new_row, new_col = empty_row + 1, empty_col
        elif move == "left" and empty_col > 0:
            new_row, new_col = empty_row, empty_col - 1
        elif move == "right" and empty_col < self.size - 1:
            new_row, new_col = empty_row, empty_col + 1
        else:
            return  # Invalid move
        
        # Swap empty space with tile
        self.grid[empty_row, empty_col] = self.grid[new_row, new_col]
        self.grid[new_row, new_col] = 0
        self.empty_pos = (new_row, new_col)

    def get_tile_at(self, row: int, col: int) -> int:
        """Get the tile number at a specific position."""
        return self.grid[row, col]

    def find_tile(self, tile: int) -> Optional[Tuple[int, int]]:
        """Find the position of a specific tile.
        
        Args:
            tile: Tile number to find (0 for empty)
            
        Returns:
            (row, col) position of the tile, or None if not found
        """
        for i in range(self.size):
            for j in range(self.size):
                if self.grid[i, j] == tile:
                    return (i, j)
        return None

    def manhattan_distance(self) -> int:
        """Calculate total Manhattan distance for all tiles."""
        distance = 0
        for i in range(self.size):
            for j in range(self.size):
                tile = self.grid[i, j]
                if tile != 0:  # Skip empty space
                    # Calculate goal position
                    goal_row = (tile - 1) // self.size
                    goal_col = (tile - 1) % self.size
                    # Add Manhattan distance
                    distance += abs(i - goal_row) + abs(j - goal_col)
        return distance

    def is_solvable(self) -> bool:
        """Check if the current configuration is solvable.
        
        Uses the inversion count method for solvability check.
        """
        # Flatten grid and count inversions
        flat = []
        for i in range(self.size):
            for j in range(self.size):
                if self.grid[i, j] != 0:
                    flat.append(self.grid[i, j])
        
        inversions = 0
        for i in range(len(flat)):
            for j in range(i + 1, len(flat)):
                if flat[i] > flat[j]:
                    inversions += 1
        
        # For odd-sized boards, solvable if inversions is even
        if self.size % 2 == 1:
            return inversions % 2 == 0
        else:
            # For even-sized boards, consider empty space row
            empty_row, _ = self.empty_pos
            # Count from bottom (1-indexed)
            empty_row_from_bottom = self.size - empty_row
            return (inversions + empty_row_from_bottom) % 2 == 1

    def __str__(self) -> str:
        """String representation of the board."""
        lines = []
        for row in self.grid:
            line = []
            for val in row:
                if val == 0:
                    line.append("  ")  # Empty space
                else:
                    line.append(f"{val:2d}")
            lines.append(" ".join(line))
        return "\n".join(lines)

    def __repr__(self) -> str:
        """Detailed representation of the board."""
        return f"FifteenBoard(size={self.size})\n{self}"

    def to_dict(self) -> dict:
        """Convert board state to dictionary."""
        return {
            "size": self.size,
            "grid": self.grid.tolist(),
            "empty_pos": self.empty_pos,
            "puzzle_type": "fifteen"
        }

    def from_dict(self, data: dict) -> None:
        """Load board state from dictionary."""
        self.size = data["size"]
        self.grid = np.array(data["grid"], dtype=int)
        self.empty_pos = tuple(data["empty_pos"])
        self.max_num = self.size * self.size - 1

    def get_state_hash(self) -> str:
        """Get a hash representing the current board state."""
        state_str = f"{self.size}:" + ",".join(str(val) for row in self.grid for val in row)
        return hashlib.md5(state_str.encode()).hexdigest()

    def shuffle(self, moves: int = 100) -> None:
        """Shuffle the board by making random legal moves.
        
        Args:
            moves: Number of random moves to make
        """
        import random
        for _ in range(moves):
            legal_moves = self.get_legal_moves()
            if legal_moves:
                move = random.choice(legal_moves)
                self.apply_move(move)