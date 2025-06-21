"""Board representation for Arrow Puzzle."""

import numpy as np


class Board:
    """Represents an Arrow Puzzle board."""

    def __init__(self, size: int = 7):
        """Initialize a board with given size.

        Args:
            size: Size of the board (default is 7x7)
        """
        self.size = size
        self.grid = np.zeros((size, size), dtype=int)

    def copy(self) -> "Board":
        """Create a deep copy of the board."""
        new_board = Board(self.size)
        new_board.grid = self.grid.copy()
        return new_board

    def tap(self, row: int, col: int) -> None:
        """Tap a cell to increment its value and adjacent cells.

        Args:
            row: Row index (0-based)
            col: Column index (0-based)
        """
        # Increment the tapped cell
        self.grid[row, col] = (self.grid[row, col] + 1) % 5

        # Increment adjacent cells (up, down, left, right)
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        for dr, dc in directions:
            new_row, new_col = row + dr, col + dc
            if 0 <= new_row < self.size and 0 <= new_col < self.size:
                self.grid[new_row, new_col] = (self.grid[new_row, new_col] + 1) % 5

    def is_solved(self) -> bool:
        """Check if the board is solved (all cells are 1)."""
        return np.all(self.grid == 1)

    def is_symmetric(self) -> bool:
        """Check if the board is symmetric."""
        return np.array_equal(self.grid, np.fliplr(self.grid))

    def get_value(self, row: int, col: int) -> int:
        """Get the value at a specific cell.

        Args:
            row: Row index (0-based)
            col: Column index (0-based)

        Returns:
            Value at the cell (0-4)
        """
        return self.grid[row, col]

    def set_value(self, row: int, col: int, value: int) -> None:
        """Set the value at a specific cell.

        Args:
            row: Row index (0-based)
            col: Column index (0-based)
            value: Value to set (0-4)
        """
        self.grid[row, col] = value % 5

    def get_row(self, row: int) -> np.ndarray:
        """Get a specific row."""
        return self.grid[row].copy()

    def get_col(self, col: int) -> np.ndarray:
        """Get a specific column."""
        return self.grid[:, col].copy()

    def __str__(self) -> str:
        """String representation of the board."""
        return "\n".join([" ".join(map(str, row)) for row in self.grid])

    def __repr__(self) -> str:
        """Detailed representation of the board."""
        return f"Board(size={self.size})\n{self}"
