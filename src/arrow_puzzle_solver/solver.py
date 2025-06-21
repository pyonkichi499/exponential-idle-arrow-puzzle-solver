"""Solver for Arrow Puzzle using Propagation algorithm."""

from typing import List, Tuple
from .board import Board


class Solver:
    """Solver for Arrow Puzzle."""

    def __init__(self, board: Board):
        """Initialize solver with a board.

        Args:
            board: The board to solve
        """
        self.board = board.copy()
        self.original_board = board.copy()
        self.moves: List[Tuple[int, int]] = []

    def solve_cell(self, row: int, col: int) -> None:
        """Solve a specific cell by tapping below it.

        Args:
            row: Row index of the cell to solve
            col: Column index of the cell to solve
        """
        # To solve a cell in row i, we tap the cell in row i+1
        if row + 1 < self.board.size:
            # Calculate how many taps needed
            current_value = self.board.get_value(row, col)
            taps_needed = (1 - current_value) % 5

            for _ in range(taps_needed):
                self.board.tap(row + 1, col)
                self.moves.append((row + 1, col))

    def propagate_row(self, row: int) -> None:
        """Apply propagation algorithm to a specific row.

        Args:
            row: Row index to propagate
        """
        center = self.board.size // 2

        # Step a: Solve the center tile
        self.solve_cell(row, center)

        # Steps b-d: Solve tiles to the left of center
        for offset in range(1, center + 1):
            if center - offset >= 0:
                self.solve_cell(row, center - offset)

        # Step e: Solve tiles to the right of center
        for offset in range(1, center + 1):
            if center + offset < self.board.size:
                self.solve_cell(row, center + offset)

    def propagate(self) -> None:
        """Apply propagation algorithm to the entire board."""
        # Propagate all rows except the bottom row
        for row in range(self.board.size - 1):
            self.propagate_row(row)

    def solve_hard_expert(self) -> bool:
        """Solve the board using Hard/Expert mode algorithm.

        Returns:
            True if solved successfully, False otherwise
        """
        # Step 1: Propagate
        self.propagate()

        # Get bottom row and top row indices
        bottom_row = self.board.size - 1
        top_row = 0

        # Label bottom right cells: A, B, C, D (from left to right)
        # Label top right cells: a, b, c, d (from left to right)
        # In a 7x7 board, these would be columns 3, 4, 5, 6
        start_col = self.board.size - 4

        # Get values
        # A = self.board.get_value(bottom_row, start_col)  # Not used in current algorithm
        B = self.board.get_value(bottom_row, start_col + 1)
        C = self.board.get_value(bottom_row, start_col + 2)
        D = self.board.get_value(bottom_row, start_col + 3)

        # Step 2: Tap 'a' so that 'a' is the same as C
        col_a = start_col
        current_a = self.board.get_value(top_row, col_a)
        taps_needed = (C - current_a) % 5
        for _ in range(taps_needed):
            self.board.tap(top_row, col_a)
            self.moves.append((top_row, col_a))

        # Step 3: Tap 'b' and 'd' the number of times needed to solve C
        col_b = start_col + 1
        col_d = start_col + 3
        taps_for_C = (1 - C) % 5

        for _ in range(taps_for_C):
            self.board.tap(top_row, col_b)
            self.moves.append((top_row, col_b))
            self.board.tap(top_row, col_d)
            self.moves.append((top_row, col_d))

        # Step 4: Tap 'a' the number of times needed to solve D
        taps_for_D = (1 - D) % 5
        for _ in range(taps_for_D):
            self.board.tap(top_row, col_a)
            self.moves.append((top_row, col_a))

        # Step 5: If B + D is odd, tap 'c' three times (or once in Hard mode)
        if (B + D) % 2 == 1:
            col_c = start_col + 2
            # For now, assuming Expert mode (3 taps)
            for _ in range(3):
                self.board.tap(top_row, col_c)
                self.moves.append((top_row, col_c))

        # Step 6: Propagate from top once more
        self.propagate()

        return bool(self.board.is_solved())

    def solve(self, mode: str = "expert") -> bool:
        """Solve the board.

        Args:
            mode: Solving mode ("normal", "hard", or "expert")

        Returns:
            True if solved successfully, False otherwise
        """
        if mode in ["hard", "expert"]:
            return self.solve_hard_expert()
        else:
            # For normal mode, just propagate
            self.propagate()
            return bool(self.board.is_solved())

    def get_moves(self) -> List[Tuple[int, int]]:
        """Get the list of moves made to solve the board."""
        return self.moves.copy()

    def verify_solution(self) -> bool:
        """Verify that the solution is correct."""
        if not self.board.is_solved():
            return False

        # Check if board is symmetric (mentioned in the guide)
        if not self.board.is_symmetric():
            print("Warning: Board is not symmetric")

        # Check bottom center adjacent tiles
        bottom_row = self.board.size - 1
        center = self.board.size // 2

        if center - 1 >= 0:
            left_val = self.board.get_value(bottom_row, center - 1)
            if left_val not in [1, 4]:
                print(
                    f"Warning: Bottom center left tile is {left_val}, expected 1 or 4"
                )

        if center + 1 < self.board.size:
            right_val = self.board.get_value(bottom_row, center + 1)
            if right_val not in [1, 4]:
                print(
                    f"Warning: Bottom center right tile is {right_val}, expected 1 or 4"
                )

        return True
