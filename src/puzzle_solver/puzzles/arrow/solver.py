"""Solver for Arrow Puzzle using Propagation algorithm."""

from typing import List, Tuple
from ...core import BaseSolver
from .board import ArrowBoard


class ArrowSolver(BaseSolver):
    """Solver for Arrow Puzzle."""

    def __init__(self, board: ArrowBoard):
        """Initialize solver with a board.

        Args:
            board: The board to solve
        """
        super().__init__(board)
        self.board: ArrowBoard = board.copy()
        self.original_board: ArrowBoard = board.copy()
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

    def solve_hard_expert(self, mode: str = "expert") -> bool:
        """Solve the board using Hard/Expert mode algorithm.

        Args:
            mode: "hard" or "expert" mode
            
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

        # Step 5: If B + D is odd, tap 'c' 
        if (B + D) % 2 == 1:
            col_c = start_col + 2
            # Tap 3 times in Expert mode, 1 time in Hard mode
            tap_count = 3 if mode == "expert" else 1
            for _ in range(tap_count):
                self.board.tap(top_row, col_c)
                self.moves.append((top_row, col_c))

        # Step 6: Propagate from top once more
        self.propagate()

        return bool(self.board.is_solved())

    def solve(self, mode: str = "expert", **kwargs) -> bool:
        """Solve the board.

        Args:
            mode: Solving mode ("normal", "hard", or "expert")
            **kwargs: Additional parameters (unused)

        Returns:
            True if solved successfully, False otherwise
        """
        self.stats["mode"] = mode
        self.stats["board_size"] = self.board.size
        
        if mode in ["hard", "expert"]:
            result = self.solve_hard_expert(mode)
        else:
            # For normal mode, just propagate
            self.propagate()
            result = bool(self.board.is_solved())
        
        self.stats["moves_count"] = len(self.moves)
        self.stats["solved"] = result
        self.solution = self.moves
        
        return result

    def get_solution(self) -> List[Tuple[int, int]]:
        """Get the solution moves."""
        return self.moves.copy()

    def get_solution_steps(self) -> List[str]:
        """Get human-readable solution steps."""
        steps = []
        for i, (row, col) in enumerate(self.moves):
            steps.append(f"Step {i+1}: Tap cell at row {row+1}, column {col+1}")
        return steps

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

    def get_algorithm_name(self) -> str:
        """Get the name of the solving algorithm."""
        return "Propagation Algorithm with Hard/Expert Mode"

    def estimate_difficulty(self) -> str:
        """Estimate the difficulty of the puzzle."""
        # Simple heuristic based on initial board state
        initial_sum = sum(sum(row) for row in self.original_board.grid)
        board_cells = self.original_board.size ** 2
        
        avg_value = initial_sum / board_cells
        
        if avg_value < 1.5:
            return "easy"
        elif avg_value < 2.5:
            return "medium"
        else:
            return "hard"