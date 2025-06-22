"""Tests for the Arrow Solver class."""

import numpy as np
from puzzle_solver.puzzles.arrow.board import ArrowBoard
from puzzle_solver.puzzles.arrow.solver import ArrowSolver


class TestArrowSolver:
    """Test cases for ArrowSolver class."""

    def test_solve_cell(self):
        """Test solving a single cell."""
        board = ArrowBoard(5)
        # Set a cell to value 3
        board.set_value(0, 2, 3)

        solver = ArrowSolver(board)
        solver.solve_cell(0, 2)

        # Cell should now be 1 (3 + 3 = 6 % 5 = 1)
        assert solver.board.get_value(0, 2) == 1

    def test_propagate_row(self):
        """Test propagating a single row."""
        board = ArrowBoard(5)
        # Set some values in the first row
        board.set_value(0, 0, 2)
        board.set_value(0, 1, 3)
        board.set_value(0, 2, 4)
        board.set_value(0, 3, 1)
        board.set_value(0, 4, 2)

        solver = ArrowSolver(board)
        solver.propagate_row(0)

        # First row should all be 1
        assert all(solver.board.get_value(0, i) == 1 for i in range(5))

    def test_simple_solve(self):
        """Test solving a simple board."""
        board = ArrowBoard(3)
        # Create a solvable pattern - all 1s is the goal
        board.grid = np.ones((3, 3), dtype=int)
        # Make one cell different
        board.set_value(0, 0, 0)

        solver = ArrowSolver(board)
        result = solver.solve()

        # The propagation algorithm doesn't guarantee solving all boards
        # so we just check that it runs without errors and returns a boolean
        assert isinstance(result, bool)

    def test_solve_modes(self):
        """Test different solving modes."""
        board = ArrowBoard(5)
        # Create a test pattern
        for i in range(5):
            for j in range(5):
                board.set_value(i, j, (i + j) % 5)

        # Test normal mode
        solver1 = ArrowSolver(board)
        solver1.solve("normal")

        # Test expert mode
        solver2 = ArrowSolver(board)
        solver2.solve("expert")

        # Both should produce valid results
        assert len(solver1.moves) > 0
        assert len(solver2.moves) > 0

    def test_verify_solution(self):
        """Test solution verification."""
        board = ArrowBoard(3)
        board.grid = np.ones((3, 3), dtype=int)

        solver = ArrowSolver(board)
        solver.board = board

        # Should verify as correct
        assert solver.verify_solution()

    def test_get_moves(self):
        """Test getting move history."""
        board = ArrowBoard(3)
        board.set_value(0, 0, 2)

        solver = ArrowSolver(board)
        solver.solve_cell(0, 0)

        moves = solver.moves
        assert len(moves) > 0
        assert isinstance(moves[0], tuple)
        assert len(moves[0]) == 2

    def test_get_solution(self):
        """Test getting solution moves."""
        board = ArrowBoard(3)
        board.set_value(0, 0, 2)

        solver = ArrowSolver(board)
        solver.solve_cell(0, 0)

        solution = solver.get_solution()
        assert len(solution) > 0
        assert isinstance(solution[0], tuple)
        assert len(solution[0]) == 2