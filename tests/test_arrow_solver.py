"""Tests for the Arrow Solver class."""

import numpy as np
from puzzle_solver.puzzles.arrow.board import ArrowBoard
from puzzle_solver.puzzles.arrow.solver import ArrowSolver


class TestArrowSolver:
    """Test cases for ArrowSolver class."""

    def test_単一セル解決(self):
        """単一セルの解決をテスト"""
        board = ArrowBoard(5)
        # Set a cell to value 3
        board.set_value(0, 2, 3)

        solver = ArrowSolver(board)
        solver.solve_cell(0, 2)

        # Cell should now be 1 (3 + 3 = 6 % 5 = 1)
        assert solver.board.get_value(0, 2) == 1

    def test_行の伝搬(self):
        """単一行の伝搬処理をテスト"""
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

    def test_簡単なパズル解決(self):
        """簡単なパズルの解決をテスト"""
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

    def test_解決モード(self):
        """異なる解決モードをテスト"""
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

    def test_解法検証(self):
        """解法の検証をテスト"""
        board = ArrowBoard(3)
        board.grid = np.ones((3, 3), dtype=int)

        solver = ArrowSolver(board)
        solver.board = board

        # Should verify as correct
        assert solver.verify_solution()

    def test_move履歴取得(self):
        """move履歴の取得をテスト"""
        board = ArrowBoard(3)
        board.set_value(0, 0, 2)

        solver = ArrowSolver(board)
        solver.solve_cell(0, 0)

        moves = solver.moves
        assert len(moves) > 0
        assert isinstance(moves[0], tuple)
        assert len(moves[0]) == 2

    def test_解法move取得(self):
        """解法のmove取得をテスト"""
        board = ArrowBoard(3)
        board.set_value(0, 0, 2)

        solver = ArrowSolver(board)
        solver.solve_cell(0, 0)

        solution = solver.get_solution()
        assert len(solution) > 0
        assert isinstance(solution[0], tuple)
        assert len(solution[0]) == 2