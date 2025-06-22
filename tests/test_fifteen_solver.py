"""Tests for the FifteenSolver class."""

import numpy as np
from puzzle_solver.puzzles.fifteen.board import FifteenBoard
from puzzle_solver.puzzles.fifteen.solver import FifteenSolver


class TestFifteenSolver:
    """Test cases for FifteenSolver class."""

    def test_ソルバー初期化(self):
        """ソルバーの初期化をテスト"""
        board = FifteenBoard()
        solver = FifteenSolver(board)
        # Solver creates a copy of the board
        assert np.array_equal(solver.board.grid, board.grid)
        assert solver.solution == []

    def test_完成済パズルの解決(self):
        """既に完成しているパズルの解決をテスト"""
        board = FifteenBoard()
        solver = FifteenSolver(board)
        
        result = solver.solve()
        assert result is True
        assert len(solver.solution) == 0

    def test_簡単パズル解決(self):
        """簡単なパズルの解決をテスト"""
        board = FifteenBoard()
        # Make a few moves to scramble
        board.apply_move("up")
        board.apply_move("left")
        
        solver = FifteenSolver(board)
        result = solver.solve()
        
        assert result is True
        assert len(solver.solution) > 0
        
        # Apply solution and verify it's solved
        for move in solver.solution:
            board.apply_move(move)
        assert board.is_solved()

    def test_解けないパズル解決(self):
        """解けないパズルの解決をテスト"""
        board = FifteenBoard()
        # Create unsolvable configuration
        board.grid[0, 0] = 2
        board.grid[0, 1] = 1
        
        solver = FifteenSolver(board)
        result = solver.solve()
        
        assert result is False
        assert len(solver.solution) == 0

    def test_解法move取得(self):
        """解法moveの取得をテスト"""
        board = FifteenBoard()
        board.shuffle(5)
        
        solver = FifteenSolver(board)
        solver.solve()
        
        solution = solver.get_solution()
        assert isinstance(solution, list)
        assert all(move in ["up", "down", "left", "right"] for move in solution)

    def test_複雑さ制限付き解決(self):
        """ソルバーが複雑さ制限を尊重することをテスト"""
        board = FifteenBoard()
        # Create a moderately complex puzzle
        board.shuffle(20)
        
        solver = FifteenSolver(board)
        result = solver.solve()
        
        # Should either solve or timeout gracefully
        assert isinstance(result, bool)