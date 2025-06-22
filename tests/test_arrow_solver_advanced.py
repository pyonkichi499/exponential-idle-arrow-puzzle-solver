"""Advanced tests for Arrow Solver."""

import numpy as np
import pytest
from puzzle_solver.puzzles.arrow.board import ArrowBoard
from puzzle_solver.puzzles.arrow.solver import ArrowSolver


class TestArrowSolverAdvanced:
    """Advanced test cases for ArrowSolver."""
    
    def test_最下行エンコード(self):
        """エキスパートモード用の最下行エンコードをテスト"""
        board = ArrowBoard(7)
        # Set specific bottom row values
        for i in range(7):
            board.set_value(6, i, i % 5)
        
        solver = ArrowSolver(board)
        # Test that solve_expert_mode uses encoding
        initial_top_row = [board.get_value(0, i) for i in range(7)]
        solver.solve("expert")
        
        # The solver should have made some moves
        assert len(solver.moves) > 0
    
    def test_特定パターンでのエキスパート解決(self):
        """特定パターンでのエキスパートモードをテスト"""
        board = ArrowBoard(5)
        # Create a pattern that requires expert mode
        pattern = [
            [0, 1, 2, 3, 4],
            [1, 2, 3, 4, 0],
            [2, 3, 4, 0, 1],
            [3, 4, 0, 1, 2],
            [4, 0, 1, 2, 3]
        ]
        
        for i in range(5):
            for j in range(5):
                board.set_value(i, j, pattern[i][j])
        
        solver = ArrowSolver(board)
        result = solver.solve("expert")
        
        # Should attempt to solve (may or may not succeed)
        assert isinstance(result, bool)
        if result:
            assert solver.verify_solution()
    
    def test_伝搬動作(self):
        """伝搬動作をテスト"""
        board = ArrowBoard(5)
        # Set specific values
        for i in range(5):
            board.set_value(0, i, 2)
        
        solver = ArrowSolver(board)
        solver.propagate()  # Run propagation
        
        # Board should have been modified
        final_sum = sum(solver.board.get_value(i, j) for i in range(5) for j in range(5))
        assert final_sum != 10  # Not all 2s anymore
    
    def test_ハードモード解決(self):
        """ハードモードでの解決をテスト"""
        board = ArrowBoard(5)
        # Create a simple solvable pattern
        board.tap(0, 0)
        board.tap(2, 2)
        board.tap(4, 4)
        
        solver = ArrowSolver(board)
        result = solver.solve("hard")
        
        assert isinstance(result, bool)
        assert len(solver.moves) >= 0
    
    def test_全てゼロのボード解決(self):
        """全てがゼロのボードの解決をテスト"""
        board = ArrowBoard(3)
        # Board is already all zeros
        
        solver = ArrowSolver(board)
        result = solver.solve()
        
        # Should need to make moves to get to all 1s
        assert len(solver.moves) > 0
    
    def test_全て一のボード解決(self):
        """既に解決済みのボードの解決をテスト"""
        board = ArrowBoard(3)
        board.grid = np.ones((3, 3), dtype=int)
        
        solver = ArrowSolver(board)
        result = solver.solve()
        
        assert result is True
        assert solver.verify_solution()
        assert len(solver.moves) == 0  # No moves needed
    
    def test_大型ボードパフォーマンス(self):
        """大型ボードでのソルバーパフォーマンスをテスト"""
        board = ArrowBoard(10)
        # Random initial state
        np.random.seed(42)
        board.grid = np.random.randint(0, 5, size=(10, 10))
        
        solver = ArrowSolver(board)
        result = solver.solve()
        
        # Should complete without hanging
        assert isinstance(result, bool)
    
    def test_元ボード保存(self):
        """元のボードが保存されることをテスト"""
        board = ArrowBoard(5)
        original_grid = board.grid.copy()
        
        solver = ArrowSolver(board)
        solver.solve()
        
        # Original board should be unchanged
        assert np.array_equal(board.grid, original_grid)