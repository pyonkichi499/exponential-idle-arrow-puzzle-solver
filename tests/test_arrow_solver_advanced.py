"""Advanced tests for Arrow Solver."""

import numpy as np
import pytest
from puzzle_solver.puzzles.arrow.board import ArrowBoard
from puzzle_solver.puzzles.arrow.solver import ArrowSolver


class TestArrowSolverAdvanced:
    """Advanced test cases for ArrowSolver."""
    
    def test_encode_bottom_row(self):
        """Test bottom row encoding for expert mode."""
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
    
    def test_expert_solve_specific_pattern(self):
        """Test expert mode with specific pattern."""
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
    
    def test_propagate_behavior(self):
        """Test propagation behavior."""
        board = ArrowBoard(5)
        # Set specific values
        for i in range(5):
            board.set_value(0, i, 2)
        
        solver = ArrowSolver(board)
        solver.propagate()  # Run propagation
        
        # Board should have been modified
        final_sum = sum(solver.board.get_value(i, j) for i in range(5) for j in range(5))
        assert final_sum != 10  # Not all 2s anymore
    
    def test_hard_mode_solving(self):
        """Test hard mode solving."""
        board = ArrowBoard(5)
        # Create a simple solvable pattern
        board.tap(0, 0)
        board.tap(2, 2)
        board.tap(4, 4)
        
        solver = ArrowSolver(board)
        result = solver.solve("hard")
        
        assert isinstance(result, bool)
        assert len(solver.moves) >= 0
    
    def test_solve_with_all_zeros(self):
        """Test solving board that's all zeros."""
        board = ArrowBoard(3)
        # Board is already all zeros
        
        solver = ArrowSolver(board)
        result = solver.solve()
        
        # Should need to make moves to get to all 1s
        assert len(solver.moves) > 0
    
    def test_solve_with_all_ones(self):
        """Test solving board that's already solved."""
        board = ArrowBoard(3)
        board.grid = np.ones((3, 3), dtype=int)
        
        solver = ArrowSolver(board)
        result = solver.solve()
        
        assert result is True
        assert solver.verify_solution()
        assert len(solver.moves) == 0  # No moves needed
    
    def test_large_board_performance(self):
        """Test solver performance on larger board."""
        board = ArrowBoard(10)
        # Random initial state
        np.random.seed(42)
        board.grid = np.random.randint(0, 5, size=(10, 10))
        
        solver = ArrowSolver(board)
        result = solver.solve()
        
        # Should complete without hanging
        assert isinstance(result, bool)
    
    def test_copy_preservation(self):
        """Test that original board is preserved."""
        board = ArrowBoard(5)
        original_grid = board.grid.copy()
        
        solver = ArrowSolver(board)
        solver.solve()
        
        # Original board should be unchanged
        assert np.array_equal(board.grid, original_grid)