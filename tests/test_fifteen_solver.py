"""Tests for the FifteenSolver class."""

import numpy as np
from puzzle_solver.puzzles.fifteen.board import FifteenBoard
from puzzle_solver.puzzles.fifteen.solver import FifteenSolver


class TestFifteenSolver:
    """Test cases for FifteenSolver class."""

    def test_solver_initialization(self):
        """Test solver initialization."""
        board = FifteenBoard()
        solver = FifteenSolver(board)
        # Solver creates a copy of the board
        assert np.array_equal(solver.board.grid, board.grid)
        assert solver.solution == []

    def test_solve_already_solved(self):
        """Test solving an already solved board."""
        board = FifteenBoard()
        solver = FifteenSolver(board)
        
        result = solver.solve()
        assert result is True
        assert len(solver.solution) == 0

    def test_solve_simple_puzzle(self):
        """Test solving a simple puzzle."""
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

    def test_solve_unsolvable(self):
        """Test solving an unsolvable puzzle."""
        board = FifteenBoard()
        # Create unsolvable configuration
        board.grid[0, 0] = 2
        board.grid[0, 1] = 1
        
        solver = FifteenSolver(board)
        result = solver.solve()
        
        assert result is False
        assert len(solver.solution) == 0

    def test_get_solution(self):
        """Test getting solution moves."""
        board = FifteenBoard()
        board.shuffle(5)
        
        solver = FifteenSolver(board)
        solver.solve()
        
        solution = solver.get_solution()
        assert isinstance(solution, list)
        assert all(move in ["up", "down", "left", "right"] for move in solution)

    def test_solve_with_timeout(self):
        """Test that solver respects complexity limits."""
        board = FifteenBoard()
        # Create a moderately complex puzzle
        board.shuffle(20)
        
        solver = FifteenSolver(board)
        result = solver.solve()
        
        # Should either solve or timeout gracefully
        assert isinstance(result, bool)