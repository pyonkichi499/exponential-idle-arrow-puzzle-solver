"""Tests for the Board class."""

import numpy as np
from arrow_puzzle_solver.board import Board


class TestBoard:
    """Test cases for Board class."""

    def test_board_initialization(self):
        """Test board initialization."""
        board = Board(7)
        assert board.size == 7
        assert board.grid.shape == (7, 7)
        assert np.all(board.grid == 0)

    def test_tap_center(self):
        """Test tapping a center cell."""
        board = Board(5)
        board.tap(2, 2)

        # Center should be 1
        assert board.get_value(2, 2) == 1

        # Adjacent cells should be 1
        assert board.get_value(1, 2) == 1  # top
        assert board.get_value(3, 2) == 1  # bottom
        assert board.get_value(2, 1) == 1  # left
        assert board.get_value(2, 3) == 1  # right

        # Diagonal cells should be 0
        assert board.get_value(1, 1) == 0
        assert board.get_value(1, 3) == 0
        assert board.get_value(3, 1) == 0
        assert board.get_value(3, 3) == 0

    def test_tap_corner(self):
        """Test tapping a corner cell."""
        board = Board(5)
        board.tap(0, 0)

        # Corner should be 1
        assert board.get_value(0, 0) == 1

        # Adjacent cells should be 1
        assert board.get_value(0, 1) == 1  # right
        assert board.get_value(1, 0) == 1  # bottom

        # Other cells should be 0
        assert board.get_value(1, 1) == 0
        assert board.get_value(2, 0) == 0
        assert board.get_value(0, 2) == 0

    def test_value_wrapping(self):
        """Test that values wrap around at 5."""
        board = Board(3)

        # Tap same cell 5 times
        for _ in range(5):
            board.tap(1, 1)

        # Should wrap back to 0
        assert board.get_value(1, 1) == 0

    def test_is_solved(self):
        """Test is_solved method."""
        board = Board(3)

        # Initially not solved
        assert not board.is_solved()

        # Set all to 1
        board.grid = np.ones((3, 3), dtype=int)
        assert board.is_solved()

        # Change one cell
        board.set_value(1, 1, 2)
        assert not board.is_solved()

    def test_is_symmetric(self):
        """Test is_symmetric method."""
        board = Board(3)

        # Create symmetric pattern
        board.set_value(0, 0, 1)
        board.set_value(0, 2, 1)
        board.set_value(1, 1, 2)

        assert board.is_symmetric()

        # Break symmetry
        board.set_value(0, 0, 3)
        assert not board.is_symmetric()

    def test_copy(self):
        """Test board copying."""
        board1 = Board(3)
        board1.set_value(1, 1, 3)

        board2 = board1.copy()

        # Should have same values
        assert np.array_equal(board1.grid, board2.grid)

        # But be different objects
        board2.set_value(0, 0, 2)
        assert board1.get_value(0, 0) == 0
        assert board2.get_value(0, 0) == 2
