"""Edge case tests for Arrow Board."""

import numpy as np
import pytest
from puzzle_solver.puzzles.arrow.board import ArrowBoard


class TestArrowBoardEdgeCases:
    """Edge case tests for ArrowBoard."""
    
    def test_minimum_board_size(self):
        """Test creating board with minimum size."""
        board = ArrowBoard(1)
        assert board.size == 1
        assert board.grid.shape == (1, 1)
    
    def test_from_string_various_formats(self):
        """Test from_string with various input formats."""
        # Comma separated
        board1 = ArrowBoard.from_string("1,2,3\n4,0,1\n2,3,4", size=3)
        assert board1.get_value(0, 0) == 1
        assert board1.get_value(1, 1) == 0
        
        # Space separated
        board2 = ArrowBoard.from_string("1 2 3\n4 0 1\n2 3 4", size=3)
        assert board2.get_value(0, 0) == 1
        assert board2.get_value(1, 1) == 0
        
        # Mixed separators
        board3 = ArrowBoard.from_string("1, 2, 3\n4  0  1\n2,3,4", size=3)
        assert board3.get_value(0, 0) == 1
        assert board3.get_value(1, 1) == 0
    
    def test_from_string_incomplete_data(self):
        """Test from_string with incomplete data."""
        # Less values than expected
        board = ArrowBoard.from_string("1,2\n3", size=3)
        assert board.get_value(0, 0) == 1
        assert board.get_value(0, 1) == 2
        assert board.get_value(0, 2) == 0  # Default value
        assert board.get_value(1, 0) == 3
        assert board.get_value(1, 1) == 0  # Default value
    
    def test_from_string_empty(self):
        """Test from_string with empty string."""
        board = ArrowBoard.from_string("", size=3)
        assert np.all(board.grid == 0)
    
    def test_to_string(self):
        """Test to_string method."""
        board = ArrowBoard(3)
        board.set_value(0, 0, 1)
        board.set_value(1, 1, 2)
        board.set_value(2, 2, 3)
        
        string_repr = board.to_string()
        assert "1" in string_repr
        assert "2" in string_repr
        assert "3" in string_repr
    
    def test_invalid_value_wrapping(self):
        """Test that invalid values wrap correctly."""
        board = ArrowBoard(3)
        
        # Set value > 4
        board.grid[0, 0] = 7
        board.tap(0, 0)  # Should wrap: 7 + 1 = 8 % 5 = 3
        assert board.get_value(0, 0) == 3
        
        # Set negative value (shouldn't happen but test anyway)
        board.grid[1, 1] = -1
        board.tap(1, 1)  # Should wrap: -1 + 1 = 0
        assert board.get_value(1, 1) == 0
    
    def test_boundary_taps(self):
        """Test tapping all boundary cells."""
        board = ArrowBoard(5)
        
        # Tap all corners
        corners = [(0, 0), (0, 4), (4, 0), (4, 4)]
        for r, c in corners:
            board.tap(r, c)
            assert board.get_value(r, c) == 1
        
        # Verify corner taps only affected 3 cells each
        assert board.get_value(0, 1) == 1  # Adjacent to (0,0)
        assert board.get_value(1, 0) == 1  # Adjacent to (0,0)
        assert board.get_value(1, 1) == 0  # Not adjacent to corners
    
    def test_symmetric_operations(self):
        """Test symmetric operations on board."""
        board = ArrowBoard(5)
        
        # Create symmetric pattern
        board.tap(2, 2)  # Center
        assert board.is_symmetric()
        
        # Add symmetric taps
        board.tap(0, 0)
        board.tap(0, 4)
        board.tap(4, 0)
        board.tap(4, 4)
        assert board.is_symmetric()
        
        # Break symmetry
        board.tap(0, 1)
        assert not board.is_symmetric()