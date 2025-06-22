"""Edge case tests for Arrow Board."""

import numpy as np
import pytest
from puzzle_solver.puzzles.arrow.board import ArrowBoard


class TestArrowBoardEdgeCases:
    """Edge case tests for ArrowBoard."""
    
    def test_最小ボードサイズ(self):
        """最小サイズのボード作成をテスト"""
        board = ArrowBoard(1)
        assert board.size == 1
        assert board.grid.shape == (1, 1)
    
    def test_文字列からの様々な形式(self):
        """文字列からの様々な入力形式をテスト"""
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
    
    def test_不完全データからの文字列(self):
        """不完全なデータからの文字列変換をテスト"""
        # Less values than expected
        board = ArrowBoard.from_string("1,2\n3", size=3)
        assert board.get_value(0, 0) == 1
        assert board.get_value(0, 1) == 2
        assert board.get_value(0, 2) == 0  # Default value
        assert board.get_value(1, 0) == 3
        assert board.get_value(1, 1) == 0  # Default value
    
    def test_空文字列からの変換(self):
        """空文字列からの変換をテスト"""
        board = ArrowBoard.from_string("", size=3)
        assert np.all(board.grid == 0)
    
    def test_文字列変換(self):
        """to_stringメソッドをテスト"""
        board = ArrowBoard(3)
        board.set_value(0, 0, 1)
        board.set_value(1, 1, 2)
        board.set_value(2, 2, 3)
        
        string_repr = board.to_string()
        assert "1" in string_repr
        assert "2" in string_repr
        assert "3" in string_repr
    
    def test_無効値の巡回(self):
        """無効値が正しく巡回することをテスト"""
        board = ArrowBoard(3)
        
        # Set value > 4
        board.grid[0, 0] = 7
        board.tap(0, 0)  # Should wrap: 7 + 1 = 8 % 5 = 3
        assert board.get_value(0, 0) == 3
        
        # Set negative value (shouldn't happen but test anyway)
        board.grid[1, 1] = -1
        board.tap(1, 1)  # Should wrap: -1 + 1 = 0
        assert board.get_value(1, 1) == 0
    
    def test_境界セルタップ(self):
        """すべての境界セルのタップをテスト"""
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
    
    def test_対称操作(self):
        """ボード上の対称操作をテスト"""
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