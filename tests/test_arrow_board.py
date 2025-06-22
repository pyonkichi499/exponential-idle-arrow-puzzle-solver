"""Tests for the Arrow Board class."""

import numpy as np
from puzzle_solver.puzzles.arrow.board import ArrowBoard


class TestArrowBoard:
    """Test cases for ArrowBoard class."""

    def test_ボード初期化(self):
        """ボードの初期化をテスト"""
        board = ArrowBoard(7)
        assert board.size == 7
        assert board.grid.shape == (7, 7)
        assert np.all(board.grid == 0)

    def test_中央セルタップ(self):
        """中央セルをタップするテスト"""
        board = ArrowBoard(5)
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

    def test_角セルタップ(self):
        """角セルをタップするテスト"""
        board = ArrowBoard(5)
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

    def test_値の巡回(self):
        """値が5で巡回することをテスト"""
        board = ArrowBoard(3)

        # Tap same cell 5 times
        for _ in range(5):
            board.tap(1, 1)

        # Should wrap back to 0
        assert board.get_value(1, 1) == 0

    def test_完成判定(self):
        """パズルが完成したかの判定をテスト"""
        board = ArrowBoard(3)

        # Initially not solved
        assert not board.is_solved()

        # Set all to 1
        board.grid = np.ones((3, 3), dtype=int)
        assert board.is_solved()

        # Change one cell
        board.set_value(1, 1, 2)
        assert not board.is_solved()

    def test_対称性判定(self):
        """対称性の判定をテスト"""
        board = ArrowBoard(3)

        # Create symmetric pattern
        board.set_value(0, 0, 1)
        board.set_value(0, 2, 1)
        board.set_value(1, 1, 2)

        assert board.is_symmetric()

        # Break symmetry
        board.set_value(0, 0, 3)
        assert not board.is_symmetric()

    def test_ボードコピー(self):
        """ボードのコピー機能をテスト"""
        board1 = ArrowBoard(3)
        board1.set_value(1, 1, 3)

        board2 = board1.copy()

        # Should have same values
        assert np.array_equal(board1.grid, board2.grid)

        # But be different objects
        board2.set_value(0, 0, 2)
        assert board1.get_value(0, 0) == 0
        assert board2.get_value(0, 0) == 2

    def test_有効なmove取得(self):
        """有効なmoveの取得をテスト"""
        board = ArrowBoard(3)
        moves = board.get_legal_moves()
        
        # Should have all positions as valid moves
        assert len(moves) == 9
        assert (0, 0) in moves
        assert (1, 1) in moves
        assert (2, 2) in moves

    def test_move適用(self):
        """moveの適用をテスト"""
        board = ArrowBoard(3)
        board.apply_move((1, 1))
        
        # Should be same as tap
        assert board.get_value(1, 1) == 1
        assert board.get_value(0, 1) == 1
        assert board.get_value(2, 1) == 1
        assert board.get_value(1, 0) == 1
        assert board.get_value(1, 2) == 1