"""Tests for the FifteenBoard class."""

import numpy as np
from puzzle_solver.puzzles.fifteen.board import FifteenBoard


class TestFifteenBoard:
    """Test cases for FifteenBoard class."""

    def test_ボード初期化(self):
        """ボードの初期化をテスト"""
        board = FifteenBoard()
        assert board.size == 4
        assert board.grid.shape == (4, 4)
        
        # Check initial solved state
        expected = np.array([
            [1, 2, 3, 4],
            [5, 6, 7, 8],
            [9, 10, 11, 12],
            [13, 14, 15, 0]
        ])
        assert np.array_equal(board.grid, expected)
        assert board.empty_pos == (3, 3)

    def test_空きマスの移動(self):
        """空きマスの移動をテスト"""
        board = FifteenBoard()
        
        # Move up
        board.apply_move("up")
        assert board.empty_pos == (2, 3)
        assert board.grid[3, 3] == 12
        assert board.grid[2, 3] == 0
        
        # Move left
        board.apply_move("left")
        assert board.empty_pos == (2, 2)
        assert board.grid[2, 3] == 11
        assert board.grid[2, 2] == 0

    def test_無効なmove(self):
        """無効なmoveがブロックされることをテスト"""
        board = FifteenBoard()
        
        # Can't move right or down from bottom-right corner
        old_pos = board.empty_pos
        board.apply_move("right")
        assert board.empty_pos == old_pos
        
        board.apply_move("down")
        assert board.empty_pos == old_pos

    def test_完成判定(self):
        """is_solvedメソッドをテスト"""
        board = FifteenBoard()
        assert board.is_solved()
        
        # Make a move
        board.apply_move("up")
        assert not board.is_solved()
        
        # Move back
        board.apply_move("down")
        assert board.is_solved()

    def test_有効なmove取得(self):
        """get_legal_movesメソッドをテスト"""
        board = FifteenBoard()
        
        # From bottom-right corner
        moves = board.get_legal_moves()
        assert set(moves) == {"up", "left"}
        
        # Move to center
        board.empty_pos = (1, 1)
        board.grid[3, 3] = board.grid[1, 1]
        board.grid[1, 1] = 0
        
        moves = board.get_legal_moves()
        assert set(moves) == {"up", "down", "left", "right"}

    def test_move適用(self):
        """apply_moveメソッドをテスト"""
        board = FifteenBoard()
        board.apply_move("up")
        
        assert board.empty_pos == (2, 3)
        assert board.grid[3, 3] == 12
        assert board.grid[2, 3] == 0

    def test_シャッフル(self):
        """shuffleメソッドをテスト"""
        board = FifteenBoard()
        initial_state = board.grid.copy()
        board.shuffle(50)  # More shuffles to ensure movement
        
        # Board state should have changed
        assert not np.array_equal(board.grid, initial_state) or board.empty_pos != (3, 3)

    def test_解決可能性判定(self):
        """is_solvableメソッドをテスト"""
        board = FifteenBoard()
        assert board.is_solvable()
        
        # Create unsolvable configuration by swapping two tiles
        board.grid[0, 0] = 2
        board.grid[0, 1] = 1
        assert not board.is_solvable()

    def test_マンハッタン距離(self):
        """マンハッタン距離の計算をテスト"""
        board = FifteenBoard()
        assert board.manhattan_distance() == 0
        
        # Move a tile
        board.apply_move("up")
        assert board.manhattan_distance() > 0

    def test_ボードコピー(self):
        """ボードのコピーをテスト"""
        board1 = FifteenBoard()
        board1.apply_move("up")
        
        board2 = board1.copy()
        
        # Should have same state
        assert np.array_equal(board1.grid, board2.grid)
        assert board1.empty_pos == board2.empty_pos
        
        # But be different objects
        board2.apply_move("left")
        assert board1.empty_pos != board2.empty_pos