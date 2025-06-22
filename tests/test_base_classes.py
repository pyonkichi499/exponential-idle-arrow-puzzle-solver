"""Tests for base classes."""

import pytest
from unittest.mock import Mock, MagicMock
from puzzle_solver.core.base_board import BaseBoard
from puzzle_solver.core.base_solver import BaseSolver
from puzzle_solver.core.base_vision import BaseVision


class TestBaseBoard:
    """Test cases for BaseBoard abstract class."""
    
    def test_抽象クラスのインスタンス化不可(self):
        """BaseBoardを直接インスタンス化できないことをテスト"""
        with pytest.raises(TypeError):
            BaseBoard()
    
    def test_抽象メソッド定義(self):
        """抽象メソッドが定義されていることをテスト"""
        # Verify that BaseBoard has the required abstract methods
        assert hasattr(BaseBoard, 'is_solved')
        assert hasattr(BaseBoard, 'get_legal_moves')
        assert hasattr(BaseBoard, 'apply_move')
        assert hasattr(BaseBoard, 'copy')


class TestBaseSolver:
    """Test cases for BaseSolver abstract class."""
    
    def test_抽象ソルバークラスのインスタンス化不可(self):
        """BaseSolverを直接インスタンス化できないことをテスト"""
        with pytest.raises(TypeError):
            BaseSolver(None)
    
    def test_抽象ソルバーメソッド定義(self):
        """抽象メソッドが定義されていることをテスト"""
        # Verify that BaseSolver has the required abstract methods
        assert hasattr(BaseSolver, 'solve')
        assert hasattr(BaseSolver, 'get_solution')
    
    def test_ベースソルバードキュメント(self):
        """BaseSolverが適切なドキュメントを持つことをテスト"""
        # Just verify the class exists and has documentation
        assert BaseSolver.__doc__ is not None
        assert "Abstract base class for puzzle solvers" in BaseSolver.__doc__


class TestBaseVision:
    """Test cases for BaseVision abstract class."""
    
    def test_抽象ビジョンクラスのインスタンス化不可(self):
        """BaseVisionを直接インスタンス化できないことをテスト"""
        with pytest.raises(TypeError):
            BaseVision()
    
    def test_抽象ビジョンメソッド定義(self):
        """抽象メソッドが定義されていることをテスト"""
        # Verify that BaseVision has the required abstract methods
        assert hasattr(BaseVision, 'detect_puzzle')
        assert hasattr(BaseVision, 'capture_screen_region')