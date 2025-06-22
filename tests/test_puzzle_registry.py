"""Tests for puzzle registry system."""

import pytest
from puzzle_solver.puzzles import (
    get_puzzle_board, 
    get_puzzle_solver,
    get_puzzle_vision,
    _PUZZLE_REGISTRY
)
from puzzle_solver.core.base_board import BaseBoard
from puzzle_solver.core.base_solver import BaseSolver
from puzzle_solver.core.base_vision import BaseVision


class MockBoard(BaseBoard):
    """Mock board for testing."""
    def is_solved(self):
        return False
    def get_legal_moves(self):
        return ["move1", "move2"]
    def apply_move(self, move):
        pass
    def copy(self):
        return MockBoard()


class MockSolver(BaseSolver):
    """Mock solver for testing."""
    def solve(self, **kwargs):
        return True
    def get_solution(self):
        return ["solution"]


class MockVision(BaseVision):
    """Mock vision for testing."""
    def detect_puzzle(self, image):
        return {"grid": [[0]]}
    def capture_screen_region(self, region):
        return None


class TestPuzzleRegistry:
    """Test cases for puzzle registry system."""
    
    def test_レジストリにパズルが登録済み(self):
        """レジストリに事前登録されたパズルがあることをテスト"""
        assert "arrow" in _PUZZLE_REGISTRY
        assert "fifteen" in _PUZZLE_REGISTRY
        assert _PUZZLE_REGISTRY["arrow"][0] is not None  # board class
        assert _PUZZLE_REGISTRY["arrow"][1] is not None  # solver class
    
    def test_パズルボード取得(self):
        """パズルボードクラスの取得をテスト"""
        # Arrow puzzle should be pre-registered
        board_class = get_puzzle_board("arrow")
        assert board_class is not None
        assert issubclass(board_class, BaseBoard)
    
    def test_パズルソルバー取得(self):
        """パズルソルバークラスの取得をテスト"""
        solver_class = get_puzzle_solver("arrow")
        assert solver_class is not None
        assert issubclass(solver_class, BaseSolver)
    
    def test_パズルビジョン取得(self):
        """パズルビジョンクラスの取得をテスト"""
        vision = get_puzzle_vision("arrow")
        # Vision might be None if opencv is not installed
        assert vision is None or isinstance(vision, BaseVision)
    
    def test_未知パズルタイプ(self):
        """未知のパズルタイプでValueErrorが発生することをテスト"""
        with pytest.raises(ValueError):
            get_puzzle_board("unknown_puzzle")
        
        with pytest.raises(ValueError):
            get_puzzle_solver("unknown_puzzle")
        
        with pytest.raises(ValueError):
            get_puzzle_vision("unknown_puzzle")
    
    def test_利用可能パズル一覧(self):
        """利用可能なパズルの一覧取得をテスト"""
        puzzles = list(_PUZZLE_REGISTRY.keys())
        assert isinstance(puzzles, list)
        assert "arrow" in puzzles
        assert "fifteen" in puzzles