"""Tests for base classes."""

import pytest
from unittest.mock import Mock, MagicMock
from puzzle_solver.core.base_board import BaseBoard
from puzzle_solver.core.base_solver import BaseSolver
from puzzle_solver.core.base_vision import BaseVision


class TestBaseBoard:
    """Test cases for BaseBoard abstract class."""
    
    def test_cannot_instantiate_abstract_class(self):
        """Test that BaseBoard cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseBoard()
    
    def test_abstract_methods(self):
        """Test that abstract methods are defined."""
        # Verify that BaseBoard has the required abstract methods
        assert hasattr(BaseBoard, 'is_solved')
        assert hasattr(BaseBoard, 'get_legal_moves')
        assert hasattr(BaseBoard, 'apply_move')
        assert hasattr(BaseBoard, 'copy')


class TestBaseSolver:
    """Test cases for BaseSolver abstract class."""
    
    def test_cannot_instantiate_abstract_class(self):
        """Test that BaseSolver cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseSolver(None)
    
    def test_abstract_methods(self):
        """Test that abstract methods are defined."""
        # Verify that BaseSolver has the required abstract methods
        assert hasattr(BaseSolver, 'solve')
        assert hasattr(BaseSolver, 'get_solution')
    
    def test_base_solver_docstring(self):
        """Test BaseSolver has proper documentation."""
        # Just verify the class exists and has documentation
        assert BaseSolver.__doc__ is not None
        assert "Abstract base class for puzzle solvers" in BaseSolver.__doc__


class TestBaseVision:
    """Test cases for BaseVision abstract class."""
    
    def test_cannot_instantiate_abstract_class(self):
        """Test that BaseVision cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseVision()
    
    def test_abstract_methods(self):
        """Test that abstract methods are defined."""
        # Verify that BaseVision has the required abstract methods
        assert hasattr(BaseVision, 'detect_puzzle')
        assert hasattr(BaseVision, 'capture_screen_region')