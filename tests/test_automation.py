"""自動化モジュールのテスト"""

import pytest
from unittest.mock import Mock, patch
from puzzle_solver.automation import PuzzleAutomator


class TestPuzzleAutomator:
    """PuzzleAutomatorクラスのテスト"""

    @patch('puzzle_solver.automation.get_puzzle_vision')
    def test_初期化(self, mock_get_vision):
        """PuzzleAutomatorの初期化が正しく動作することをテスト"""
        # ビジョンクラスのモック
        mock_vision_class = Mock()
        mock_vision_instance = Mock()
        mock_vision_class.return_value = mock_vision_instance
        mock_get_vision.return_value = mock_vision_class
        
        automator = PuzzleAutomator('arrow')
        
        assert automator.puzzle_type == 'arrow'
        assert automator.click_delay == 0.1
        assert automator.move_duration == 0.1
        assert automator.detector == mock_vision_instance
        mock_get_vision.assert_called_once_with('arrow')

    @patch('puzzle_solver.automation.pyautogui.click')
    @patch('puzzle_solver.automation.pyautogui.moveTo')
    @patch('puzzle_solver.automation.time.sleep')
    @patch('puzzle_solver.automation.get_puzzle_vision')
    def test_矢印パズルmove実行(self, mock_get_vision, mock_sleep, mock_moveTo, mock_click):
        """矢印パズルのmove実行が正しく動作することをテスト"""
        # ビジョンクラスのモック
        mock_vision_class = Mock()
        mock_vision_instance = Mock()
        mock_vision_class.return_value = mock_vision_instance
        mock_get_vision.return_value = mock_vision_class
        
        automator = PuzzleAutomator('arrow')
        
        # テスト用のmoveと座標
        moves = [(0, 0), (1, 1), (2, 2)]
        cell_coordinates = [
            [(100, 100), (200, 100), (300, 100)],
            [(100, 200), (200, 200), (300, 200)],
            [(100, 300), (200, 300), (300, 300)]
        ]
        
        automator.execute_arrow_moves(moves, cell_coordinates)
        
        # 正しい順序でクリックされたか確認
        assert mock_moveTo.call_count == 3
        assert mock_click.call_count == 3
        assert mock_sleep.call_count == 3
        
        # 各moveの座標が正しいか確認
        mock_moveTo.assert_any_call(100, 100, duration=0.1)
        mock_moveTo.assert_any_call(200, 200, duration=0.1)
        mock_moveTo.assert_any_call(300, 300, duration=0.1)

    @patch('puzzle_solver.automation.pyautogui.click')
    @patch('puzzle_solver.automation.pyautogui.moveTo')
    @patch('puzzle_solver.automation.time.sleep')
    @patch('puzzle_solver.automation.get_puzzle_vision')
    def test_15パズルmove実行(self, mock_get_vision, mock_sleep, mock_moveTo, mock_click):
        """15パズルのmove実行が正しく動作することをテスト"""
        # ビジョンクラスのモック
        mock_vision_class = Mock()
        mock_vision_instance = Mock()
        mock_vision_class.return_value = mock_vision_instance
        mock_get_vision.return_value = mock_vision_class
        
        automator = PuzzleAutomator('fifteen')
        
        # テスト用のmoveと座標
        moves = ['up', 'left']
        tile_coordinates = [
            [(0, 0), (50, 0), (100, 0), (150, 0)],
            [(0, 50), (50, 50), (100, 50), (150, 50)],
            [(0, 100), (50, 100), (100, 100), (150, 100)],
            [(0, 150), (50, 150), (100, 150), (150, 150)]
        ]
        empty_pos = (1, 1)  # 空きマスの位置
        
        automator.execute_fifteen_moves(moves, tile_coordinates, empty_pos)
        
        # 正しい順序でクリックされたか確認
        assert mock_moveTo.call_count == 2
        assert mock_click.call_count == 2
        assert mock_sleep.call_count == 2

    @patch('puzzle_solver.automation.get_puzzle_vision')
    def test_カスタムクリック遅延(self, mock_get_vision):
        """カスタムクリック遅延が設定できることをテスト"""
        # ビジョンクラスのモック
        mock_vision_class = Mock()
        mock_vision_instance = Mock()
        mock_vision_class.return_value = mock_vision_instance
        mock_get_vision.return_value = mock_vision_class
        
        automator = PuzzleAutomator('arrow', click_delay=0.5, move_duration=0.3)
        
        assert automator.click_delay == 0.5
        assert automator.move_duration == 0.3

    @patch('puzzle_solver.automation.get_puzzle_vision')
    def test_異なるパズルタイプ(self, mock_get_vision):
        """異なるパズルタイプで初期化できることをテスト"""
        # ビジョンクラスのモック
        mock_vision_class = Mock()
        mock_vision_instance = Mock()
        mock_vision_class.return_value = mock_vision_instance
        mock_get_vision.return_value = mock_vision_class
        
        # 矢印パズル
        automator_arrow = PuzzleAutomator('arrow')
        assert automator_arrow.puzzle_type == 'arrow'
        
        # 15パズル
        automator_fifteen = PuzzleAutomator('fifteen')
        assert automator_fifteen.puzzle_type == 'fifteen'

    @patch('puzzle_solver.automation.pyautogui.press')
    @patch('puzzle_solver.automation.time.sleep')
    @patch('puzzle_solver.automation.get_puzzle_vision')
    def test_キーボード操作(self, mock_get_vision, mock_sleep, mock_press):
        """キーボード操作が正しく動作することをテスト"""
        # ビジョンクラスのモック
        mock_vision_class = Mock()
        mock_vision_instance = Mock()
        mock_vision_class.return_value = mock_vision_instance
        mock_get_vision.return_value = mock_vision_class
        
        automator = PuzzleAutomator('arrow')
        
        # execute_keyboard_movesメソッドがある場合のテスト
        if hasattr(automator, 'execute_keyboard_moves'):
            moves = ['up', 'down', 'left', 'right']
            automator.execute_keyboard_moves(moves)
            
            assert mock_press.call_count == 4
            mock_press.assert_any_call('up')
            mock_press.assert_any_call('down')
            mock_press.assert_any_call('left')
            mock_press.assert_any_call('right')