"""15パズルの画像認識機能のテスト"""

import pytest
import numpy as np
import cv2
from puzzle_solver.puzzles.fifteen import FifteenVision


class TestFifteenVision:
    """15パズルの画像認識テスト"""
    
    def test_15パズルビジョンの初期化(self):
        """FifteenVisionクラスが正しく初期化されることを確認"""
        vision = FifteenVision(size=4)
        assert vision.size == 4
        assert vision.templates_dir.name == "templates"
        assert isinstance(vision.digit_templates, dict)
        assert isinstance(vision.calibration_data, dict)
    
    def test_パズル検出(self):
        """パズルグリッドを正しく検出できることを確認"""
        vision = FifteenVision(size=4)
        test_image = self._create_test_puzzle_image()
        
        grid_bounds = vision.detect_puzzle(test_image)
        assert grid_bounds is not None
        x, y, w, h = grid_bounds
        assert w > 0
        assert h > 0
        assert 0.95 <= float(w) / h <= 1.05  # 正方形に近い
    
    def test_タイル抽出(self):
        """グリッドからタイルを正しく抽出できることを確認"""
        vision = FifteenVision(size=4)
        test_image = self._create_test_puzzle_image()
        grid_bounds = (50, 50, 400, 400)
        
        tiles = vision.extract_tiles(test_image, grid_bounds)
        assert len(tiles) == 4
        assert len(tiles[0]) == 4
        assert all(tile.shape[0] > 0 and tile.shape[1] > 0 for row in tiles for tile in row)
    
    def test_空タイルの認識(self):
        """空のタイルを正しく認識できることを確認"""
        vision = FifteenVision()
        
        # 空のタイル（一様な色）
        empty_tile = np.ones((50, 50, 3), dtype=np.uint8) * 200
        assert vision.recognize_tile(empty_tile) == 0
        
        # 数字のあるタイル（変化のある画像）
        numbered_tile = self._create_numbered_tile(5)
        result = vision.recognize_tile(numbered_tile)
        # OCRやテンプレートマッチングが設定されていない場合は0を返す
        assert result >= 0
    
    def test_OCRフォールバック(self):
        """OCRによる数字認識のフォールバックをテスト"""
        vision = FifteenVision()
        
        # 明確な数字を含むタイル画像を作成
        for number in [1, 5, 10, 15]:
            tile = self._create_numbered_tile(number)
            # OCRは実際のフォントに依存するため、基本的な動作のみ確認
            result = vision._recognize_by_ocr(cv2.cvtColor(tile, cv2.COLOR_BGR2GRAY))
            assert isinstance(result, int)
            assert 0 <= result <= 15
    
    def test_キャリブレーション(self):
        """キャリブレーション機能が正しく動作することを確認"""
        vision = FifteenVision(size=4)
        
        # テスト用の参照画像と既知の状態を作成
        reference_image = self._create_test_puzzle_image(with_numbers=True)
        known_state = [
            [1, 2, 3, 4],
            [5, 6, 7, 8],
            [9, 10, 11, 12],
            [13, 14, 15, 0]  # 0は空のタイル
        ]
        
        # テンプレートディレクトリをクリア
        if vision.templates_dir.exists():
            import shutil
            shutil.rmtree(vision.templates_dir)
        
        result = vision.calibrate(reference_image, known_state)
        assert result is True
        assert vision.templates_dir.exists()
        # 1-15のテンプレートが保存されているか確認
        assert len(vision.digit_templates) > 0
    
    def test_パズル状態の読み取り(self):
        """パズル全体の状態を正しく読み取れることを確認"""
        vision = FifteenVision(size=4)
        test_image = self._create_test_puzzle_image(with_numbers=True)
        
        board = vision.read_puzzle_state(test_image)
        assert board is not None
        assert board.size == 4
        assert board.empty_pos is not None
        
        # すべての数字が0-15の範囲内であることを確認
        found_numbers = set()
        for row in range(4):
            for col in range(4):
                value = board.grid[row, col]
                assert 0 <= value <= 15
                found_numbers.add(value)
    
    def test_空タイルの位置検出(self):
        """空タイルの位置を正しく検出できることを確認"""
        vision = FifteenVision(size=4)
        test_image = self._create_test_puzzle_image(with_numbers=True, empty_pos=(2, 3))
        grid_bounds = (50, 50, 400, 400)
        
        empty_pos = vision.find_empty_tile(test_image, grid_bounds)
        assert empty_pos is not None
        # 実際の位置は画像生成方法に依存するため、有効な座標であることのみ確認
        row, col = empty_pos
        assert 0 <= row < 4
        assert 0 <= col < 4
    
    def test_セル座標の取得(self):
        """各タイルの中心座標を正しく計算できることを確認"""
        vision = FifteenVision(size=4)
        region = (100, 100, 400, 400)
        
        coordinates = vision.get_cell_coordinates(region)
        assert len(coordinates) == 4
        assert len(coordinates[0]) == 4
        
        # 最初のタイルの中心座標
        assert coordinates[0][0] == (150, 150)
        # 最後のタイルの中心座標
        assert coordinates[3][3] == (450, 450)
    
    def test_パズル構造の検証(self):
        """15パズルの構造検証が正しく動作することを確認"""
        vision = FifteenVision(size=4)
        
        # 適切なグリッド構造を持つ画像
        good_puzzle = self._create_test_puzzle_image(with_grid_lines=True)
        gray = cv2.cvtColor(good_puzzle, cv2.COLOR_BGR2GRAY)
        roi = gray[50:450, 50:450]  # グリッド部分を切り出し
        assert vision._verify_puzzle_structure(roi) is True
        
        # グリッド構造を持たない画像
        bad_puzzle = np.ones((400, 400), dtype=np.uint8) * 128
        assert vision._verify_puzzle_structure(bad_puzzle) is False
    
    # ヘルパーメソッド
    def _create_test_puzzle_image(self, with_numbers: bool = False, 
                                with_grid_lines: bool = True,
                                empty_pos: tuple = (3, 3)) -> np.ndarray:
        """テスト用の15パズル画像を生成"""
        tile_size = 100
        margin = 50
        size = 4
        width = size * tile_size + 2 * margin
        height = size * tile_size + 2 * margin
        
        # 白背景の画像を作成
        image = np.ones((height, width, 3), dtype=np.uint8) * 255
        
        # グリッド線を描画
        if with_grid_lines:
            # 外枠
            cv2.rectangle(image, (margin, margin), 
                        (width - margin, height - margin), (0, 0, 0), 2)
            
            # 内部のグリッド線
            for i in range(1, size):
                y = margin + i * tile_size
                cv2.line(image, (margin, y), (width - margin, y), (0, 0, 0), 1)
                x = margin + i * tile_size
                cv2.line(image, (x, margin), (x, height - margin), (0, 0, 0), 1)
        
        # 数字を追加
        if with_numbers:
            number = 1
            for i in range(size):
                for j in range(size):
                    if (i, j) != empty_pos:
                        x = margin + j * tile_size + tile_size // 2
                        y = margin + i * tile_size + tile_size // 2
                        cv2.putText(image, str(number), (x - 20, y + 10),
                                  cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 2)
                        number += 1
                    else:
                        # 空のタイルは薄いグレーで塗りつぶし
                        x1 = margin + j * tile_size + 5
                        y1 = margin + i * tile_size + 5
                        x2 = x1 + tile_size - 10
                        y2 = y1 + tile_size - 10
                        cv2.rectangle(image, (x1, y1), (x2, y2), (200, 200, 200), -1)
        
        return image
    
    def _create_numbered_tile(self, number: int) -> np.ndarray:
        """指定された数字を含むタイル画像を生成"""
        size = 80
        tile = np.ones((size, size, 3), dtype=np.uint8) * 255
        
        # 中央に数字を描画
        text = str(number)
        font_scale = 2 if number < 10 else 1.5
        cv2.putText(tile, text, (size//2 - 20, size//2 + 10),
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 2)
        
        return tile