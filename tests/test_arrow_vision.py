"""矢印パズルの画像認識機能のテスト"""

import pytest
import numpy as np
import cv2
from puzzle_solver.puzzles.arrow import ArrowVision


class TestArrowVision:
    """矢印パズルの画像認識テスト"""
    
    def test_矢印パズルビジョンの初期化(self):
        """ArrowVisionクラスが正しく初期化されることを確認"""
        vision = ArrowVision(grid_size=7)
        assert vision.grid_size == 7
        assert vision.templates_dir.name == "templates"
        assert isinstance(vision.cell_templates, dict)
        assert isinstance(vision.calibration_data, dict)
    
    def test_空のグリッド検出(self):
        """グリッド検出メソッドが正しく動作することを確認"""
        vision = ArrowVision(grid_size=5)
        # 5x5のテスト画像を作成
        test_image = self._create_test_grid_image(5, 5)
        
        grid_bounds = vision.detect_puzzle(test_image)
        assert grid_bounds is not None
        x, y, w, h = grid_bounds
        assert w > 0
        assert h > 0
        assert abs(w - h) < 10  # ほぼ正方形
    
    def test_セル抽出(self):
        """グリッドからセルを正しく抽出できることを確認"""
        vision = ArrowVision(grid_size=3)
        test_image = self._create_test_grid_image(3, 3)
        grid_bounds = (50, 50, 300, 300)
        
        cells = vision.extract_cells(test_image, grid_bounds)
        assert len(cells) == 3
        assert len(cells[0]) == 3
        assert all(cell.shape[0] > 0 and cell.shape[1] > 0 for row in cells for cell in row)
    
    def test_数字認識の特徴ベース(self):
        """特徴ベースの数字認識が正しく動作することを確認"""
        vision = ArrowVision()
        
        # 異なる密度のテストセル画像を作成
        for digit, density in [(0, 0.05), (1, 0.25), (2, 0.45), (3, 0.65), (4, 0.85)]:
            cell_image = self._create_test_cell_with_density(density)
            recognized = vision._recognize_by_features(cell_image)
            assert recognized == digit, f"密度{density}で数字{digit}を期待したが{recognized}が返された"
    
    def test_キャリブレーション機能(self):
        """キャリブレーション機能が正しく動作することを確認"""
        vision = ArrowVision(grid_size=3)
        
        # テスト用の参照画像と既知の値を作成
        reference_image = self._create_test_grid_image(3, 3, with_numbers=True)
        known_values = [
            [0, 1, 2],
            [3, 4, 0],
            [1, 2, 3]
        ]
        
        # テンプレートディレクトリが存在しない状態でテスト
        if vision.templates_dir.exists():
            import shutil
            shutil.rmtree(vision.templates_dir)
        
        result = vision.calibrate(reference_image, known_values)
        assert result is True
        assert vision.templates_dir.exists()
        assert len(vision.cell_templates) > 0
    
    def test_パズル状態の読み取り(self):
        """パズル全体の状態を正しく読み取れることを確認"""
        vision = ArrowVision(grid_size=3)
        test_image = self._create_test_grid_image(3, 3, with_numbers=True)
        
        board = vision.read_puzzle_state(test_image)
        assert board is not None
        assert board.size == 3
        # すべてのセルが0-4の範囲内であることを確認
        for row in range(3):
            for col in range(3):
                assert 0 <= board.get_value(row, col) <= 4
    
    def test_セル座標の取得(self):
        """各セルの中心座標を正しく計算できることを確認"""
        vision = ArrowVision(grid_size=4)
        region = (100, 100, 400, 400)
        
        coordinates = vision.get_cell_coordinates(region)
        assert len(coordinates) == 4
        assert len(coordinates[0]) == 4
        
        # 最初のセルの中心座標を確認
        assert coordinates[0][0] == (150, 150)
        # 最後のセルの中心座標を確認
        assert coordinates[3][3] == (450, 450)
    
    def test_グリッド構造の検証(self):
        """グリッド構造の検証が正しく動作することを確認"""
        vision = ArrowVision(grid_size=5)
        
        # グリッド線を含むテスト画像
        grid_image = self._create_test_grid_image(5, 5, with_grid_lines=True)
        gray = cv2.cvtColor(grid_image, cv2.COLOR_BGR2GRAY)
        assert vision._verify_grid_structure(gray) is True
        
        # グリッド線を含まないテスト画像
        no_grid_image = np.ones((500, 500), dtype=np.uint8) * 128
        assert vision._verify_grid_structure(no_grid_image) is False
    
    # ヘルパーメソッド
    def _create_test_grid_image(self, rows: int, cols: int, 
                              with_numbers: bool = False, 
                              with_grid_lines: bool = True) -> np.ndarray:
        """テスト用のグリッド画像を生成"""
        cell_size = 100
        margin = 50
        width = cols * cell_size + 2 * margin
        height = rows * cell_size + 2 * margin
        
        # 白背景の画像を作成
        image = np.ones((height, width, 3), dtype=np.uint8) * 255
        
        # グリッド線を描画
        if with_grid_lines:
            for i in range(rows + 1):
                y = margin + i * cell_size
                cv2.line(image, (margin, y), (width - margin, y), (0, 0, 0), 2)
            
            for j in range(cols + 1):
                x = margin + j * cell_size
                cv2.line(image, (x, margin), (x, height - margin), (0, 0, 0), 2)
        
        # 数字を追加
        if with_numbers:
            for i in range(rows):
                for j in range(cols):
                    digit = (i + j) % 5
                    x = margin + j * cell_size + cell_size // 2
                    y = margin + i * cell_size + cell_size // 2
                    cv2.putText(image, str(digit), (x - 20, y + 20),
                              cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3)
        
        return image
    
    def _create_test_cell_with_density(self, density: float) -> np.ndarray:
        """指定された密度のテストセル画像を生成"""
        size = 50
        image = np.ones((size, size), dtype=np.uint8) * 255
        
        # 密度に応じて黒いピクセルを追加
        num_black_pixels = int(size * size * density)
        indices = np.random.choice(size * size, num_black_pixels, replace=False)
        
        for idx in indices:
            row = idx // size
            col = idx % size
            image[row, col] = 0
        
        return image