"""Computer vision module for detecting and reading 15 puzzle from screen."""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict
import pyautogui
from ...core import BaseVision
from .board import FifteenBoard


class FifteenVision(BaseVision):
    """Detects and reads 15 puzzle from screen captures."""

    def __init__(self, size: int = 4):
        """Initialize the puzzle detector.
        
        Args:
            size: Size of the puzzle grid (default 4x4)
        """
        super().__init__()
        self.size = size
        self.digit_templates = self._load_digit_templates()
        
    def _load_digit_templates(self) -> Dict[int, np.ndarray]:
        """Load or create templates for digits 1-15.
        
        Returns:
            Dictionary mapping numbers to template images
        """
        # In a real implementation, you would load actual templates
        templates = {}
        return templates
    
    def detect_puzzle(self, image: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """Detect the puzzle grid in the image.
        
        Args:
            image: Screenshot image
            
        Returns:
            (x, y, width, height) of the grid, or None if not found
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Look for square-like contours
        for contour in contours:
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            if len(approx) == 4:
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = float(w) / h
                
                # 15 puzzle should be square
                if 0.9 <= aspect_ratio <= 1.1 and w > 200 and h > 200:
                    return (x, y, w, h)
        
        return None
    
    def extract_tiles(self, image: np.ndarray, grid_bounds: Tuple[int, int, int, int]) -> List[List[np.ndarray]]:
        """Extract individual tiles from the grid.
        
        Args:
            image: Screenshot image
            grid_bounds: (x, y, width, height) of the grid
            
        Returns:
            2D list of tile images
        """
        x, y, w, h = grid_bounds
        tile_width = w // self.size
        tile_height = h // self.size
        
        tiles = []
        for row in range(self.size):
            row_tiles = []
            for col in range(self.size):
                tile_x = x + col * tile_width
                tile_y = y + row * tile_height
                
                # Extract with some margin to avoid borders
                margin = 5
                tile = image[
                    tile_y + margin:tile_y + tile_height - margin,
                    tile_x + margin:tile_x + tile_width - margin
                ]
                row_tiles.append(tile)
            tiles.append(row_tiles)
        
        return tiles
    
    def recognize_tile(self, tile_image: np.ndarray) -> int:
        """Recognize the number on a tile (1-15) or 0 for empty.
        
        Args:
            tile_image: Image of a single tile
            
        Returns:
            Recognized number (1-15) or 0 for empty
        """
        # Convert to grayscale
        gray = cv2.cvtColor(tile_image, cv2.COLOR_BGR2GRAY)
        
        # Check if it's an empty tile (uniform color)
        std_dev = np.std(gray)
        if std_dev < 10:  # Low variation suggests empty tile
            return 0
        
        # Apply thresholding
        _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        
        # WARNING: Placeholder implementation
        # Real implementation would use:
        # - OCR with pytesseract
        # - Template matching with digit templates
        # - Or a trained ML model for digit recognition
        
        # For now, return a placeholder
        # This needs to be implemented with actual digit recognition
        return 1  # Placeholder
    
    def read_puzzle_state(self, image: np.ndarray, region: Optional[Tuple[int, int, int, int]] = None) -> Optional[FifteenBoard]:
        """Read the entire puzzle from image.
        
        Args:
            image: Screenshot image
            region: Optional region to constrain search
            
        Returns:
            Board object with the puzzle state, or None if detection failed
        """
        # Detect grid
        if region:
            grid_bounds = (0, 0, region[2], region[3])
            x, y, w, h = region
            image = image[y:y+h, x:x+w]
        else:
            grid_bounds = self.detect_puzzle(image)
            if not grid_bounds:
                return None
        
        # Extract tiles
        tiles = self.extract_tiles(image, grid_bounds)
        
        # Create board
        board = FifteenBoard(self.size)
        
        # Recognize numbers and populate board
        for row in range(self.size):
            for col in range(self.size):
                number = self.recognize_tile(tiles[row][col])
                board.grid[row, col] = number
                if number == 0:
                    board.empty_pos = (row, col)
        
        return board
    
    def get_cell_coordinates(self, region: Tuple[int, int, int, int]) -> List[List[Tuple[int, int]]]:
        """Get screen coordinates for the center of each tile.
        
        Args:
            region: (x, y, width, height) of the grid
            
        Returns:
            2D list of (x, y) coordinates for each tile center
        """
        x, y, w, h = region
        tile_width = w // self.size
        tile_height = h // self.size
        
        coordinates = []
        for row in range(self.size):
            row_coords = []
            for col in range(self.size):
                tile_x = x + col * tile_width + tile_width // 2
                tile_y = y + row * tile_height + tile_height // 2
                row_coords.append((tile_x, tile_y))
            coordinates.append(row_coords)
        
        return coordinates
    
    def calibrate(self, reference_image: np.ndarray) -> bool:
        """Calibrate detection for specific game graphics.
        
        Args:
            reference_image: Reference image with known puzzle state
            
        Returns:
            True if calibration successful, False otherwise
        """
        # TODO: Implement calibration
        # This would involve:
        # 1. User providing a reference image with known tile positions
        # 2. Extracting and storing digit templates
        # 3. Testing recognition accuracy
        return False
    
    def find_empty_tile(self, image: np.ndarray, grid_bounds: Tuple[int, int, int, int]) -> Optional[Tuple[int, int]]:
        """Find the position of the empty tile.
        
        Args:
            image: Screenshot image
            grid_bounds: (x, y, width, height) of the grid
            
        Returns:
            (row, col) of empty tile, or None if not found
        """
        tiles = self.extract_tiles(image, grid_bounds)
        
        for row in range(self.size):
            for col in range(self.size):
                if self.recognize_tile(tiles[row][col]) == 0:
                    return (row, col)
        
        return None