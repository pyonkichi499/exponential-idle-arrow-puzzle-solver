"""Computer vision module for detecting and reading 15 puzzle from screen."""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict
import pyautogui
from pathlib import Path
import json
import pytesseract
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
        self.templates_dir = Path(__file__).parent / "templates"
        self.digit_templates = self._load_digit_templates()
        self.calibration_data = self._load_calibration_data()
        
    def _load_digit_templates(self) -> Dict[int, np.ndarray]:
        """Load or create templates for digits 1-15.
        
        Returns:
            Dictionary mapping numbers to template images
        """
        templates = {}
        
        # テンプレート画像が存在する場合は読み込む
        if self.templates_dir.exists():
            for i in range(1, 16):  # 1-15
                template_path = self.templates_dir / f"tile_{i}.png"
                if template_path.exists():
                    template = cv2.imread(str(template_path), cv2.IMREAD_GRAYSCALE)
                    if template is not None:
                        templates[i] = template
        
        return templates
    
    def _load_calibration_data(self) -> Dict:
        """Load calibration data if exists.
        
        Returns:
            Calibration data dictionary
        """
        calibration_file = self.templates_dir / "calibration.json"
        if calibration_file.exists():
            with open(calibration_file, 'r') as f:
                return json.load(f)
        return {
            "tile_size": None,
            "background_color": None,
            "text_color": None,
            "empty_tile_threshold": 10.0,
            "ocr_config": "--oem 3 --psm 8 -c tessedit_char_whitelist=0123456789"
        }
    
    def detect_puzzle(self, image: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """Detect the puzzle grid in the image.
        
        Args:
            image: Screenshot image
            
        Returns:
            (x, y, width, height) of the grid, or None if not found
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply edge detection with bilateral filter for noise reduction
        blurred = cv2.bilateralFilter(gray, 9, 75, 75)
        edges = cv2.Canny(blurred, 30, 100)
        
        # Apply morphological operations
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)
        edges = cv2.erode(edges, kernel, iterations=1)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Sort contours by area
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        
        # Look for square-like contours
        for contour in contours[:10]:
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            if len(approx) >= 4:
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = float(w) / h
                area = w * h
                
                # 15 puzzle should be square and reasonably sized
                if 0.95 <= aspect_ratio <= 1.05 and area > 40000:
                    # Verify it has a grid structure
                    if self._verify_puzzle_structure(gray[y:y+h, x:x+w]):
                        return (x, y, w, h)
        
        return None
    
    def _verify_puzzle_structure(self, roi: np.ndarray) -> bool:
        """Verify if the ROI contains a 15-puzzle structure.
        
        Args:
            roi: Region of interest (grayscale)
            
        Returns:
            True if 15-puzzle structure is detected
        """
        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, 11, 2)
        
        # Check for grid lines
        edges = cv2.Canny(thresh, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=100, maxLineGap=20)
        
        if lines is None:
            return False
        
        # Count grid lines
        horizontal_lines = 0
        vertical_lines = 0
        
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
            
            if angle < 10 or angle > 170:  # Horizontal
                horizontal_lines += 1
            elif 80 < angle < 100:  # Vertical
                vertical_lines += 1
        
        # Expect 3 internal lines in each direction for a 4x4 grid
        return horizontal_lines >= 3 and vertical_lines >= 3
    
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
                # Extract with margin to avoid borders
                margin = max(5, min(tile_width, tile_height) // 15)
                tile_x = x + col * tile_width + margin
                tile_y = y + row * tile_height + margin
                tile_w = tile_width - 2 * margin
                tile_h = tile_height - 2 * margin
                
                tile = image[tile_y:tile_y + tile_h, tile_x:tile_x + tile_w]
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
        # Convert to grayscale if needed
        if len(tile_image.shape) == 3:
            gray = cv2.cvtColor(tile_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = tile_image
        
        # Check if it's an empty tile
        std_dev = np.std(gray)
        empty_threshold = self.calibration_data.get("empty_tile_threshold", 10.0)
        if std_dev < empty_threshold:
            return 0
        
        # Try template matching first if available
        if self.digit_templates:
            result = self._recognize_by_template_matching(gray)
            if result > 0:
                return result
        
        # Fall back to OCR
        return self._recognize_by_ocr(gray)
    
    def _recognize_by_template_matching(self, gray: np.ndarray) -> int:
        """Recognize tile using template matching.
        
        Args:
            gray: Grayscale tile image
            
        Returns:
            Recognized number (1-15) or -1 if not confident
        """
        best_match = -1
        best_score = -1
        threshold = 0.8
        
        for number, template in self.digit_templates.items():
            # Resize template to match tile size
            template_resized = cv2.resize(template, (gray.shape[1], gray.shape[0]))
            
            # Apply template matching
            result = cv2.matchTemplate(gray, template_resized, cv2.TM_CCOEFF_NORMED)
            score = result.max()
            
            if score > best_score and score > threshold:
                best_score = score
                best_match = number
        
        return best_match
    
    def _recognize_by_ocr(self, gray: np.ndarray) -> int:
        """Recognize tile using OCR.
        
        Args:
            gray: Grayscale tile image
            
        Returns:
            Recognized number (1-15) or 0 if failed
        """
        # Preprocess for OCR
        # Apply thresholding
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Invert if necessary (we want dark text on light background)
        if np.mean(thresh) < 127:
            thresh = cv2.bitwise_not(thresh)
        
        # Apply morphological operations to clean up
        kernel = np.ones((2, 2), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        # Resize for better OCR accuracy
        scale_factor = 3
        width = int(thresh.shape[1] * scale_factor)
        height = int(thresh.shape[0] * scale_factor)
        resized = cv2.resize(thresh, (width, height), interpolation=cv2.INTER_CUBIC)
        
        # Apply OCR
        try:
            ocr_config = self.calibration_data.get("ocr_config", 
                                                  "--oem 3 --psm 8 -c tessedit_char_whitelist=0123456789")
            text = pytesseract.image_to_string(resized, config=ocr_config).strip()
            
            # Parse the result
            if text.isdigit():
                number = int(text)
                if 1 <= number <= 15:
                    return number
        except Exception as e:
            print(f"OCR failed: {e}")
        
        return 0
    
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
        found_empty = False
        recognized_numbers = set()
        
        for row in range(self.size):
            for col in range(self.size):
                number = self.recognize_tile(tiles[row][col])
                
                # Validation
                if number == 0:
                    if found_empty:
                        print("Warning: Multiple empty tiles detected")
                    found_empty = True
                    board.empty_pos = (row, col)
                elif number in recognized_numbers:
                    print(f"Warning: Duplicate number {number} detected")
                else:
                    recognized_numbers.add(number)
                
                board.grid[row, col] = number
        
        # Validate the board
        if not found_empty:
            print("Warning: No empty tile detected")
            # Try to find the missing number
            all_numbers = set(range(1, 16))
            missing = all_numbers - recognized_numbers
            if len(missing) == 1:
                # Find a 0 and replace it with the missing number
                for row in range(self.size):
                    for col in range(self.size):
                        if board.grid[row, col] == 0:
                            board.grid[row, col] = missing.pop()
                            break
        
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
    
    def calibrate(self, reference_image: np.ndarray, known_state: List[List[int]]) -> bool:
        """Calibrate detection for specific game graphics.
        
        Args:
            reference_image: Reference image with known puzzle state
            known_state: 2D list of known tile values (0 for empty)
            
        Returns:
            True if calibration successful, False otherwise
        """
        # Detect grid in reference image
        grid_bounds = self.detect_puzzle(reference_image)
        if not grid_bounds:
            print("Failed to detect grid in reference image")
            return False
        
        # Extract tiles
        tiles = self.extract_tiles(reference_image, grid_bounds)
        
        # Create templates directory
        self.templates_dir.mkdir(exist_ok=True)
        
        # Extract and save templates
        tile_samples = {i: [] for i in range(16)}  # 0-15
        
        for row in range(self.size):
            for col in range(self.size):
                number = known_state[row][col]
                tile = tiles[row][col]
                tile_samples[number].append(tile)
        
        # Save templates
        for number, samples in tile_samples.items():
            if samples and number > 0:  # Don't save empty tile
                # Use the clearest sample
                template = samples[0]
                template_path = self.templates_dir / f"tile_{number}.png"
                cv2.imwrite(str(template_path), template)
        
        # Calculate empty tile threshold
        if tile_samples[0]:  # Empty tile samples
            empty_stds = []
            for sample in tile_samples[0]:
                gray = cv2.cvtColor(sample, cv2.COLOR_BGR2GRAY) if len(sample.shape) == 3 else sample
                empty_stds.append(np.std(gray))
            
            self.calibration_data["empty_tile_threshold"] = max(empty_stds) + 5.0
        
        # Save calibration data
        self.calibration_data["tile_size"] = (tiles[0][0].shape[1], tiles[0][0].shape[0])
        
        calibration_file = self.templates_dir / "calibration.json"
        with open(calibration_file, 'w') as f:
            json.dump(self.calibration_data, f, indent=2)
        
        # Reload templates
        self.digit_templates = self._load_digit_templates()
        
        print(f"Calibration successful! Saved {len(self.digit_templates)} templates.")
        return True
    
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


class InteractivePuzzleSelector:
    """Interactive tool for selecting puzzle region on screen."""
    
    def __init__(self):
        self.selection = None
        self.selecting = False
        self.start_point = None
        
    def select_region(self) -> Optional[Tuple[int, int, int, int]]:
        """Let user select a region on screen interactively.
        
        Returns:
            (x, y, width, height) of selected region
        """
        print("Please select the puzzle region by clicking and dragging...")
        print("Press ESC to cancel, Enter to confirm selection")
        
        # Take a screenshot
        screenshot = pyautogui.screenshot()
        screenshot_np = np.array(screenshot)
        image = cv2.cvtColor(screenshot_np, cv2.COLOR_RGB2BGR)
        
        # Create window
        cv2.namedWindow('Select 15-Puzzle Region', cv2.WINDOW_NORMAL)
        cv2.setMouseCallback('Select 15-Puzzle Region', self._mouse_callback)
        
        clone = image.copy()
        
        while True:
            display = clone.copy()
            
            # Draw selection rectangle
            if self.selecting and self.start_point:
                # Get current mouse position
                x, y = pyautogui.position()
                cv2.rectangle(display, self.start_point, (x, y), (0, 255, 0), 2)
            elif self.selection:
                x, y, w, h = self.selection
                cv2.rectangle(display, (x, y), (x + w, y + h), (0, 255, 0), 2)
                # Show instructions
                cv2.putText(display, "Press Enter to confirm, ESC to cancel", 
                          (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            cv2.imshow('Select 15-Puzzle Region', display)
            
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                cv2.destroyAllWindows()
                return None
            elif key == 13 and self.selection:  # Enter
                cv2.destroyAllWindows()
                return self.selection
        
    def _mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events for region selection."""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.start_point = (x, y)
            self.selecting = True
            self.selection = None
            
        elif event == cv2.EVENT_LBUTTONUP:
            self.selecting = False
            if self.start_point:
                x1, y1 = self.start_point
                x2, y2 = x, y
                
                # Ensure positive width and height
                x_min = min(x1, x2)
                y_min = min(y1, y2)
                width = abs(x2 - x1)
                height = abs(y2 - y1)
                
                if width > 10 and height > 10:
                    self.selection = (x_min, y_min, width, height)