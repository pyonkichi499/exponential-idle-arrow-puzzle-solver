"""Computer vision module for detecting and reading arrow puzzle from screen."""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict
import pyautogui
from pathlib import Path
import json
from ...core import BaseVision
from .board import ArrowBoard


class ArrowVision(BaseVision):
    """Detects and reads arrow puzzle from screen captures."""

    def __init__(self, grid_size: int = 7):
        """Initialize the puzzle detector.
        
        Args:
            grid_size: Size of the puzzle grid (default 7x7)
        """
        super().__init__()
        self.grid_size = grid_size
        self.templates_dir = Path(__file__).parent / "templates"
        self.cell_templates = self._load_cell_templates()
        self.calibration_data = self._load_calibration_data()
        
    def _load_cell_templates(self) -> Dict[int, np.ndarray]:
        """Load or create templates for each cell value (0-4).
        
        Returns:
            Dictionary mapping cell values to template images
        """
        templates = {}
        
        # テンプレート画像が存在する場合は読み込む
        if self.templates_dir.exists():
            for i in range(5):
                template_path = self.templates_dir / f"digit_{i}.png"
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
            "cell_size": None,
            "grid_color_range": None,
            "digit_color_range": None,
            "threshold_values": {
                "0": {"min": 0.0, "max": 0.15},
                "1": {"min": 0.15, "max": 0.35},
                "2": {"min": 0.35, "max": 0.55},
                "3": {"min": 0.55, "max": 0.75},
                "4": {"min": 0.75, "max": 1.0}
            }
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
        
        # Apply adaptive thresholding for better edge detection
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, 11, 2)
        
        # Apply morphological operations to clean up the image
        kernel = np.ones((3, 3), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Sort contours by area (largest first)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        
        # Look for square-like contours that could be the grid
        for contour in contours[:10]:  # Check only the 10 largest contours
            # Approximate the contour
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # Check if it's roughly square
            if len(approx) >= 4:
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = float(w) / h
                area = w * h
                
                # Check if aspect ratio is close to 1 (square) and has reasonable size
                if 0.9 <= aspect_ratio <= 1.1 and area > 10000:
                    # Additional check: verify it has a grid-like structure
                    if self._verify_grid_structure(gray[y:y+h, x:x+w]):
                        return (x, y, w, h)
        
        return None
    
    def _verify_grid_structure(self, roi: np.ndarray) -> bool:
        """Verify if the ROI contains a grid structure.
        
        Args:
            roi: Region of interest (grayscale)
            
        Returns:
            True if grid structure is detected
        """
        # Apply edge detection
        edges = cv2.Canny(roi, 50, 150)
        
        # Detect lines using Hough transform
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, minLineLength=50, maxLineGap=10)
        
        if lines is None:
            return False
        
        # Count horizontal and vertical lines
        horizontal_lines = 0
        vertical_lines = 0
        
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
            
            if angle < 10 or angle > 170:  # Horizontal
                horizontal_lines += 1
            elif 80 < angle < 100:  # Vertical
                vertical_lines += 1
        
        # Expect at least grid_size lines in each direction
        return horizontal_lines >= self.grid_size and vertical_lines >= self.grid_size
    
    def extract_cells(self, image: np.ndarray, grid_bounds: Tuple[int, int, int, int]) -> List[List[np.ndarray]]:
        """Extract individual cells from the grid.
        
        Args:
            image: Screenshot image
            grid_bounds: (x, y, width, height) of the grid
            
        Returns:
            2D list of cell images
        """
        x, y, w, h = grid_bounds
        cell_width = w // self.grid_size
        cell_height = h // self.grid_size
        
        cells = []
        for row in range(self.grid_size):
            row_cells = []
            for col in range(self.grid_size):
                # Extract cell with some margin to avoid grid lines
                margin = max(2, min(cell_width, cell_height) // 10)
                cell_x = x + col * cell_width + margin
                cell_y = y + row * cell_height + margin
                cell_w = cell_width - 2 * margin
                cell_h = cell_height - 2 * margin
                
                cell = image[cell_y:cell_y + cell_h, cell_x:cell_x + cell_w]
                row_cells.append(cell)
            cells.append(row_cells)
        
        return cells
    
    def recognize_digit(self, cell_image: np.ndarray) -> int:
        """Recognize the digit (0-4) in a cell image.
        
        Args:
            cell_image: Image of a single cell
            
        Returns:
            Recognized digit (0-4)
        """
        # Convert to grayscale if needed
        if len(cell_image.shape) == 3:
            gray = cv2.cvtColor(cell_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = cell_image
        
        # If we have templates, use template matching
        if self.cell_templates:
            return self._recognize_by_template_matching(gray)
        
        # Otherwise, use feature-based recognition
        return self._recognize_by_features(gray)
    
    def _recognize_by_template_matching(self, gray: np.ndarray) -> int:
        """Recognize digit using template matching.
        
        Args:
            gray: Grayscale cell image
            
        Returns:
            Recognized digit (0-4)
        """
        best_match = -1
        best_score = -1
        
        for digit, template in self.cell_templates.items():
            # Resize template to match cell size
            template_resized = cv2.resize(template, (gray.shape[1], gray.shape[0]))
            
            # Apply template matching
            result = cv2.matchTemplate(gray, template_resized, cv2.TM_CCOEFF_NORMED)
            score = result.max()
            
            if score > best_score:
                best_score = score
                best_match = digit
        
        # If confidence is too low, fall back to feature-based
        if best_score < 0.7:
            return self._recognize_by_features(gray)
        
        return best_match
    
    def _recognize_by_features(self, gray: np.ndarray) -> int:
        """Recognize digit using image features.
        
        Args:
            gray: Grayscale cell image
            
        Returns:
            Recognized digit (0-4)
        """
        # Apply thresholding
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Calculate features
        total_pixels = thresh.shape[0] * thresh.shape[1]
        white_pixels = cv2.countNonZero(thresh)
        black_pixels = total_pixels - white_pixels
        
        # Calculate density (ratio of dark pixels)
        density = black_pixels / total_pixels
        
        # Use calibrated thresholds if available
        thresholds = self.calibration_data.get("threshold_values", {})
        
        for digit_str, ranges in thresholds.items():
            if ranges["min"] <= density < ranges["max"]:
                return int(digit_str)
        
        # Fallback to default thresholds
        if density < 0.15:
            return 0
        elif density < 0.35:
            return 1
        elif density < 0.55:
            return 2
        elif density < 0.75:
            return 3
        else:
            return 4
    
    def read_puzzle_state(self, image: np.ndarray, region: Optional[Tuple[int, int, int, int]] = None) -> Optional[ArrowBoard]:
        """Read the entire puzzle from image.
        
        Args:
            image: Screenshot image
            region: Optional region to constrain search
            
        Returns:
            Board object with the puzzle state, or None if detection failed
        """
        # Detect grid
        if region:
            # If region provided, use it directly
            grid_bounds = (0, 0, region[2], region[3])
            # Extract the region from the image
            x, y, w, h = region
            image = image[y:y+h, x:x+w]
        else:
            grid_bounds = self.detect_puzzle(image)
            if not grid_bounds:
                return None
        
        # Extract cells
        cells = self.extract_cells(image, grid_bounds)
        
        # Create board
        board = ArrowBoard(self.grid_size)
        
        # Recognize digits and populate board
        for row in range(self.grid_size):
            for col in range(self.grid_size):
                digit = self.recognize_digit(cells[row][col])
                board.set_value(row, col, digit)
        
        return board
    
    def get_cell_coordinates(self, region: Tuple[int, int, int, int]) -> List[List[Tuple[int, int]]]:
        """Get screen coordinates for the center of each cell.
        
        Args:
            region: (x, y, width, height) of the grid
            
        Returns:
            2D list of (x, y) coordinates for each cell center
        """
        x, y, w, h = region
        cell_width = w // self.grid_size
        cell_height = h // self.grid_size
        
        coordinates = []
        for row in range(self.grid_size):
            row_coords = []
            for col in range(self.grid_size):
                cell_x = x + col * cell_width + cell_width // 2
                cell_y = y + row * cell_height + cell_height // 2
                row_coords.append((cell_x, cell_y))
            coordinates.append(row_coords)
        
        return coordinates
    
    def calibrate(self, reference_image: np.ndarray, known_values: List[List[int]]) -> bool:
        """Calibrate detection for specific game graphics.
        
        Args:
            reference_image: Reference image with known puzzle state
            known_values: 2D list of known cell values
            
        Returns:
            True if calibration successful, False otherwise
        """
        # Detect grid in reference image
        grid_bounds = self.detect_puzzle(reference_image)
        if not grid_bounds:
            print("Failed to detect grid in reference image")
            return False
        
        # Extract cells
        cells = self.extract_cells(reference_image, grid_bounds)
        
        # Create templates directory if it doesn't exist
        self.templates_dir.mkdir(exist_ok=True)
        
        # Extract and save templates for each digit
        digit_samples = {i: [] for i in range(5)}
        
        for row in range(self.grid_size):
            for col in range(self.grid_size):
                digit = known_values[row][col]
                cell = cells[row][col]
                digit_samples[digit].append(cell)
        
        # Save representative templates
        for digit, samples in digit_samples.items():
            if samples:
                # Use the first sample as template (could be improved by averaging)
                template = samples[0]
                template_path = self.templates_dir / f"digit_{digit}.png"
                cv2.imwrite(str(template_path), template)
        
        # Calculate threshold values based on samples
        threshold_values = {}
        for digit, samples in digit_samples.items():
            if samples:
                densities = []
                for sample in samples:
                    gray = cv2.cvtColor(sample, cv2.COLOR_BGR2GRAY) if len(sample.shape) == 3 else sample
                    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                    density = (gray.shape[0] * gray.shape[1] - cv2.countNonZero(thresh)) / (gray.shape[0] * gray.shape[1])
                    densities.append(density)
                
                min_density = min(densities) - 0.05
                max_density = max(densities) + 0.05
                threshold_values[str(digit)] = {"min": min_density, "max": max_density}
        
        # Save calibration data
        self.calibration_data["threshold_values"] = threshold_values
        self.calibration_data["cell_size"] = (cells[0][0].shape[1], cells[0][0].shape[0])
        
        calibration_file = self.templates_dir / "calibration.json"
        with open(calibration_file, 'w') as f:
            json.dump(self.calibration_data, f, indent=2)
        
        # Reload templates
        self.cell_templates = self._load_cell_templates()
        
        print(f"Calibration successful! Saved {len(self.cell_templates)} templates.")
        return True


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
        cv2.namedWindow('Select Puzzle Region', cv2.WINDOW_NORMAL)
        cv2.setMouseCallback('Select Puzzle Region', self._mouse_callback)
        
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
            
            cv2.imshow('Select Puzzle Region', display)
            
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