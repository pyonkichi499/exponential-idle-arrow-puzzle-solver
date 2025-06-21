"""Computer vision module for detecting and reading arrow puzzle from screen."""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict
import pyautogui
from .board import Board


class PuzzleDetector:
    """Detects and reads arrow puzzle from screen captures."""

    def __init__(self, grid_size: int = 7):
        """Initialize the puzzle detector.

        Args:
            grid_size: Size of the puzzle grid (default 7x7)
        """
        self.grid_size = grid_size
        self.cell_templates = self._load_cell_templates()

    def _load_cell_templates(self) -> Dict[int, np.ndarray]:
        """Load or create templates for each cell value (0-4).

        Returns:
            Dictionary mapping cell values to template images
        """
        # For now, we'll use simple digit recognition
        # In a real implementation, you would load actual game screenshots
        templates = {}
        return templates

    def capture_screen_region(
        self, region: Optional[Tuple[int, int, int, int]] = None
    ) -> np.ndarray:
        """Capture a region of the screen.

        Args:
            region: (x, y, width, height) tuple. If None, capture entire screen

        Returns:
            Screenshot as numpy array
        """
        if region:
            screenshot = pyautogui.screenshot(region=region)
        else:
            screenshot = pyautogui.screenshot()

        # Convert PIL Image to OpenCV format
        screenshot_np = np.array(screenshot)
        return cv2.cvtColor(screenshot_np, cv2.COLOR_RGB2BGR)

    def detect_grid(self, image: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
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
        contours, _ = cv2.findContours(
            edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # Look for square-like contours that could be the grid
        for contour in contours:
            # Approximate the contour
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)

            # Check if it's roughly square
            if len(approx) == 4:
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = float(w) / h

                # Check if aspect ratio is close to 1 (square)
                if 0.8 <= aspect_ratio <= 1.2 and w > 100 and h > 100:
                    return (x, y, w, h)

        return None

    def extract_cells(
        self, image: np.ndarray, grid_bounds: Tuple[int, int, int, int]
    ) -> List[List[np.ndarray]]:
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
                cell_x = x + col * cell_width
                cell_y = y + row * cell_height
                cell = image[
                    cell_y : cell_y + cell_height, cell_x : cell_x + cell_width
                ]
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
        # Convert to grayscale
        gray = cv2.cvtColor(cell_image, cv2.COLOR_BGR2GRAY)

        # Apply thresholding
        _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

        # Simple template matching or contour analysis
        # For now, we'll use a simplified approach
        # In a real implementation, you would use:
        # - Template matching with actual game digits
        # - OCR with pytesseract
        # - Or a trained ML model

        # WARNING: This is a placeholder implementation!
        # The pixel ratio thresholds below are NOT calibrated for actual game graphics
        # and will need to be adjusted based on real game screenshots.
        
        # Count white pixels as a simple heuristic
        white_pixels = cv2.countNonZero(thresh)
        total_pixels = thresh.shape[0] * thresh.shape[1]
        white_ratio = white_pixels / total_pixels

        # Map white ratio to digits (this is a placeholder)
        # These thresholds are arbitrary and need calibration with actual game assets
        if white_ratio < 0.1:
            return 0
        elif white_ratio < 0.3:
            return 1
        elif white_ratio < 0.5:
            return 2
        elif white_ratio < 0.7:
            return 3
        else:
            return 4

    def read_puzzle_from_screen(
        self, region: Optional[Tuple[int, int, int, int]] = None
    ) -> Optional[Board]:
        """Read the entire puzzle from screen.

        Args:
            region: Screen region to capture, or None for entire screen

        Returns:
            Board object with the puzzle state, or None if detection failed
        """
        # Capture screen
        screenshot = self.capture_screen_region(region)

        # Detect grid
        grid_bounds = self.detect_grid(screenshot)
        if not grid_bounds:
            return None

        # Extract cells
        cells = self.extract_cells(screenshot, grid_bounds)

        # Create board
        board = Board(self.grid_size)

        # Recognize digits and populate board
        for row in range(self.grid_size):
            for col in range(self.grid_size):
                digit = self.recognize_digit(cells[row][col])
                board.set_value(row, col, digit)

        return board

    def get_cell_coordinates(
        self, grid_bounds: Tuple[int, int, int, int]
    ) -> List[List[Tuple[int, int]]]:
        """Get screen coordinates for the center of each cell.

        Args:
            grid_bounds: (x, y, width, height) of the grid

        Returns:
            2D list of (x, y) coordinates for each cell center
        """
        x, y, w, h = grid_bounds
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
        print("Press ESC to cancel")

        # Take a screenshot
        screenshot = pyautogui.screenshot()
        screenshot_np = np.array(screenshot)
        image = cv2.cvtColor(screenshot_np, cv2.COLOR_RGB2BGR)

        # Create window
        cv2.namedWindow("Select Puzzle Region", cv2.WINDOW_NORMAL)
        cv2.setMouseCallback("Select Puzzle Region", self._mouse_callback)

        clone = image.copy()

        while True:
            display = clone.copy()

            # Draw selection rectangle
            if self.selecting and self.start_point:
                current_pos = pyautogui.position()
                cv2.rectangle(display, self.start_point, current_pos, (0, 255, 0), 2)
            elif self.selection:
                x, y, w, h = self.selection
                cv2.rectangle(display, (x, y), (x + w, y + h), (0, 255, 0), 2)

            cv2.imshow("Select Puzzle Region", display)

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
