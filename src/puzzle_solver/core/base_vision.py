"""Base vision interface for puzzle detection and recognition."""

from abc import ABC, abstractmethod
from typing import Optional, Tuple, List, Any
import numpy as np
from .base_board import BaseBoard


class BaseVision(ABC):
    """Abstract base class for puzzle vision/detection."""

    def __init__(self):
        """Initialize the vision detector."""
        pass

    @abstractmethod
    def detect_puzzle(self, image: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """Detect puzzle location in the image.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            (x, y, width, height) of puzzle region, or None if not found
        """
        pass

    @abstractmethod
    def read_puzzle_state(self, image: np.ndarray, region: Optional[Tuple[int, int, int, int]] = None) -> Optional[BaseBoard]:
        """Read puzzle state from image.
        
        Args:
            image: Input image as numpy array
            region: Optional (x, y, width, height) to constrain search
            
        Returns:
            Board instance with detected state, or None if failed
        """
        pass

    @abstractmethod
    def get_cell_coordinates(self, region: Tuple[int, int, int, int]) -> List[List[Tuple[int, int]]]:
        """Get screen coordinates for each cell/tile.
        
        Args:
            region: (x, y, width, height) of the puzzle
            
        Returns:
            2D list of (x, y) coordinates for each cell center
        """
        pass

    def capture_screen_region(self, region: Optional[Tuple[int, int, int, int]] = None) -> np.ndarray:
        """Capture a region of the screen.
        
        Args:
            region: (x, y, width, height) tuple. If None, capture entire screen
            
        Returns:
            Screenshot as numpy array
        """
        import pyautogui
        import cv2
        
        if region:
            screenshot = pyautogui.screenshot(region=region)
        else:
            screenshot = pyautogui.screenshot()
        
        # Convert PIL Image to OpenCV format
        screenshot_np = np.array(screenshot)
        return cv2.cvtColor(screenshot_np, cv2.COLOR_RGB2BGR)

    @abstractmethod
    def calibrate(self, reference_image: np.ndarray) -> bool:
        """Calibrate detection for specific game graphics.
        
        Args:
            reference_image: Reference image with known puzzle state
            
        Returns:
            True if calibration successful, False otherwise
        """
        pass