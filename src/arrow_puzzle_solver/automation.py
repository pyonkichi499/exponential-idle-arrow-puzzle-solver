"""Automation module for executing puzzle solutions with mouse clicks."""

import pyautogui
import time
from typing import List, Tuple, Optional
from .solver import Solver
from .vision import PuzzleDetector


class PuzzleAutomator:
    """Automates the solving of arrow puzzles using mouse clicks."""

    def __init__(self, click_delay: float = 0.1, move_duration: float = 0.1):
        """Initialize the automator.

        Args:
            click_delay: Delay between clicks in seconds
            move_duration: Duration of mouse movement in seconds
        """
        self.click_delay = click_delay
        self.move_duration = move_duration
        self.detector = PuzzleDetector()

        # Safety features
        pyautogui.FAILSAFE = True  # Move mouse to corner to abort
        pyautogui.PAUSE = 0.01  # Small pause between commands

    def execute_moves(
        self,
        moves: List[Tuple[int, int]],
        cell_coordinates: List[List[Tuple[int, int]]],
    ) -> None:
        """Execute a sequence of moves by clicking cells.

        Args:
            moves: List of (row, col) tuples representing cells to click
            cell_coordinates: 2D list of screen coordinates for each cell
        """
        print(f"Executing {len(moves)} moves...")

        for i, (row, col) in enumerate(moves):
            x, y = cell_coordinates[row][col]

            # Move to the cell
            pyautogui.moveTo(x, y, duration=self.move_duration)

            # Click the cell
            pyautogui.click()

            # Progress update
            if (i + 1) % 10 == 0:
                print(f"Progress: {i + 1}/{len(moves)} moves completed")

            # Delay between clicks
            time.sleep(self.click_delay)

        print("All moves completed!")

    def solve_from_region(
        self, region: Tuple[int, int, int, int], mode: str = "expert"
    ) -> bool:
        """Solve puzzle from a specific screen region.

        Args:
            region: (x, y, width, height) of the puzzle region
            mode: Solving mode ("normal", "hard", or "expert")

        Returns:
            True if solved successfully, False otherwise
        """
        # Read puzzle from screen
        print("Reading puzzle from screen...")
        board = self.detector.read_puzzle_from_screen(region)

        if not board:
            print("Failed to detect puzzle!")
            return False

        print("Detected puzzle:")
        print(board)

        # Solve the puzzle
        print(f"Solving with {mode} mode...")
        solver = Solver(board)
        if not solver.solve(mode):
            print("Failed to solve puzzle!")
            return False

        print(f"Solution found with {len(solver.moves)} moves")

        # Get cell coordinates
        screenshot = self.detector.capture_screen_region(region)
        grid_bounds = self.detector.detect_grid(screenshot)

        if not grid_bounds:
            print("Failed to detect grid bounds!")
            return False

        # Adjust grid bounds to screen coordinates
        grid_x = region[0] + grid_bounds[0]
        grid_y = region[1] + grid_bounds[1]
        adjusted_bounds = (grid_x, grid_y, grid_bounds[2], grid_bounds[3])

        cell_coords = self.detector.get_cell_coordinates(adjusted_bounds)

        # Execute moves
        self.execute_moves(solver.moves, cell_coords)

        return True

    def solve_interactive(self, mode: str = "expert") -> bool:
        """Solve puzzle with interactive region selection.

        Args:
            mode: Solving mode ("normal", "hard", or "expert")

        Returns:
            True if solved successfully, False otherwise
        """
        from .vision import InteractivePuzzleSelector

        selector = InteractivePuzzleSelector()
        region = selector.select_region()

        if not region:
            print("No region selected.")
            return False

        return self.solve_from_region(region, mode)

    def continuous_solve(
        self,
        region: Tuple[int, int, int, int],
        mode: str = "expert",
        solve_delay: float = 2.0,
        max_puzzles: Optional[int] = None,
    ) -> None:
        """Continuously solve puzzles in the same region.

        Args:
            region: (x, y, width, height) of the puzzle region
            mode: Solving mode
            solve_delay: Delay between solving puzzles
            max_puzzles: Maximum number of puzzles to solve (None for infinite)
        """
        puzzles_solved = 0

        print("Starting continuous solve mode...")
        print("Move mouse to upper-left corner to stop")

        while max_puzzles is None or puzzles_solved < max_puzzles:
            try:
                # Check for abort (mouse in corner)
                x, y = pyautogui.position()
                if x < 10 and y < 10:
                    print("Aborted by user")
                    break

                # Solve current puzzle
                success = self.solve_from_region(region, mode)

                if success:
                    puzzles_solved += 1
                    print(f"Puzzles solved: {puzzles_solved}")

                    # Wait before next puzzle
                    print(f"Waiting {solve_delay} seconds...")
                    time.sleep(solve_delay)
                else:
                    print("Failed to solve puzzle, retrying...")
                    time.sleep(1)

            except pyautogui.FailSafeException:
                print("Aborted by user (FailSafe)")
                break
            except Exception as e:
                print(f"Error: {e}")
                time.sleep(1)

        print(f"Total puzzles solved: {puzzles_solved}")


def calibrate_click_timing() -> float:
    """Help user calibrate the optimal click delay.

    Returns:
        Recommended click delay in seconds
    """
    print("Click timing calibration")
    print("========================")
    print("This will help determine the optimal click delay for your system.")
    print("Please have a puzzle ready on screen.")
    input("Press Enter when ready...")

    delays = [0.05, 0.1, 0.15, 0.2, 0.3]

    for delay in delays:
        print(f"\nTesting with {delay}s delay...")
        print("Watch if the game registers all clicks properly.")

        # Perform test clicks
        for i in range(5):
            pyautogui.click()
            time.sleep(delay)

        response = input("Did all clicks register? (y/n): ").lower()
        if response == "y":
            print(f"Recommended delay: {delay}s")
            return delay

    print("No suitable delay found. Using default 0.2s")
    return 0.2
