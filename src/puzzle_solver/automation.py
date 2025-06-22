"""Generic automation module for executing puzzle solutions with mouse clicks."""

import pyautogui
import time
from typing import List, Tuple, Optional, Any, Union
from .core import BaseBoard, BaseSolver, BaseVision
from .puzzles import get_puzzle_board, get_puzzle_solver, get_puzzle_vision


class PuzzleAutomator:
    """Automates the solving of puzzles using mouse clicks."""
    
    def __init__(self, puzzle_type: str, click_delay: float = 0.1, move_duration: float = 0.1):
        """Initialize the automator.
        
        Args:
            puzzle_type: Type of puzzle ("arrow", "fifteen", etc.)
            click_delay: Delay between clicks in seconds
            move_duration: Duration of mouse movement in seconds
        """
        self.puzzle_type = puzzle_type
        self.click_delay = click_delay
        self.move_duration = move_duration
        
        # Get vision class for this puzzle type
        vision_class = get_puzzle_vision(puzzle_type)
        self.detector = vision_class()
        
        # Safety features
        pyautogui.FAILSAFE = True  # Move mouse to corner to abort
        pyautogui.PAUSE = 0.01  # Small pause between commands
    
    def execute_arrow_moves(self, moves: List[Tuple[int, int]], 
                           cell_coordinates: List[List[Tuple[int, int]]]) -> None:
        """Execute arrow puzzle moves by clicking cells.
        
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
    
    def execute_fifteen_moves(self, moves: List[str], 
                             tile_coordinates: List[List[Tuple[int, int]]],
                             empty_pos: Tuple[int, int]) -> None:
        """Execute 15 puzzle moves by clicking adjacent tiles.
        
        Args:
            moves: List of directions to move empty space
            tile_coordinates: 2D list of screen coordinates for each tile
            empty_pos: Current position of empty tile
        """
        print(f"Executing {len(moves)} moves...")
        
        current_empty = list(empty_pos)
        
        for i, move in enumerate(moves):
            # Calculate which tile to click (opposite of empty space movement)
            if move == "up" and current_empty[0] > 0:
                tile_row = current_empty[0] - 1
                tile_col = current_empty[1]
                current_empty[0] -= 1
            elif move == "down" and current_empty[0] < 3:
                tile_row = current_empty[0] + 1
                tile_col = current_empty[1]
                current_empty[0] += 1
            elif move == "left" and current_empty[1] > 0:
                tile_row = current_empty[0]
                tile_col = current_empty[1] - 1
                current_empty[1] -= 1
            elif move == "right" and current_empty[1] < 3:
                tile_row = current_empty[0]
                tile_col = current_empty[1] + 1
                current_empty[1] += 1
            else:
                continue
            
            # Click the tile
            x, y = tile_coordinates[tile_row][tile_col]
            pyautogui.moveTo(x, y, duration=self.move_duration)
            pyautogui.click()
            
            # Progress update
            if (i + 1) % 5 == 0:
                print(f"Progress: {i + 1}/{len(moves)} moves completed")
            
            # Delay between moves
            time.sleep(self.click_delay)
        
        print("All moves completed!")
    
    def solve_from_region(self, region: Tuple[int, int, int, int], 
                         solver_params: dict = None) -> bool:
        """Solve puzzle from a specific screen region.
        
        Args:
            region: (x, y, width, height) of the puzzle region
            solver_params: Parameters to pass to the solver
            
        Returns:
            True if solved successfully, False otherwise
        """
        if solver_params is None:
            solver_params = {}
        
        # Read puzzle from screen
        print("Reading puzzle from screen...")
        screenshot = self.detector.capture_screen_region(region)
        board = self.detector.read_puzzle_state(screenshot, (0, 0, region[2], region[3]))
        
        if not board:
            print("Failed to detect puzzle!")
            return False
        
        print("Detected puzzle:")
        print(board)
        
        # Create and run solver
        print(f"Solving {self.puzzle_type} puzzle...")
        solver_class = get_puzzle_solver(self.puzzle_type)
        solver = solver_class(board)
        
        if not solver.solve(**solver_params):
            print("Failed to solve puzzle!")
            return False
        
        solution = solver.get_solution()
        print(f"Solution found with {len(solution)} moves")
        
        # Get cell/tile coordinates
        cell_coords = self.detector.get_cell_coordinates(region)
        
        # Execute moves based on puzzle type
        if self.puzzle_type == "arrow":
            self.execute_arrow_moves(solution, cell_coords)
        elif self.puzzle_type == "fifteen":
            # Find empty position for 15 puzzle
            empty_pos = board.empty_pos if hasattr(board, 'empty_pos') else (3, 3)
            self.execute_fifteen_moves(solution, cell_coords, empty_pos)
        else:
            print(f"Move execution not implemented for {self.puzzle_type}")
            return False
        
        return True
    
    def solve_interactive(self, solver_params: dict = None) -> bool:
        """Solve puzzle with interactive region selection.
        
        Args:
            solver_params: Parameters to pass to the solver
            
        Returns:
            True if solved successfully, False otherwise
        """
        from .puzzles.arrow.vision import InteractivePuzzleSelector
        
        selector = InteractivePuzzleSelector()
        region = selector.select_region()
        
        if not region:
            print("No region selected.")
            return False
        
        return self.solve_from_region(region, solver_params)
    
    def continuous_solve(self, region: Tuple[int, int, int, int], 
                        solver_params: dict = None,
                        solve_delay: float = 2.0,
                        max_puzzles: Optional[int] = None) -> None:
        """Continuously solve puzzles in the same region.
        
        Args:
            region: (x, y, width, height) of the puzzle region
            solver_params: Parameters to pass to the solver
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
                success = self.solve_from_region(region, solver_params)
                
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
        if response == 'y':
            print(f"Recommended delay: {delay}s")
            return delay
    
    print("No suitable delay found. Using default 0.2s")
    return 0.2