"""Unified CLI interface for Puzzle Solver."""

import click
import numpy as np
from typing import Optional
from .puzzles import AVAILABLE_PUZZLES, get_puzzle_board, get_puzzle_solver

# Import automation only if available
try:
    from .automation import PuzzleAutomator, calibrate_click_timing
    HAS_AUTOMATION = True
except ImportError:
    HAS_AUTOMATION = False


# Arrow puzzle specific functions
def parse_arrow_board_input(input_str: str, size: int = 7):
    """Parse arrow puzzle board input from string."""
    from .puzzles.arrow import ArrowBoard
    board = ArrowBoard(size)
    
    lines = input_str.strip().split("\n")
    for i, line in enumerate(lines[:size]):
        values = line.replace(",", " ").split()
        for j, val in enumerate(values[:size]):
            board.set_value(i, j, int(val))
    
    return board


def parse_fifteen_board_input(input_str: str, size: int = 4):
    """Parse 15 puzzle board input from string."""
    from .puzzles.fifteen import FifteenBoard
    board = FifteenBoard(size)
    
    lines = input_str.strip().split("\n")
    for i, line in enumerate(lines[:size]):
        values = line.replace(",", " ").split()
        for j, val in enumerate(values[:size]):
            num = int(val) if val.strip() != "" else 0
            board.grid[i, j] = num
            if num == 0:
                board.empty_pos = (i, j)
    
    return board


def display_solution(solver, puzzle_type: str) -> None:
    """Display the solution with moves."""
    print("\nSolved board:")
    print(solver.board)
    
    solution = solver.get_solution()
    print(f"\nTotal moves: {len(solution)}")
    
    if solver.verify_solution():
        print("✓ Solution verified")
    else:
        print("✗ Solution verification failed")
    
    # Show statistics
    stats = solver.get_stats()
    if "nodes_explored" in stats:
        print(f"Nodes explored: {stats['nodes_explored']}")
    if "algorithm" in stats:
        print(f"Algorithm: {stats['algorithm']}")


@click.group()
def cli():
    """Puzzle Solver - A framework for solving various puzzles."""
    pass


# Arrow puzzle commands
@cli.group()
def arrow():
    """Commands for arrow puzzles."""
    pass


@arrow.command()
@click.option("--size", default=7, help="Board size (default: 7)")
@click.option(
    "--mode",
    type=click.Choice(["normal", "hard", "expert"]),
    default="expert",
    help="Solving mode",
)
@click.option("--input-file", type=click.File("r"), help="Read board from file")
def solve(size, mode, input_file):
    """Solve an arrow puzzle."""
    if input_file:
        board_str = input_file.read()
        board = parse_arrow_board_input(board_str, size)
    else:
        print(f"Enter the {size}x{size} board (space-separated values, 0-4):")
        lines = []
        for i in range(size):
            line = click.prompt(f"Row {i+1}")
            lines.append(line)
        board_str = "\n".join(lines)
        board = parse_arrow_board_input(board_str, size)
    
    print("\nInitial board:")
    print(board)
    
    from .puzzles.arrow import ArrowSolver
    solver = ArrowSolver(board)
    if solver.solve(mode=mode):
        display_solution(solver, "arrow")
    else:
        print("\n✗ Failed to solve the puzzle")


@arrow.command()
@click.option("--size", default=7, help="Board size (default: 7)")
@click.option(
    "--difficulty",
    type=click.Choice(["easy", "medium", "hard"]),
    default="medium",
    help="Puzzle difficulty",
)
def generate(size, difficulty):
    """Generate a random arrow puzzle."""
    from .puzzles.arrow import ArrowBoard
    
    board = ArrowBoard(size)
    board.grid = np.ones((size, size), dtype=int)
    
    import random
    tap_counts = {"easy": size * 2, "medium": size * 4, "hard": size * 6}
    num_taps = tap_counts[difficulty]
    
    for _ in range(num_taps):
        row = random.randint(0, size - 1)
        col = random.randint(0, size - 1)
        board.tap(row, col)
    
    print(f"Generated {difficulty} {size}x{size} puzzle:")
    print(board)
    
    from .puzzles.arrow import ArrowSolver
    solver = ArrowSolver(board)
    if solver.solve("expert"):
        print(f"\n✓ Puzzle is solvable in {len(solver.get_solution())} moves")
    else:
        print("\n✗ Generated puzzle might not be solvable")


# 15 puzzle commands
@cli.group()
def fifteen():
    """Commands for 15 puzzles."""
    pass


@fifteen.command()
@click.option("--size", default=4, help="Board size (default: 4)")
@click.option(
    "--algorithm",
    type=click.Choice(["astar", "iddfs"]),
    default="astar",
    help="Solving algorithm",
)
@click.option("--input-file", type=click.File("r"), help="Read board from file")
def solve(size, algorithm, input_file):
    """Solve a 15 puzzle."""
    if input_file:
        board_str = input_file.read()
        board = parse_fifteen_board_input(board_str, size)
    else:
        print(f"Enter the {size}x{size} board (1-{size*size-1}, 0 or space for empty):")
        lines = []
        for i in range(size):
            line = click.prompt(f"Row {i+1}")
            lines.append(line)
        board_str = "\n".join(lines)
        board = parse_fifteen_board_input(board_str, size)
    
    print("\nInitial board:")
    print(board)
    
    # Check solvability
    if not board.is_solvable():
        print("\n✗ This puzzle configuration is not solvable!")
        return
    
    from .puzzles.fifteen import FifteenSolver
    solver = FifteenSolver(board)
    if solver.solve(algorithm=algorithm):
        display_solution(solver, "fifteen")
        
        # Show move sequence
        print("\nMove sequence:")
        for i, step in enumerate(solver.get_solution_steps()):
            print(f"  {step}")
    else:
        print("\n✗ Failed to solve the puzzle")


@fifteen.command()
@click.option("--size", default=4, help="Board size (default: 4)")
@click.option("--shuffles", default=50, help="Number of shuffle moves")
def generate(size, shuffles):
    """Generate a random 15 puzzle."""
    from .puzzles.fifteen import FifteenBoard
    
    board = FifteenBoard(size)
    board.shuffle(shuffles)
    
    print(f"Generated {size}x{size} puzzle ({shuffles} shuffle moves):")
    print(board)
    
    if board.is_solvable():
        print("\n✓ Puzzle is solvable")
        print(f"Manhattan distance: {board.manhattan_distance()}")
    else:
        print("\n✗ Generated puzzle is not solvable")


# Auto-solve commands
@cli.command()
@click.argument("puzzle_type", type=click.Choice(["arrow", "fifteen"]))
@click.option(
    "--region",
    nargs=4,
    type=int,
    help="Screen region as 'x y width height'",
)
@click.option(
    "--continuous",
    is_flag=True,
    help="Continuously solve puzzles",
)
@click.option(
    "--click-delay",
    default=0.1,
    type=float,
    help="Delay between clicks in seconds",
)
@click.option(
    "--solve-delay",
    default=2.0,
    type=float,
    help="Delay between puzzles in continuous mode",
)
@click.option(
    "--max-puzzles",
    type=int,
    help="Maximum puzzles to solve in continuous mode",
)
# Arrow-specific options
@click.option(
    "--mode",
    type=click.Choice(["normal", "hard", "expert"]),
    default="expert",
    help="Arrow puzzle solving mode",
)
# 15-puzzle specific options
@click.option(
    "--algorithm",
    type=click.Choice(["astar", "iddfs"]),
    default="astar",
    help="15 puzzle solving algorithm",
)
def auto_solve(puzzle_type, region, continuous, click_delay, solve_delay, 
               max_puzzles, mode, algorithm):
    """Automatically solve puzzles on screen using image recognition."""
    if not HAS_AUTOMATION:
        print("Error: Automation features require pyautogui and opencv-python packages.")
        print("Install them with: rye add pyautogui opencv-python")
        return
    
    automator = PuzzleAutomator(puzzle_type, click_delay=click_delay)
    
    # Prepare solver parameters
    solver_params = {}
    if puzzle_type == "arrow":
        solver_params["mode"] = mode
    elif puzzle_type == "fifteen":
        solver_params["algorithm"] = algorithm
    
    if region:
        x, y, w, h = region
        screen_region = (x, y, w, h)
        print(f"Using region: x={x}, y={y}, width={w}, height={h}")
    else:
        print("Please select the puzzle region on screen...")
        if not automator.solve_interactive(solver_params):
            return
        if continuous:
            print("Note: Continuous mode requires --region parameter")
            return
    
    if continuous and region:
        automator.continuous_solve(
            screen_region, 
            solver_params=solver_params,
            solve_delay=solve_delay,
            max_puzzles=max_puzzles
        )
    elif region:
        automator.solve_from_region(screen_region, solver_params)


@cli.command()
def calibrate():
    """Calibrate click timing for optimal performance."""
    if not HAS_AUTOMATION:
        print("Error: Calibration requires pyautogui and opencv-python packages.")
        print("Install them with: rye add pyautogui opencv-python")
        return
    
    recommended_delay = calibrate_click_timing()
    print(f"\nRecommended --click-delay: {recommended_delay}")
    print(f"Use: puzzle-solver auto-solve [puzzle_type] --click-delay {recommended_delay}")


@cli.command()
def list_puzzles():
    """List available puzzle types."""
    print("Available puzzle types:")
    # Since AVAILABLE_PUZZLES is a property, we need to access it differently
    from .puzzles import _PUZZLE_REGISTRY
    puzzles = list(_PUZZLE_REGISTRY.keys())
    
    for puzzle in puzzles:
        print(f"  - {puzzle}")
    
    if not puzzles:
        print("  No puzzles registered yet.")


# Demo commands
@cli.group()
def demo():
    """Demo commands for various puzzles."""
    pass


@demo.command()
def arrow():
    """Run arrow puzzle demonstration."""
    from .puzzles.arrow import ArrowBoard, ArrowSolver
    
    print("Demo: Solving a 7x7 arrow puzzle\n")
    
    board = ArrowBoard(7)
    sample_values = [
        [2, 3, 1, 4, 0, 2, 3],
        [1, 4, 2, 3, 1, 0, 4],
        [3, 0, 4, 1, 2, 3, 1],
        [4, 2, 1, 0, 3, 4, 2],
        [0, 3, 2, 4, 1, 2, 0],
        [2, 1, 3, 2, 4, 0, 3],
        [3, 4, 0, 1, 2, 3, 4],
    ]
    
    for i in range(7):
        for j in range(7):
            board.set_value(i, j, sample_values[i][j])
    
    print("Initial board:")
    print(board)
    
    solver = ArrowSolver(board)
    print("\nSolving with Expert mode...")
    
    if solver.solve("expert"):
        display_solution(solver, "arrow")
    else:
        print("\n✗ Failed to solve the demo puzzle")


@demo.command()
def fifteen():
    """Run 15 puzzle demonstration."""
    from .puzzles.fifteen import FifteenBoard, FifteenSolver
    
    print("Demo: Solving a 4x4 fifteen puzzle\n")
    
    board = FifteenBoard(4)
    board.shuffle(20)  # Light shuffle for demo
    
    print("Initial board:")
    print(board)
    print(f"Manhattan distance: {board.manhattan_distance()}")
    
    solver = FifteenSolver(board)
    print("\nSolving with A* algorithm...")
    
    if solver.solve("astar"):
        display_solution(solver, "fifteen")
    else:
        print("\n✗ Failed to solve the demo puzzle")


if __name__ == "__main__":
    cli()