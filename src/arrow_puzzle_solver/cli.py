"""CLI interface for Arrow Puzzle Solver."""

import click
import numpy as np
from .board import Board
from .solver import Solver


def parse_board_input(input_str: str, size: int = 7) -> Board:
    """Parse board input from string.

    Args:
        input_str: Space or comma-separated values, rows separated by newlines
        size: Expected board size

    Returns:
        Board object
    """
    board = Board(size)

    # Split by lines and parse each row
    lines = input_str.strip().split("\n")
    for i, line in enumerate(lines[:size]):
        values = line.replace(",", " ").split()
        for j, val in enumerate(values[:size]):
            board.set_value(i, j, int(val))

    return board


def display_solution(solver: Solver) -> None:
    """Display the solution with moves."""
    print("\nSolved board:")
    print(solver.board)
    print(f"\nTotal moves: {len(solver.moves)}")

    if solver.verify_solution():
        print("✓ Solution verified")
    else:
        print("✗ Solution verification failed")


@click.group()
def cli():
    """Arrow Puzzle Solver for Exponential Idle."""
    pass


@cli.command()
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
        board = parse_board_input(board_str, size)
    else:
        # Interactive input
        print(f"Enter the {size}x{size} board (space-separated values, 0-4):")
        lines = []
        for i in range(size):
            line = click.prompt(f"Row {i+1}")
            lines.append(line)
        board_str = "\n".join(lines)
        board = parse_board_input(board_str, size)

    print("\nInitial board:")
    print(board)

    solver = Solver(board)
    if solver.solve(mode):
        display_solution(solver)
    else:
        print("\n✗ Failed to solve the puzzle")


@cli.command()
@click.option("--size", default=7, help="Board size (default: 7)")
@click.option(
    "--difficulty",
    type=click.Choice(["easy", "medium", "hard"]),
    default="medium",
    help="Puzzle difficulty",
)
def generate(size, difficulty):
    """Generate a random arrow puzzle."""
    # Create a solved board (all 1s)
    board = Board(size)
    board.grid = np.ones((size, size), dtype=int)

    # Apply random taps to scramble it
    import random

    tap_counts = {"easy": size * 2, "medium": size * 4, "hard": size * 6}

    num_taps = tap_counts[difficulty]

    for _ in range(num_taps):
        row = random.randint(0, size - 1)
        col = random.randint(0, size - 1)
        board.tap(row, col)

    print(f"Generated {difficulty} {size}x{size} puzzle:")
    print(board)

    # Verify it can be solved
    solver = Solver(board)
    if solver.solve("expert"):
        print(f"\n✓ Puzzle is solvable in {len(solver.moves)} moves")
    else:
        print("\n✗ Generated puzzle might not be solvable")


@cli.command()
def demo():
    """Run a demonstration with a sample puzzle."""
    print("Demo: Solving a 7x7 arrow puzzle\n")

    # Create a sample puzzle
    board = Board(7)
    # Set some initial values for demonstration
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

    solver = Solver(board)
    print("\nSolving with Expert mode...")

    if solver.solve("expert"):
        display_solution(solver)

        # Show move sequence
        print("\nMove sequence (row, col):")
        for i, (row, col) in enumerate(solver.moves):
            if i % 10 == 0:
                print()
            print(f"({row},{col})", end=" ")
        print()
    else:
        print("\n✗ Failed to solve the demo puzzle")


if __name__ == "__main__":
    cli()
