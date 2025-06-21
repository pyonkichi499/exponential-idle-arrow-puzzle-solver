# Exponential Idle Arrow Puzzle Solver

A Python library and CLI tool for solving arrow puzzles that appear in the mobile game [Exponential Idle](https://conicgames.github.io/exponentialidle/).

## Overview

Arrow puzzles in Exponential Idle are logic puzzles where:
- The board consists of cells with values from 0 to 4
- Tapping a cell increments it and its orthogonal neighbors (mod 5)
- The goal is to make all cells equal to 1
- Puzzles can be solved using the Propagation algorithm and Hard/Expert mode strategies

## Installation

### Prerequisites

- Python 3.8 or higher
- [Rye](https://rye-up.com/) (Python project manager)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/arrow-puzzle-solver.git
cd arrow-puzzle-solver
```

2. Install dependencies using Rye:
```bash
rye sync
```

## Usage

### Command Line Interface

The solver provides a CLI with several commands:

#### Solve a puzzle

```bash
# Interactive input
rye run python -m arrow_puzzle_solver solve

# From file
rye run python -m arrow_puzzle_solver solve --input-file puzzle.txt

# With specific mode
rye run python -m arrow_puzzle_solver solve --mode expert
```

#### Generate a random puzzle

```bash
# Generate a 7x7 puzzle with medium difficulty
rye run python -m arrow_puzzle_solver generate --size 7 --difficulty medium
```

#### Run the demo

```bash
rye run python -m arrow_puzzle_solver demo
```

#### Automatically solve puzzles on screen

```bash
# Interactive region selection
rye run python -m arrow_puzzle_solver auto-solve

# With specific region
rye run python -m arrow_puzzle_solver auto-solve --region 100 200 500 500

# Continuous mode
rye run python -m arrow_puzzle_solver auto-solve --region 100 200 500 500 --continuous

# Calibrate click timing
rye run python -m arrow_puzzle_solver calibrate
```

### Python API

You can also use the solver programmatically:

```python
from arrow_puzzle_solver import Board, Solver

# Create a board
board = Board(size=7)

# Set initial values (0-4)
board.set_value(0, 0, 2)
board.set_value(0, 1, 3)
# ... set other values

# Create solver and solve
solver = Solver(board)
if solver.solve(mode='expert'):
    print("Solved!")
    print(solver.board)
    print(f"Total moves: {len(solver.moves)}")
else:
    print("Could not solve the puzzle")
```

## Algorithm

### Propagation Method

The basic solving strategy that processes each row:
1. Solve the center tile
2. Solve tiles to the left of center (1, 2, 3 spaces)
3. Solve tiles to the right of center (1, 2, 3 spaces)
4. Repeat for all rows except the bottom row

### Hard/Expert Mode

An advanced strategy for difficult puzzles:
1. Apply propagation first
2. Encode bottom row information onto the top row
3. Apply specific tap sequences based on bottom row values
4. Propagate again from the top

## Automatic Screen Solving

The solver can automatically detect and solve puzzles displayed on your screen:

### Features
- **Screen capture and puzzle detection**: Automatically finds puzzle grids on screen
- **Interactive region selection**: Click and drag to select puzzle area
- **Automatic mouse clicking**: Executes solution with configurable delays
- **Continuous mode**: Solve multiple puzzles automatically

### Usage
```bash
# Interactive mode - select region with mouse
rye run python -m arrow_puzzle_solver auto-solve

# Specify exact region (x, y, width, height)
rye run python -m arrow_puzzle_solver auto-solve --region 100 200 500 500

# Continuous solving
rye run python -m arrow_puzzle_solver auto-solve --region 100 200 500 500 --continuous --max-puzzles 10

# Adjust click timing
rye run python -m arrow_puzzle_solver auto-solve --click-delay 0.2
```

### Calibration
For optimal performance, calibrate the click timing for your system:
```bash
rye run python -m arrow_puzzle_solver calibrate
```

### Important Notes on Screen Recognition

**⚠️ The current image recognition implementation is a proof of concept and has not been tested with actual Exponential Idle gameplay.**

- The digit recognition uses a simple pixel ratio heuristic that will likely need adjustment for actual game graphics
- For production use, you should:
  - Capture actual game screenshots of digits 0-4
  - Implement proper template matching or OCR
  - Test and calibrate the grid detection for your specific game resolution
- The current implementation serves as a framework that can be adapted once actual game assets are available

## Input Format

Puzzles are represented as space or comma-separated values:

```
2 3 1 4 0 2 3
1 4 2 3 1 0 4
3 0 4 1 2 3 1
4 2 1 0 3 4 2
0 3 2 4 1 2 0
2 1 3 2 4 0 3
3 4 0 1 2 3 4
```

## Development

### Project Structure

```
arrow-puzzle-solver/
├── src/arrow_puzzle_solver/
│   ├── __init__.py
│   ├── board.py        # Board representation
│   ├── solver.py       # Solving algorithms
│   └── cli.py          # CLI interface
├── tests/              # Test suite
├── pyproject.toml      # Project configuration
└── README.md
```

### Running Tests

```bash
rye run pytest tests/ -v
```

### Code Quality

```bash
# Format code
rye run black src/ tests/

# Lint code
rye run ruff src/ tests/
```

## Limitations

- Not all randomly generated puzzles may be solvable with the current algorithm
- The algorithm is designed for puzzles that appear in Exponential Idle, which are guaranteed to be solvable
- Currently supports square boards (default 7x7)
- **Image recognition for auto-solve is not production-ready** - see [KNOWN_ISSUES.md](KNOWN_ISSUES.md) for details

## Examples

### Basic Usage Example

```python
from arrow_puzzle_solver import Board, Solver

# Create a simple 3x3 board
board = Board(size=3)

# Set up a solvable pattern
board.tap(1, 1)  # Tap center

# Solve it
solver = Solver(board)
solver.solve()
```

### CLI Example

```bash
# Solve a puzzle from file
echo "2 3 1
      1 4 2
      3 0 4" > puzzle.txt
      
rye run python -m arrow_puzzle_solver solve --input-file puzzle.txt --size 3
```

## References

- [Exponential Idle Guide - Arrow Puzzles](https://exponential-idle-guides.netlify.app/guides/asd/)
- Algorithm implementation based on the solving methods described in the guide

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.