# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is the "puzzle-solver" project - an extensible framework for solving various types of puzzles. Currently supports:
- Arrow puzzles from the mobile game "Exponential Idle"
- 15-puzzle (sliding puzzle)

## Current State

The project has been implemented with the following components:
- Extensible architecture with abstract base classes
- Arrow puzzle solver:
  - Board representation for arrow puzzles
  - Solver implementing the Propagation algorithm
  - Support for Hard/Expert mode solving
  - Experimental image recognition for auto-solving
- 15-puzzle solver:
  - Board representation with solvability checking
  - A* algorithm implementation
  - Manhattan distance heuristic
- Unified CLI interface for all puzzle types
- Test suite using pytest

## Development Notes

1. **Language**: Python (managed with Rye)

2. **Project Structure**:
   - `src/puzzle_solver/`: Main package
     - `core/`: Base classes and interfaces
       - `base_board.py`: Abstract board interface
       - `base_solver.py`: Abstract solver interface
       - `base_vision.py`: Abstract vision interface
     - `puzzles/`: Puzzle implementations
       - `arrow/`: Arrow puzzle implementation
       - `fifteen/`: 15-puzzle implementation
     - `cli.py`: Unified command-line interface
     - `automation.py`: Screen automation utilities
   - `tests/`: Test suite

3. **Key Commands**:
   - Install dependencies: `rye sync`
   - Run tests: `rye run pytest tests/ -v`
   - Arrow puzzle commands:
     - Demo: `rye run python -m puzzle_solver arrow demo`
     - Solve: `rye run python -m puzzle_solver arrow solve`
     - Auto-solve: `rye run python -m puzzle_solver arrow auto-solve`
   - 15-puzzle commands:
     - Demo: `rye run python -m puzzle_solver fifteen demo`
     - Solve: `rye run python -m puzzle_solver fifteen solve`

4. **Algorithm Implementations**:
   - Arrow Puzzle:
     - Propagation method for solving rows sequentially
     - Hard/Expert mode with advanced encoding strategy
     - Verification of solution correctness
   - 15-Puzzle:
     - A* search algorithm
     - Manhattan distance heuristic
     - Solvability checking based on inversion count

## Puzzle Types

### Arrow Puzzle
Arrow puzzles in Exponential Idle are logic puzzles where:
- Board consists of cells with values 0-4
- Tapping a cell increments it and its orthogonal neighbors (mod 5)
- Goal is to make all cells equal to 1
- Uses Propagation algorithm and Hard/Expert mode strategies

### 15-Puzzle
Classic sliding puzzle where:
- 4x4 grid with numbers 1-15 and one empty space
- Numbers can slide into the empty space
- Goal is to arrange numbers in ascending order
- Uses A* algorithm for optimal solving

## Important Notes

- Always keep both CLAUDE.md and CLAUDE-JP.md synchronized
- When making updates or adding new features, ensure both files contain the same information
- CLAUDE-JP.md is the Japanese version of this file for Japanese-speaking developers
- **Development Diary**: When working, create subjective development diaries following the instructions in `/devlog/README.md`
- The image recognition feature for arrow puzzles is experimental and not tested with actual game
- When adding new puzzle types, follow the established pattern with base class inheritance