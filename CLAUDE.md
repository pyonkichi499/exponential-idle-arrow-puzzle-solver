# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is the "exponential-idle-arrow-puzzle-solver" project - a library designed to solve arrow puzzles that appear in the mobile game "Exponential Idle".

## Current State

The project has been implemented with the following components:
- Board representation for arrow puzzles
- Solver implementing the Propagation algorithm
- Support for Hard/Expert mode solving
- CLI interface for solving puzzles
- Test suite using pytest

## Development Notes

1. **Language**: Python (managed with Rye)

2. **Project Structure**:
   - `src/arrow_puzzle_solver/`: Main package
     - `board.py`: Board representation
     - `solver.py`: Solving algorithms
     - `cli.py`: Command-line interface
   - `tests/`: Test suite

3. **Key Commands**:
   - Install dependencies: `rye sync`
   - Run tests: `rye run pytest tests/ -v`
   - Run CLI: `rye run python -m arrow_puzzle_solver [command]`
   - Demo: `rye run python -m arrow_puzzle_solver demo`

4. **Algorithm Implementation**:
   - Propagation method for solving rows sequentially
   - Hard/Expert mode with advanced encoding strategy
   - Verification of solution correctness

## Arrow Puzzle Context

Arrow puzzles in Exponential Idle are logic puzzles where:
- Board consists of cells with values 0-4
- Tapping a cell increments it and its orthogonal neighbors (mod 5)
- Goal is to make all cells equal to 1
- Uses Propagation algorithm and Hard/Expert mode strategies

## Important Notes

- Always keep both CLAUDE.md and CLAUDE-JP.md synchronized
- When making updates or adding new features, ensure both files contain the same information
- CLAUDE-JP.md is the Japanese version of this file for Japanese-speaking developers