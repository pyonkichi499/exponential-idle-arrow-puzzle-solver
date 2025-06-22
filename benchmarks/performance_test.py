"""Performance benchmarks for puzzle solvers."""

import time
import numpy as np
import statistics
from typing import Dict, List, Tuple
from puzzle_solver.puzzles.arrow import ArrowBoard, ArrowSolver
from puzzle_solver.puzzles.fifteen import FifteenBoard, FifteenSolver


class PerformanceBenchmark:
    """Run performance benchmarks for puzzle solvers."""
    
    def __init__(self, runs: int = 10):
        """Initialize benchmark with number of runs."""
        self.runs = runs
        self.results: Dict[str, List[float]] = {}
    
    def time_function(self, func, *args, **kwargs) -> float:
        """Time a single function execution."""
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        return end - start, result
    
    def benchmark_arrow_solver(self, sizes: List[int] = [5, 7, 10]) -> Dict[str, Dict]:
        """Benchmark arrow puzzle solver with different sizes."""
        results = {}
        
        for size in sizes:
            print(f"\n=== Arrow Puzzle {size}x{size} ===")
            times = []
            moves_counts = []
            success_count = 0
            
            for run in range(self.runs):
                # Generate random board
                board = ArrowBoard(size)
                np.random.seed(42 + run)  # Reproducible
                board.grid = np.random.randint(0, 5, size=(size, size))
                
                # Time the solving
                solver = ArrowSolver(board)
                elapsed, success = self.time_function(solver.solve, mode="normal")
                
                times.append(elapsed)
                if success:
                    success_count += 1
                    moves_counts.append(len(solver.moves))
            
            results[f"arrow_{size}x{size}"] = {
                "avg_time": statistics.mean(times),
                "std_time": statistics.stdev(times) if len(times) > 1 else 0,
                "min_time": min(times),
                "max_time": max(times),
                "success_rate": success_count / self.runs,
                "avg_moves": statistics.mean(moves_counts) if moves_counts else 0,
                "size": size
            }
            
            self._print_results(f"Arrow {size}x{size}", results[f"arrow_{size}x{size}"])
        
        return results
    
    def benchmark_fifteen_solver(self, shuffle_counts: List[int] = [10, 20, 50]) -> Dict[str, Dict]:
        """Benchmark 15 puzzle solver with different complexities."""
        results = {}
        
        for shuffles in shuffle_counts:
            print(f"\n=== 15 Puzzle (shuffled {shuffles} times) ===")
            times = []
            moves_counts = []
            nodes_explored = []
            
            for run in range(self.runs):
                # Create shuffled board
                board = FifteenBoard()
                np.random.seed(42 + run)  # Reproducible
                board.shuffle(shuffles)
                
                # Time the solving
                solver = FifteenSolver(board)
                elapsed, success = self.time_function(solver.solve)
                
                times.append(elapsed)
                if success:
                    moves_counts.append(len(solver.solution))
                    if hasattr(solver, 'nodes_explored'):
                        nodes_explored.append(solver.nodes_explored)
            
            results[f"fifteen_{shuffles}_shuffles"] = {
                "avg_time": statistics.mean(times),
                "std_time": statistics.stdev(times) if len(times) > 1 else 0,
                "min_time": min(times),
                "max_time": max(times),
                "avg_moves": statistics.mean(moves_counts) if moves_counts else 0,
                "avg_nodes": statistics.mean(nodes_explored) if nodes_explored else 0,
                "shuffle_count": shuffles
            }
            
            self._print_results(f"15 Puzzle ({shuffles} shuffles)", 
                              results[f"fifteen_{shuffles}_shuffles"])
        
        return results
    
    def _print_results(self, name: str, results: Dict):
        """Print benchmark results in a nice format."""
        print(f"{name} Results:")
        print(f"  Average time: {results['avg_time']:.4f}s ± {results['std_time']:.4f}s")
        print(f"  Min/Max time: {results['min_time']:.4f}s / {results['max_time']:.4f}s")
        if 'success_rate' in results:
            print(f"  Success rate: {results['success_rate']*100:.1f}%")
        if 'avg_moves' in results and results['avg_moves'] > 0:
            print(f"  Average moves: {results['avg_moves']:.1f}")
        if 'avg_nodes' in results and results['avg_nodes'] > 0:
            print(f"  Average nodes explored: {results['avg_nodes']:.0f}")
    
    def compare_solver_modes(self):
        """Compare different solving modes for arrow puzzle."""
        print("\n=== Comparing Arrow Solver Modes ===")
        size = 7
        modes = ["normal", "hard", "expert"]
        
        # Generate a challenging board
        board = ArrowBoard(size)
        np.random.seed(100)
        board.grid = np.random.randint(0, 5, size=(size, size))
        
        for mode in modes:
            times = []
            successes = 0
            
            for _ in range(self.runs):
                solver = ArrowSolver(board.copy())
                elapsed, success = self.time_function(solver.solve, mode=mode)
                times.append(elapsed)
                if success:
                    successes += 1
            
            print(f"\n{mode.capitalize()} mode:")
            print(f"  Average time: {statistics.mean(times):.4f}s")
            print(f"  Success rate: {successes}/{self.runs}")
    
    def run_all_benchmarks(self):
        """Run all benchmarks and return results."""
        print("Running Puzzle Solver Benchmarks")
        print("=" * 50)
        
        # Arrow puzzle benchmarks
        arrow_results = self.benchmark_arrow_solver()
        
        # 15 puzzle benchmarks
        fifteen_results = self.benchmark_fifteen_solver()
        
        # Mode comparison
        self.compare_solver_modes()
        
        # Summary
        print("\n" + "=" * 50)
        print("BENCHMARK SUMMARY")
        print("=" * 50)
        
        all_results = {**arrow_results, **fifteen_results}
        
        # Find fastest/slowest
        fastest = min(all_results.items(), key=lambda x: x[1]['avg_time'])
        slowest = max(all_results.items(), key=lambda x: x[1]['avg_time'])
        
        print(f"Fastest: {fastest[0]} ({fastest[1]['avg_time']:.4f}s avg)")
        print(f"Slowest: {slowest[0]} ({slowest[1]['avg_time']:.4f}s avg)")
        
        return all_results


if __name__ == "__main__":
    benchmark = PerformanceBenchmark(runs=5)
    results = benchmark.run_all_benchmarks()