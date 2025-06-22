"""Memory usage profiling for puzzle solvers."""

import tracemalloc
import gc
from typing import Dict, Tuple, Any
import numpy as np
from puzzle_solver.puzzles.arrow import ArrowBoard, ArrowSolver
from puzzle_solver.puzzles.fifteen import FifteenBoard, FifteenSolver


class MemoryProfiler:
    """Profile memory usage of puzzle solvers."""
    
    def profile_function(self, func, *args, **kwargs) -> Tuple[Any, Dict[str, float]]:
        """Profile memory usage of a function call."""
        # Force garbage collection
        gc.collect()
        
        # Start tracing
        tracemalloc.start()
        snapshot1 = tracemalloc.take_snapshot()
        
        # Run function
        result = func(*args, **kwargs)
        
        # Take second snapshot
        snapshot2 = tracemalloc.take_snapshot()
        
        # Calculate differences
        stats = snapshot2.compare_to(snapshot1, 'lineno')
        
        # Get peak memory
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        memory_info = {
            'current_mb': current / 1024 / 1024,
            'peak_mb': peak / 1024 / 1024,
            'top_allocations': self._get_top_allocations(stats, 5)
        }
        
        return result, memory_info
    
    def _get_top_allocations(self, stats, limit=5):
        """Get top memory allocations."""
        top_stats = sorted(stats, key=lambda x: x.size_diff, reverse=True)[:limit]
        
        allocations = []
        for stat in top_stats:
            allocations.append({
                'file': stat.traceback[0].filename.split('/')[-1],
                'line': stat.traceback[0].lineno,
                'size_mb': stat.size_diff / 1024 / 1024
            })
        
        return allocations
    
    def profile_arrow_solver(self, sizes=[5, 7, 10, 15]):
        """Profile memory usage for different arrow puzzle sizes."""
        print("Memory Profile: Arrow Puzzle Solver")
        print("=" * 50)
        
        for size in sizes:
            # Create board
            board = ArrowBoard(size)
            np.random.seed(42)
            board.grid = np.random.randint(0, 5, size=(size, size))
            
            # Profile solving
            solver = ArrowSolver(board)
            _, memory_info = self.profile_function(solver.solve)
            
            print(f"\n{size}x{size} Arrow Puzzle:")
            print(f"  Current memory: {memory_info['current_mb']:.2f} MB")
            print(f"  Peak memory: {memory_info['peak_mb']:.2f} MB")
            
            if memory_info['top_allocations']:
                print("  Top allocations:")
                for alloc in memory_info['top_allocations'][:3]:
                    print(f"    {alloc['file']}:{alloc['line']} - {alloc['size_mb']:.3f} MB")
    
    def profile_fifteen_solver(self, complexities=[10, 20, 30]):
        """Profile memory usage for 15 puzzle with different complexities."""
        print("\n\nMemory Profile: 15 Puzzle Solver")
        print("=" * 50)
        
        for shuffles in complexities:
            # Create board
            board = FifteenBoard()
            np.random.seed(42)
            board.shuffle(shuffles)
            
            # Profile solving
            solver = FifteenSolver(board)
            _, memory_info = self.profile_function(solver.solve)
            
            print(f"\n15 Puzzle ({shuffles} shuffles):")
            print(f"  Current memory: {memory_info['current_mb']:.2f} MB")
            print(f"  Peak memory: {memory_info['peak_mb']:.2f} MB")
    
    def compare_data_structures(self):
        """Compare memory usage of different board sizes."""
        print("\n\nMemory Profile: Data Structure Comparison")
        print("=" * 50)
        
        sizes = [5, 10, 20, 50, 100]
        
        for size in sizes:
            # Profile ArrowBoard creation
            _, memory_info = self.profile_function(ArrowBoard, size)
            
            expected_mb = (size * size * 8) / 1024 / 1024  # 8 bytes per int64
            print(f"\nArrowBoard({size}x{size}):")
            print(f"  Actual memory: {memory_info['current_mb']:.3f} MB")
            print(f"  Expected (grid only): {expected_mb:.3f} MB")
            print(f"  Overhead: {memory_info['current_mb'] - expected_mb:.3f} MB")
    
    def run_all_profiles(self):
        """Run all memory profiling tests."""
        self.profile_arrow_solver()
        self.profile_fifteen_solver()
        self.compare_data_structures()
        
        print("\n" + "=" * 50)
        print("Memory profiling complete!")


if __name__ == "__main__":
    profiler = MemoryProfiler()
    profiler.run_all_profiles()