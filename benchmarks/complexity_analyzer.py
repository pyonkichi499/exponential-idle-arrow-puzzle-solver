"""Complexity analysis for puzzle solvers."""

import time
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict
from puzzle_solver.puzzles.arrow import ArrowBoard, ArrowSolver
from puzzle_solver.puzzles.fifteen import FifteenBoard, FifteenSolver


class ComplexityAnalyzer:
    """Analyze time and space complexity of puzzle solvers."""
    
    def __init__(self):
        """Initialize analyzer."""
        self.results = {}
    
    def analyze_arrow_solver_scaling(self, max_size: int = 20, step: int = 2):
        """Analyze how arrow solver scales with board size."""
        sizes = list(range(3, max_size + 1, step))
        times = []
        moves = []
        
        print("Analyzing Arrow Solver Scaling...")
        print("Size | Time (s) | Moves | Time/n²")
        print("-" * 40)
        
        for size in sizes:
            # Create random board
            board = ArrowBoard(size)
            np.random.seed(42)
            board.grid = np.random.randint(0, 5, size=(size, size))
            
            # Time solving
            solver = ArrowSolver(board)
            start = time.perf_counter()
            success = solver.solve("normal")
            elapsed = time.perf_counter() - start
            
            times.append(elapsed)
            move_count = len(solver.moves) if success else 0
            moves.append(move_count)
            
            # Normalized by n²
            normalized_time = elapsed / (size * size)
            
            print(f"{size:4d} | {elapsed:8.4f} | {move_count:5d} | {normalized_time:.6f}")
        
        self.results['arrow_scaling'] = {
            'sizes': sizes,
            'times': times,
            'moves': moves
        }
        
        return sizes, times, moves
    
    def analyze_fifteen_puzzle_complexity(self, max_shuffles: int = 100, step: int = 10):
        """Analyze 15 puzzle complexity vs shuffle count."""
        shuffle_counts = list(range(10, max_shuffles + 1, step))
        times = []
        solution_lengths = []
        nodes_explored = []
        
        print("\n\nAnalyzing 15 Puzzle Complexity...")
        print("Shuffles | Time (s) | Solution | Nodes")
        print("-" * 45)
        
        for shuffles in shuffle_counts:
            # Create board
            board = FifteenBoard()
            np.random.seed(42)
            board.shuffle(shuffles)
            
            # Time solving
            solver = FifteenSolver(board)
            start = time.perf_counter()
            success = solver.solve()
            elapsed = time.perf_counter() - start
            
            times.append(elapsed)
            
            if success:
                solution_lengths.append(len(solver.solution))
                # Try to get nodes explored if available
                nodes = getattr(solver, 'nodes_explored', 0)
                nodes_explored.append(nodes)
            else:
                solution_lengths.append(0)
                nodes_explored.append(0)
            
            print(f"{shuffles:8d} | {elapsed:8.4f} | {solution_lengths[-1]:8d} | {nodes_explored[-1]:5d}")
        
        self.results['fifteen_complexity'] = {
            'shuffles': shuffle_counts,
            'times': times,
            'solution_lengths': solution_lengths,
            'nodes_explored': nodes_explored
        }
        
        return shuffle_counts, times, solution_lengths
    
    def plot_results(self, save_path: str = None):
        """Plot complexity analysis results."""
        try:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
            
            # Arrow solver scaling
            if 'arrow_scaling' in self.results:
                data = self.results['arrow_scaling']
                sizes = data['sizes']
                times = data['times']
                
                # Time vs size
                ax1.plot(sizes, times, 'b-o')
                ax1.set_xlabel('Board Size (n)')
                ax1.set_ylabel('Time (seconds)')
                ax1.set_title('Arrow Solver: Time Complexity')
                ax1.grid(True)
                
                # Time/n² to check if O(n²)
                normalized = [t/(s*s) for t, s in zip(times, sizes)]
                ax2.plot(sizes, normalized, 'r-o')
                ax2.set_xlabel('Board Size (n)')
                ax2.set_ylabel('Time / n²')
                ax2.set_title('Arrow Solver: Normalized Time (checking O(n²))')
                ax2.grid(True)
            
            # 15 puzzle complexity
            if 'fifteen_complexity' in self.results:
                data = self.results['fifteen_complexity']
                shuffles = data['shuffles']
                times = data['times']
                lengths = data['solution_lengths']
                
                # Time vs shuffles
                ax3.plot(shuffles, times, 'g-o')
                ax3.set_xlabel('Number of Shuffles')
                ax3.set_ylabel('Time (seconds)')
                ax3.set_title('15 Puzzle: Time vs Complexity')
                ax3.grid(True)
                
                # Solution length vs shuffles
                ax4.plot(shuffles, lengths, 'm-o')
                ax4.set_xlabel('Number of Shuffles')
                ax4.set_ylabel('Solution Length')
                ax4.set_title('15 Puzzle: Solution Length vs Shuffles')
                ax4.grid(True)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path)
                print(f"\nPlot saved to {save_path}")
            else:
                plt.show()
                
        except ImportError:
            print("\nMatplotlib not available. Skipping plots.")
    
    def estimate_complexity(self):
        """Estimate big-O complexity from empirical data."""
        print("\n\nComplexity Estimation:")
        print("=" * 50)
        
        # Arrow solver
        if 'arrow_scaling' in self.results:
            data = self.results['arrow_scaling']
            sizes = np.array(data['sizes'])
            times = np.array(data['times'])
            
            # Fit polynomial to log-log plot
            log_sizes = np.log(sizes)
            log_times = np.log(times)
            
            # Linear regression on log-log
            coeffs = np.polyfit(log_sizes, log_times, 1)
            exponent = coeffs[0]
            
            print(f"\nArrow Solver:")
            print(f"  Estimated complexity: O(n^{exponent:.2f})")
            print(f"  Expected: O(n²) for propagation algorithm")
            
            if abs(exponent - 2) < 0.3:
                print("  ✓ Matches expected O(n²) complexity")
            else:
                print("  ✗ Does not match expected complexity")
        
        # 15 puzzle
        if 'fifteen_complexity' in self.results:
            print(f"\n15 Puzzle:")
            print(f"  Algorithm: A* with Manhattan distance heuristic")
            print(f"  Worst case: O(b^d) where b=branching factor, d=depth")
            print(f"  With good heuristic: Much better in practice")
    
    def run_full_analysis(self):
        """Run complete complexity analysis."""
        print("Running Complexity Analysis")
        print("=" * 50)
        
        # Analyze scaling
        self.analyze_arrow_solver_scaling(max_size=15, step=1)
        self.analyze_fifteen_puzzle_complexity(max_shuffles=50, step=5)
        
        # Estimate complexity
        self.estimate_complexity()
        
        # Plot results
        self.plot_results("complexity_analysis.png")
        
        return self.results


if __name__ == "__main__":
    analyzer = ComplexityAnalyzer()
    results = analyzer.run_full_analysis()