"""Solver for 15 puzzle using A* algorithm."""

import heapq
from typing import List, Tuple, Optional, Set
from dataclasses import dataclass, field
from ...core import BaseSolver
from .board import FifteenBoard


@dataclass(order=True)
class Node:
    """Node in the search tree."""
    priority: int
    board: FifteenBoard = field(compare=False)
    moves: List[str] = field(default_factory=list, compare=False)
    g_score: int = field(default=0, compare=False)  # Cost from start
    h_score: int = field(default=0, compare=False)  # Heuristic to goal


class FifteenSolver(BaseSolver):
    """Solver for 15 puzzle using A* algorithm."""

    def __init__(self, board: FifteenBoard):
        """Initialize solver with a board.

        Args:
            board: The board to solve
        """
        super().__init__(board)
        self.board: FifteenBoard = board.copy()
        self.original_board: FifteenBoard = board.copy()
        self.solution_moves: List[str] = []
        self.nodes_explored = 0
        self.max_depth = 100  # Maximum search depth

    def heuristic(self, board: FifteenBoard) -> int:
        """Calculate heuristic value for A* search.
        
        Uses Manhattan distance as the heuristic.
        """
        return board.manhattan_distance()

    def solve_astar(self) -> bool:
        """Solve using A* algorithm.
        
        Returns:
            True if solution found, False otherwise
        """
        # Check if puzzle is solvable
        if not self.board.is_solvable():
            self.stats["solvable"] = False
            return False

        # Initialize priority queue
        start_node = Node(
            priority=self.heuristic(self.board),
            board=self.board.copy(),
            moves=[],
            g_score=0,
            h_score=self.heuristic(self.board)
        )
        
        open_set = [start_node]
        closed_set: Set[str] = set()
        
        while open_set and self.nodes_explored < 100000:  # Limit nodes to prevent infinite search
            # Get node with lowest f_score
            current = heapq.heappop(open_set)
            self.nodes_explored += 1
            
            # Check if we reached the goal
            if current.board.is_solved():
                self.solution_moves = current.moves
                return True
            
            # Skip if we've seen this state
            state_hash = current.board.get_state_hash()
            if state_hash in closed_set:
                continue
            closed_set.add(state_hash)
            
            # Don't explore beyond maximum depth
            if len(current.moves) >= self.max_depth:
                continue
            
            # Explore neighbors
            for move in current.board.get_legal_moves():
                # Create new board with move applied
                new_board = current.board.copy()
                new_board.apply_move(move)
                
                # Skip if we've seen this state
                new_state_hash = new_board.get_state_hash()
                if new_state_hash in closed_set:
                    continue
                
                # Create new node
                new_g_score = current.g_score + 1
                new_h_score = self.heuristic(new_board)
                new_node = Node(
                    priority=new_g_score + new_h_score,
                    board=new_board,
                    moves=current.moves + [move],
                    g_score=new_g_score,
                    h_score=new_h_score
                )
                
                heapq.heappush(open_set, new_node)
        
        return False

    def solve_iddfs(self, max_depth: int = 50) -> bool:
        """Solve using Iterative Deepening Depth-First Search.
        
        Alternative to A* for memory-constrained situations.
        
        Args:
            max_depth: Maximum depth to search
            
        Returns:
            True if solution found, False otherwise
        """
        # Check if puzzle is solvable
        if not self.board.is_solvable():
            self.stats["solvable"] = False
            return False

        for depth in range(1, max_depth + 1):
            result = self._dfs(self.board.copy(), [], depth, set())
            if result is not None:
                self.solution_moves = result
                return True
        
        return False

    def _dfs(self, board: FifteenBoard, moves: List[str], depth: int, visited: Set[str]) -> Optional[List[str]]:
        """Depth-first search helper."""
        if depth == 0:
            return None
        
        if board.is_solved():
            return moves
        
        state_hash = board.get_state_hash()
        if state_hash in visited:
            return None
        visited.add(state_hash)
        
        for move in board.get_legal_moves():
            new_board = board.copy()
            new_board.apply_move(move)
            
            result = self._dfs(new_board, moves + [move], depth - 1, visited)
            if result is not None:
                return result
        
        return None

    def solve(self, algorithm: str = "astar", **kwargs) -> bool:
        """Solve the puzzle.
        
        Args:
            algorithm: Algorithm to use ("astar" or "iddfs")
            **kwargs: Additional parameters
            
        Returns:
            True if solution found, False otherwise
        """
        self.stats["algorithm"] = algorithm
        self.stats["board_size"] = self.board.size
        self.nodes_explored = 0
        
        if algorithm == "astar":
            result = self.solve_astar()
        elif algorithm == "iddfs":
            max_depth = kwargs.get("max_depth", 50)
            result = self.solve_iddfs(max_depth)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
        
        self.stats["nodes_explored"] = self.nodes_explored
        self.stats["solution_length"] = len(self.solution_moves)
        self.stats["solved"] = result
        self.solution = self.solution_moves
        
        return result

    def get_solution(self) -> List[str]:
        """Get the solution moves."""
        return self.solution_moves.copy()

    def get_solution_steps(self) -> List[str]:
        """Get human-readable solution steps."""
        steps = []
        opposite = {"up": "down", "down": "up", "left": "right", "right": "left"}
        
        for i, move in enumerate(self.solution_moves):
            # In UI, we describe moving tiles, not the empty space
            tile_direction = opposite[move]
            steps.append(f"Step {i+1}: Move tile {tile_direction}")
        
        return steps

    def verify_solution(self) -> bool:
        """Verify that the solution is correct."""
        test_board = self.original_board.copy()
        
        # Apply all moves
        for move in self.solution_moves:
            if move not in test_board.get_legal_moves():
                return False
            test_board.apply_move(move)
        
        return test_board.is_solved()

    def get_algorithm_name(self) -> str:
        """Get the name of the solving algorithm."""
        return "A* Search with Manhattan Distance Heuristic"

    def estimate_difficulty(self) -> str:
        """Estimate the difficulty of the puzzle."""
        # Based on Manhattan distance and solvability
        if not self.original_board.is_solvable():
            return "impossible"
        
        distance = self.original_board.manhattan_distance()
        
        if distance < 10:
            return "easy"
        elif distance < 20:
            return "medium"
        elif distance < 30:
            return "hard"
        else:
            return "very hard"