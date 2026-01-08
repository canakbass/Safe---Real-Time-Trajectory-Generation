"""
RRT* Solver Module
==================
Classical Rapidly-exploring Random Tree Star (RRT*) implementation
for baseline trajectory generation.

This serves as Solver A (Baseline) in the benchmark comparison.
Computationally intensive but reliable pathfinding.
"""

from __future__ import annotations

import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass

from .base_solver import (
    BaseSolver,
    SolverResult,
    SolverType,
    SolverTimer,
    TimingRecord,
)


# =============================================================================
# RRT* DATA STRUCTURES
# =============================================================================

@dataclass
class RRTNode:
    """
    Node in the RRT* tree.
    
    Attributes:
        position: (x, y) position in environment coordinates
        parent: Parent node index in the tree
        cost: Cost-to-come from root to this node
    """
    position: np.ndarray
    parent: Optional[int] = None
    cost: float = 0.0
    
    def __post_init__(self):
        self.position = np.asarray(self.position, dtype=np.float32)


@dataclass
class RRTConfig:
    """
    Configuration parameters for RRT* algorithm.
    
    Attributes:
        max_iterations: Maximum tree expansion iterations
        step_size: Maximum extension distance per iteration (m)
        goal_bias: Probability of sampling goal instead of random point
        goal_tolerance: Distance threshold for goal reaching (m)
        rewire_radius: Radius for rewiring neighbors
        collision_resolution: Step size for collision checking along edges (m)
    """
    max_iterations: int = 5000
    step_size: float = 5.0
    goal_bias: float = 0.10
    goal_tolerance: float = 3.0
    rewire_radius: float = 15.0
    collision_resolution: float = 0.5


# =============================================================================
# RRT* SOLVER IMPLEMENTATION
# =============================================================================

class RRTSolver(BaseSolver):
    """
    RRT* (Rapidly-exploring Random Tree Star) Trajectory Solver.
    
    Implementation of the asymptotically optimal RRT* algorithm for
    spacecraft path planning. Uses informed sampling and rewiring
    to converge toward optimal paths.
    
    Performance Characteristics:
        - CPU-bound operation (no GPU usage)
        - Time complexity: O(n log n) per iteration
        - Space complexity: O(n) nodes
    
    Reference:
        Karaman, S., & Frazzoli, E. (2011). Sampling-based algorithms 
        for optimal motion planning. IJRR, 30(7), 846-894.
    
    Example:
        >>> solver = RRTSolver(config=RRTConfig(max_iterations=3000))
        >>> result = solver.solve(env)
        >>> print(f"Path length: {result.path_length:.2f}m")
    """
    
    def __init__(
        self,
        config: Optional[RRTConfig] = None,
        n_waypoints: int = 50
    ):
        """
        Initialize RRT* Solver.
        
        Args:
            config: RRT* configuration parameters
            n_waypoints: Target number of waypoints in output trajectory
        """
        super().__init__(
            name="RRT* Classical Solver",
            solver_type=SolverType.CLASSICAL_RRT,
            n_waypoints=n_waypoints
        )
        
        self.config = config or RRTConfig()
        
        # Tree storage
        self._nodes: List[RRTNode] = []
        self._env = None
        self._rng = np.random.default_rng()
    
    def solve(self, env: 'SpaceEnv') -> SolverResult:
        """
        Generate trajectory using RRT* algorithm.
        
        Algorithm:
            1. Initialize tree with start node
            2. For each iteration:
               a. Sample random point (with goal bias)
               b. Find nearest node in tree
               c. Steer toward sample
               d. Check collision along edge
               e. Find best parent among neighbors
               f. Rewire neighbors if beneficial
            3. Extract path from start to goal
        
        Args:
            env: SpaceEnv instance with start, target, obstacles
        
        Returns:
            SolverResult with trajectory and timing
        """
        self._env = env
        self._nodes = []
        self._rng = np.random.default_rng()
        
        timer = SolverTimer()
        timer.start()
        
        with timer.cpu_section("rrt_planning"):
            # Initialize tree with start node
            start_node = RRTNode(position=env.start_pos.copy(), parent=None, cost=0.0)
            self._nodes.append(start_node)
            
            goal_node_idx: Optional[int] = None
            best_goal_cost = float('inf')
            
            # Main RRT* loop
            for iteration in range(self.config.max_iterations):
                # Sample point
                if self._rng.random() < self.config.goal_bias:
                    sample = env.target_pos.copy()
                else:
                    sample = self._sample_free_space()
                
                # Find nearest node
                nearest_idx = self._find_nearest(sample)
                nearest_node = self._nodes[nearest_idx]
                
                # Steer toward sample
                new_pos = self._steer(nearest_node.position, sample)
                
                # Check if edge is collision-free
                if not self._edge_collision_free(nearest_node.position, new_pos):
                    continue
                
                # Find neighbors for potential rewiring
                neighbor_indices = self._find_neighbors(new_pos)
                
                # Find best parent
                best_parent_idx = nearest_idx
                best_cost = nearest_node.cost + self._distance(nearest_node.position, new_pos)
                
                for n_idx in neighbor_indices:
                    neighbor = self._nodes[n_idx]
                    candidate_cost = neighbor.cost + self._distance(neighbor.position, new_pos)
                    
                    if candidate_cost < best_cost:
                        if self._edge_collision_free(neighbor.position, new_pos):
                            best_parent_idx = n_idx
                            best_cost = candidate_cost
                
                # Add new node
                new_node = RRTNode(
                    position=new_pos,
                    parent=best_parent_idx,
                    cost=best_cost
                )
                new_node_idx = len(self._nodes)
                self._nodes.append(new_node)
                
                # Rewire neighbors
                for n_idx in neighbor_indices:
                    neighbor = self._nodes[n_idx]
                    rewire_cost = best_cost + self._distance(new_pos, neighbor.position)
                    
                    if rewire_cost < neighbor.cost:
                        if self._edge_collision_free(new_pos, neighbor.position):
                            neighbor.parent = new_node_idx
                            neighbor.cost = rewire_cost
                            self._propagate_cost_update(n_idx)
                
                # Check if goal reached
                goal_dist = self._distance(new_pos, env.target_pos)
                if goal_dist < self.config.goal_tolerance:
                    if best_cost + goal_dist < best_goal_cost:
                        goal_node_idx = new_node_idx
                        best_goal_cost = best_cost + goal_dist
            
            # Extract path
            if goal_node_idx is not None:
                raw_path = self._extract_path(goal_node_idx)
                # Ensure path ends exactly at target
                raw_path = np.vstack([raw_path, env.target_pos])
                trajectory = self.interpolate_trajectory(raw_path, self.n_waypoints)
                success = True
            else:
                # Return straight line as fallback (will likely fail collision check)
                trajectory = np.linspace(env.start_pos, env.target_pos, self.n_waypoints)
                success = False
        
        timing = timer.get_timing()
        
        return SolverResult(
            trajectory=trajectory,
            success=success,
            timing=timing,
            solver_type=self.solver_type,
            metadata={
                'tree_size': len(self._nodes),
                'goal_cost': best_goal_cost if success else None,
                'iterations': self.config.max_iterations
            }
        )
    
    # =========================================================================
    # HELPER METHODS
    # =========================================================================
    
    def _sample_free_space(self) -> np.ndarray:
        """Sample a random point in the environment."""
        return self._rng.uniform(
            low=[0, 0],
            high=[self._env.config.width, self._env.config.height]
        ).astype(np.float32)
    
    def _find_nearest(self, point: np.ndarray) -> int:
        """Find index of nearest node to point."""
        positions = np.array([n.position for n in self._nodes])
        distances = np.linalg.norm(positions - point, axis=1)
        return int(np.argmin(distances))
    
    def _find_neighbors(self, point: np.ndarray) -> List[int]:
        """Find all nodes within rewire radius of point."""
        neighbors = []
        for i, node in enumerate(self._nodes):
            if self._distance(node.position, point) < self.config.rewire_radius:
                neighbors.append(i)
        return neighbors
    
    def _steer(self, from_pos: np.ndarray, to_pos: np.ndarray) -> np.ndarray:
        """Steer from one position toward another, limited by step size."""
        direction = to_pos - from_pos
        dist = np.linalg.norm(direction)
        
        if dist < self.config.step_size:
            return to_pos.copy()
        
        return from_pos + (direction / dist) * self.config.step_size
    
    def _distance(self, p1: np.ndarray, p2: np.ndarray) -> float:
        """Compute Euclidean distance between two points."""
        return float(np.linalg.norm(p1 - p2))
    
    def _edge_collision_free(self, p1: np.ndarray, p2: np.ndarray) -> bool:
        """Check if edge from p1 to p2 is collision-free."""
        direction = p2 - p1
        dist = np.linalg.norm(direction)
        
        if dist < 1e-6:
            return not self._env._check_collision(p1)
        
        n_checks = int(np.ceil(dist / self.config.collision_resolution))
        
        for i in range(n_checks + 1):
            t = i / n_checks
            point = p1 + t * direction
            
            if self._env._check_collision(point):
                return False
        
        return True
    
    def _extract_path(self, goal_idx: int) -> np.ndarray:
        """Extract path from root to goal node."""
        path = []
        current_idx = goal_idx
        
        while current_idx is not None:
            path.append(self._nodes[current_idx].position)
            current_idx = self._nodes[current_idx].parent
        
        path.reverse()
        return np.array(path)
    
    def _propagate_cost_update(self, node_idx: int) -> None:
        """Propagate cost updates to descendants after rewiring."""
        # Simple BFS to update children
        stack = [node_idx]
        
        while stack:
            idx = stack.pop()
            node = self._nodes[idx]
            
            # Find children
            for i, n in enumerate(self._nodes):
                if n.parent == idx:
                    parent_node = self._nodes[n.parent]
                    n.cost = parent_node.cost + self._distance(parent_node.position, n.position)
                    stack.append(i)


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    'RRTSolver',
    'RRTConfig',
    'RRTNode',
]
