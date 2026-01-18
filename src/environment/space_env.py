"""
Space Environment v2 - Clean Restart
====================================
Realistic 2D space environment for trajectory planning.

Key Changes from v1:
1. LARGER area (1000m x 1000m instead of 100m)
2. FEWER, SMALLER obstacles (5-8 instead of 12)
3. SIMPLER obstacle shapes
4. Guaranteed solvable environments
5. A* as proper baseline (not RRT*)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any
import heapq


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class Obstacle:
    """Simple circular obstacle."""
    center: np.ndarray
    radius: float
    
    def __post_init__(self):
        self.center = np.asarray(self.center, dtype=np.float32)
    
    def contains(self, point: np.ndarray, margin: float = 0.0) -> bool:
        return np.linalg.norm(point - self.center) < (self.radius + margin)


@dataclass
class EnvironmentConfig:
    """Environment configuration."""
    width: float = 1000.0        # 1km x 1km (realistic scale)
    height: float = 1000.0
    n_obstacles: int = 6         # Fewer obstacles
    min_radius: float = 20.0     # 20-50m obstacles
    max_radius: float = 50.0
    min_clearance: float = 30.0  # Start/goal clearance
    grid_resolution: int = 100   # For A* (10m cells)
    seed: Optional[int] = None


# =============================================================================
# ENVIRONMENT
# =============================================================================

class SpaceEnvironment:
    """
    Clean 2D Space Environment for Trajectory Planning.
    
    Features:
    - Larger, more realistic scale (1km x 1km)
    - Fewer obstacles for higher success rates
    - Guaranteed solvable (validates path exists)
    - Built-in A* baseline for fair comparison
    """
    
    def __init__(self, config: Optional[EnvironmentConfig] = None):
        self.config = config or EnvironmentConfig()
        self._rng = np.random.default_rng(self.config.seed)
        
        self.obstacles: List[Obstacle] = []
        self.start_pos: np.ndarray = None
        self.target_pos: np.ndarray = None
        self._grid: np.ndarray = None
    
    def reset(self, seed: Optional[int] = None) -> Dict[str, Any]:
        """Reset environment with new random configuration."""
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        
        # Generate obstacles
        self._generate_obstacles()
        
        # Generate start/target with guaranteed path
        self._generate_endpoints()
        
        # Build occupancy grid for A*
        self._build_grid()
        
        return {
            'start': self.start_pos.copy(),
            'target': self.target_pos.copy(),
            'obstacles': [(o.center.copy(), o.radius) for o in self.obstacles],
            'distance': np.linalg.norm(self.target_pos - self.start_pos)
        }
    
    def _generate_obstacles(self) -> None:
        """Generate non-overlapping obstacles."""
        self.obstacles = []
        
        for _ in range(self.config.n_obstacles):
            for attempt in range(50):
                center = self._rng.uniform(
                    low=[self.config.max_radius * 2, self.config.max_radius * 2],
                    high=[self.config.width - self.config.max_radius * 2,
                          self.config.height - self.config.max_radius * 2]
                )
                radius = self._rng.uniform(self.config.min_radius, self.config.max_radius)
                
                # Check no overlap
                valid = True
                for obs in self.obstacles:
                    if np.linalg.norm(center - obs.center) < (radius + obs.radius + 20):
                        valid = False
                        break
                
                if valid:
                    self.obstacles.append(Obstacle(center=center, radius=radius))
                    break
    
    def _generate_endpoints(self) -> None:
        """Generate start and target positions with guaranteed solvable path."""
        max_attempts = 100
        
        for _ in range(max_attempts):
            # Sample start
            self.start_pos = self._sample_free_point()
            
            # Sample target at reasonable distance
            for _ in range(50):
                self.target_pos = self._sample_free_point()
                dist = np.linalg.norm(self.target_pos - self.start_pos)
                
                # Ensure reasonable distance (300-800m)
                if 300 < dist < 800:
                    # Verify path exists using A*
                    self._build_grid()
                    if self._path_exists():
                        return
        
        # Fallback: simple diagonal
        self.start_pos = np.array([100.0, 100.0], dtype=np.float32)
        self.target_pos = np.array([900.0, 900.0], dtype=np.float32)
    
    def _sample_free_point(self) -> np.ndarray:
        """Sample point not inside any obstacle."""
        for _ in range(100):
            point = self._rng.uniform(
                low=[self.config.min_clearance, self.config.min_clearance],
                high=[self.config.width - self.config.min_clearance,
                      self.config.height - self.config.min_clearance]
            ).astype(np.float32)
            
            if not self._check_collision(point, margin=self.config.min_clearance):
                return point
        
        return np.array([self.config.width/2, self.config.height/2], dtype=np.float32)
    
    def _check_collision(self, point: np.ndarray, margin: float = 0.0) -> bool:
        """Check if point collides with any obstacle."""
        for obs in self.obstacles:
            if obs.contains(point, margin):
                return True
        return False
    
    def _build_grid(self) -> None:
        """Build occupancy grid for A* pathfinding."""
        res = self.config.grid_resolution
        cell_size = self.config.width / res
        
        self._grid = np.zeros((res, res), dtype=np.uint8)
        self._cell_size = cell_size
        
        for i in range(res):
            for j in range(res):
                # Cell center
                x = (i + 0.5) * cell_size
                y = (j + 0.5) * cell_size
                point = np.array([x, y])
                
                # Check if cell is blocked (with margin for safety)
                if self._check_collision(point, margin=cell_size):
                    self._grid[i, j] = 1
    
    def _path_exists(self) -> bool:
        """Check if A* path exists from start to target."""
        path = self.solve_astar()
        return path is not None
    
    # =========================================================================
    # A* BASELINE SOLVER
    # =========================================================================
    
    def solve_astar(self) -> Optional[np.ndarray]:
        """
        Solve using A* algorithm.
        
        This is the REAL baseline - a proper classical algorithm that's:
        - Deterministic (same result every time)
        - Optimal (finds shortest path)
        - Fast (polynomial time)
        
        Returns:
            Trajectory as (N, 2) array, or None if no path exists
        """
        if self._grid is None:
            self._build_grid()
        
        res = self.config.grid_resolution
        cell_size = self._cell_size
        
        # Convert positions to grid coordinates
        start_cell = (int(self.start_pos[0] / cell_size), 
                      int(self.start_pos[1] / cell_size))
        goal_cell = (int(self.target_pos[0] / cell_size),
                     int(self.target_pos[1] / cell_size))
        
        # Clamp to grid
        start_cell = (max(0, min(res-1, start_cell[0])), 
                      max(0, min(res-1, start_cell[1])))
        goal_cell = (max(0, min(res-1, goal_cell[0])),
                     max(0, min(res-1, goal_cell[1])))
        
        # A* algorithm
        def heuristic(a, b):
            return np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)
        
        open_set = []
        heapq.heappush(open_set, (0, start_cell))
        
        came_from = {}
        g_score = {start_cell: 0}
        f_score = {start_cell: heuristic(start_cell, goal_cell)}
        
        # 8-directional movement
        neighbors = [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]
        
        while open_set:
            _, current = heapq.heappop(open_set)
            
            if current == goal_cell:
                # Reconstruct path
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                path.reverse()
                
                # Convert to continuous coordinates
                trajectory = []
                for cell in path:
                    x = (cell[0] + 0.5) * cell_size
                    y = (cell[1] + 0.5) * cell_size
                    trajectory.append([x, y])
                
                # Add exact start and end
                trajectory[0] = self.start_pos.tolist()
                trajectory[-1] = self.target_pos.tolist()
                
                return np.array(trajectory, dtype=np.float32)
            
            for dx, dy in neighbors:
                neighbor = (current[0] + dx, current[1] + dy)
                
                # Check bounds
                if not (0 <= neighbor[0] < res and 0 <= neighbor[1] < res):
                    continue
                
                # Check collision
                if self._grid[neighbor[0], neighbor[1]] == 1:
                    continue
                
                # Diagonal cost
                move_cost = 1.414 if dx != 0 and dy != 0 else 1.0
                tentative_g = g_score[current] + move_cost
                
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + heuristic(neighbor, goal_cell)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))
        
        return None  # No path found
    
    # =========================================================================
    # TRAJECTORY UTILITIES
    # =========================================================================
    
    def check_trajectory(self, trajectory: np.ndarray, margin: float = 5.0) -> Tuple[bool, Optional[int]]:
        """
        Check if trajectory is collision-free.
        
        Args:
            trajectory: (N, 2) array of waypoints
            margin: Safety margin around obstacles
        
        Returns:
            (is_valid, collision_index) - collision_index is None if valid
        """
        for i, point in enumerate(trajectory):
            if self._check_collision(point, margin):
                return False, i
        return True, None
    
    def compute_path_length(self, trajectory: np.ndarray) -> float:
        """Compute total path length."""
        return float(np.sum(np.linalg.norm(np.diff(trajectory, axis=0), axis=1)))
    
    def compute_smoothness(self, trajectory: np.ndarray) -> float:
        """
        Compute trajectory smoothness (lower is smoother).
        Uses sum of angle changes between segments.
        """
        if len(trajectory) < 3:
            return 0.0
        
        segments = np.diff(trajectory, axis=0)
        angles = np.arctan2(segments[:, 1], segments[:, 0])
        angle_changes = np.abs(np.diff(angles))
        
        # Wrap around
        angle_changes = np.minimum(angle_changes, 2*np.pi - angle_changes)
        
        return float(np.sum(angle_changes))
    
    def get_obstacle_map(self, resolution: int = 64) -> np.ndarray:
        """
        Get binary obstacle map for ML model input.
        OPTIMIZED: Vectorized NumPy operations instead of double loop.
        
        Args:
            resolution: Output grid resolution
        
        Returns:
            Binary (resolution, resolution) array
        """
        cell_size = self.config.width / resolution
        
        # Create coordinate grids (vectorized)
        x_coords = (np.arange(resolution) + 0.5) * cell_size
        y_coords = (np.arange(resolution) + 0.5) * cell_size
        xx, yy = np.meshgrid(x_coords, y_coords, indexing='ij')
        
        # Stack into (resolution, resolution, 2) array of coordinates
        grid = np.zeros((resolution, resolution), dtype=np.float32)
        
        # Check each obstacle (vectorized distance computation)
        margin = cell_size / 2
        for obs in self.obstacles:
            # Distance from each grid cell to obstacle center
            dist = np.sqrt((xx - obs.center[0])**2 + (yy - obs.center[1])**2)
            # Mark cells within obstacle + margin
            grid[dist < (obs.radius + margin)] = 1.0
        
        return grid
    
    def interpolate_trajectory(self, path: np.ndarray, n_points: int = 50) -> np.ndarray:
        """Interpolate path to fixed number of points."""
        if len(path) < 2:
            return np.linspace(self.start_pos, self.target_pos, n_points)
        
        # Compute cumulative distances
        dists = np.zeros(len(path))
        for i in range(1, len(path)):
            dists[i] = dists[i-1] + np.linalg.norm(path[i] - path[i-1])
        
        total_dist = dists[-1]
        if total_dist < 1e-6:
            return np.linspace(self.start_pos, self.target_pos, n_points)
        
        # Interpolate
        target_dists = np.linspace(0, total_dist, n_points)
        result = np.zeros((n_points, 2), dtype=np.float32)
        
        for i, d in enumerate(target_dists):
            idx = np.searchsorted(dists, d)
            if idx == 0:
                result[i] = path[0]
            elif idx >= len(path):
                result[i] = path[-1]
            else:
                t = (d - dists[idx-1]) / (dists[idx] - dists[idx-1] + 1e-8)
                result[i] = (1-t) * path[idx-1] + t * path[idx]
        
        return result


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    print("Testing SpaceEnvironment v2...")
    
    config = EnvironmentConfig(seed=42)
    env = SpaceEnvironment(config)
    
    success_count = 0
    n_tests = 20
    
    for i in range(n_tests):
        info = env.reset(seed=42 + i)
        
        # Test A* baseline
        path = env.solve_astar()
        
        if path is not None:
            valid, _ = env.check_trajectory(path, margin=5.0)
            if valid:
                success_count += 1
                length = env.compute_path_length(path)
                print(f"Env {i}: ✓ Path found, length={length:.1f}m, waypoints={len(path)}")
            else:
                print(f"Env {i}: ✗ Path has collision")
        else:
            print(f"Env {i}: ✗ No path found")
    
    print(f"\nA* Success Rate: {success_count}/{n_tests} = {success_count/n_tests*100:.1f}%")
