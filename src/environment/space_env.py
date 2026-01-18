
"""
Space Environment 3D (IAC 2026 Standards)
=========================================
3D space environment for trajectory planning on embedded systems (Jetson Nano).

Specs:
- 1000x1000x1000 coordinate system
- Analytic Sphere Obstacles
- Efficient 100^3 Grid Downsampling (Max Pooling)
- 3D A* Solver with 26-connectivity
"""

import numpy as np
import heapq
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any, Union

# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class Obstacle:
    """
    Analytic Spherical Obstacle.
    Stored analytically to save memory (vs dense grid).
    """
    center: np.ndarray  # Shape (3,)
    radius: float

    def __post_init__(self):
        self.center = np.asarray(self.center, dtype=np.float32)

    def contains(self, point: np.ndarray, margin: float = 0.0) -> bool:
        """Check collision with point (x,y,z)."""
        dist = np.linalg.norm(point - self.center)
        return dist < (self.radius + margin)

@dataclass
class EnvironmentConfig:
    """Environment configuration for 3D Space."""
    # Space Dimensions
    x_range: float = 1000.0
    y_range: float = 1000.0
    z_range: float = 1000.0
    
    # Model Input Grid (Downsampling)
    grid_dim: int = 100  # 100x100x100 grid for model input
    
    # Obstacles
    n_obstacles: int = 10
    min_radius: float = 50.0   # Larger obstacles for 3D
    max_radius: float = 100.0
    
    # Safety
    min_start_goal_dist: float = 400.0
    clearance: float = 20.0  # Margin
    
    # Random Seed
    seed: Optional[int] = None

# =============================================================================
# 3D SPACE ENVIRONMENT
# =============================================================================

class SpaceEnv:
    """
    3D Simulation Environment.
    Handles obstacle generation, efficient grid downsampling, and collision checks.
    """
    
    def __init__(self, config: Optional[EnvironmentConfig] = None):
        self.config = config or EnvironmentConfig()
        self._rng = np.random.default_rng(self.config.seed)
        
        self.obstacles: List[Obstacle] = []
        self.start: np.ndarray = np.zeros(3)
        self.goal: np.ndarray = np.zeros(3)
        
        # Computed properties
        self.resolution = np.array([
            self.config.x_range / self.config.grid_dim,
            self.config.y_range / self.config.grid_dim,
            self.config.z_range / self.config.grid_dim
        ], dtype=np.float32)

    def reset(self, seed: Optional[int] = None) -> Dict[str, Any]:
        """Reset scene with new obstacles and start/goal."""
        if seed is not None:
            self._rng = np.random.default_rng(seed)
            
        self._generate_obstacles()
        self._generate_endpoints()
        
        return {
            "start": self.start,
            "goal": self.goal,
            "obstacles": self.obstacles,
            "grid": self.get_downsampled_grid()
        }

    def _generate_obstacles(self):
        """Generate random spherical obstacles without strict overlap checking (for speed)."""
        self.obstacles = []
        for _ in range(self.config.n_obstacles):
            # Keep away from edges
            center = self._rng.uniform(
                low=[0, 0, 0],
                high=[self.config.x_range, self.config.y_range, self.config.z_range]
            )
            radius = self._rng.uniform(self.config.min_radius, self.config.max_radius)
            self.obstacles.append(Obstacle(center, radius))

    def _generate_endpoints(self):
        """Generate start and goal points ensuring they are free."""
        # Simple rejection sampling
        while True:
            self.start = self._sample_free_point()
            self.goal = self._sample_free_point()
            if np.linalg.norm(self.start - self.goal) > self.config.min_start_goal_dist:
                break

    def _sample_free_point(self) -> np.ndarray:
        for _ in range(100):
            pt = self._rng.uniform(
                low=[0, 0, 0],
                high=[self.config.x_range, self.config.y_range, self.config.z_range]
            )
            if not self.check_collision(pt, self.config.clearance):
                return pt.astype(np.float32)
        return np.array([0, 0, 0], dtype=np.float32) # Fallback

    def check_collision(self, point: np.ndarray, margin: float = 0.0) -> bool:
        """Analytic collision check (Exact)."""
        for obs in self.obstacles:
            if obs.contains(point, margin):
                return True
        return False

    def get_downsampled_grid(self) -> np.ndarray:
        """
        Generate 100x100x100 binary occupancy grid.
        Optimized using NumPy broadcasting (Max Pooling logic).
        """
        D = self.config.grid_dim
        
        # 1. Create voxel center coordinates (100, 100, 100, 3)
        rx = np.linspace(self.resolution[0]/2, self.config.x_range - self.resolution[0]/2, D)
        ry = np.linspace(self.resolution[1]/2, self.config.y_range - self.resolution[1]/2, D)
        rz = np.linspace(self.resolution[2]/2, self.config.z_range - self.resolution[2]/2, D)
        
        # Memory efficient: Process in batches or broadcast carefully?
        # 100^3 = 1M points. 1M * 3 (coords) * 4 bytes = 12MB. Lightweight.
        
        X, Y, Z = np.meshgrid(rx, ry, rz, indexing='ij')
        # Flatten to (N, 3) for vectorized dist calc
        grid_points = np.stack([X, Y, Z], axis=-1).reshape(-1, 3)
        
        occupancy = np.zeros(D * D * D, dtype=np.float32)
        
        # Vectorized check per obstacle
        # Logic: If voxel center is inside obstacle (mostly), mark 1.
        # To strictly satisfy "Max Pooling" (any part of block), we assume a conservative radius check.
        # We increase obstacle radius effectively by voxel_radius.
        voxel_radius = np.linalg.norm(self.resolution) / 2.0
        
        for obs in self.obstacles:
            # Distance from all grid points to this obstacle center
            # (N, 3) - (3,) -> (N, 3) -> norm -> (N,)
            dists = np.linalg.norm(grid_points - obs.center, axis=1)
            
            # Condition: dist < obs.radius + voxel_radius (Conservative / Max Pool effect)
            mask = dists < (obs.radius + voxel_radius)
            occupancy[mask] = 1.0
            
        return occupancy.reshape(D, D, D)

# =============================================================================
# 3D A* SOLVER
# =============================================================================

class AStarPlanner:
    """
    3D A* Solver.
    Uses 26-neighbor connectivity and Euclidean heuristic.
    Optimized with NumPy arrays for score tracking.
    """
    
    def __init__(self, env: SpaceEnv):
        self.env = env
        # Neighbor offsets (26 connectivity)
        self.neighbors = []
        for x in [-1, 0, 1]:
            for y in [-1, 0, 1]:
                for z in [-1, 0, 1]:
                    if x == 0 and y == 0 and z == 0:
                        continue
                    self.neighbors.append((x, y, z))
        self.neighbors = np.array(self.neighbors, dtype=np.int32)
    
    def solve(self, start_pos: np.ndarray, goal_pos: np.ndarray, timeout_steps: int = 100000) -> Optional[np.ndarray]:
        """
        Run A* on the grid graph.
        """
        # Discretize start/goal to grid indices
        res = self.env.resolution
        # Clamp indices to be safe
        D = self.env.config.grid_dim
        
        start_idx = np.floor(start_pos / res).astype(np.int32)
        goal_idx = np.floor(goal_pos / res).astype(np.int32)
        
        start_idx = np.clip(start_idx, 0, D-1)
        goal_idx = np.clip(goal_idx, 0, D-1)
        
        target = tuple(goal_idx)
        start_node = tuple(start_idx)
        
        if start_node == target:
            return np.array([start_pos, goal_pos], dtype=np.float32)

        # Get Occupancy Grid
        grid = self.env.get_downsampled_grid()
        if grid[start_node] > 0.5 or grid[target] > 0.5:
            # Start or Goal is blocked
            return None

        # Data Structures
        g_score = np.full((D, D, D), np.inf, dtype=np.float32)
        g_score[start_node] = 0.0
        
        # Parent pointer grid: stores delta to parent (dx, dy, dz) to save space or absolute?
        # Absolute is easier. (D, D, D, 3) 
        # Using -1 to indicate no parent
        parents = np.full((D, D, D, 3), -1, dtype=np.int16)
        
        # Heuristic Function
        goal_arr = goal_idx.astype(np.float32)
        
        # Priority Queue: (f_score, h_score, x, y, z)
        # Tie-break with h_score (smaller h -> closer to goal -> DFS-like)
        h_start = np.linalg.norm(start_idx - goal_arr)
        open_set = [(h_start, h_start, start_idx[0], start_idx[1], start_idx[2])]
        
        steps = 0
        
        while open_set:
            steps += 1
            if steps > timeout_steps:
                print("A* Timeout")
                return None
                
            f, h, cx, cy, cz = heapq.heappop(open_set)
            
            if (cx, cy, cz) == target:
                return self._reconstruct_path(parents, (cx, cy, cz), res)
            
            # Current g
            curr_g = g_score[cx, cy, cz]
            if curr_g == np.inf: continue # Stale node
            
            # Check neighbors
            # Vectorized check might be hard with heap, so we loop neighbors
            # But we can optimized neighbor loop?
            # 26 neighbors is constant.
            
            for i in range(len(self.neighbors)):
                dx, dy, dz = self.neighbors[i]
                nx, ny, nz = cx + dx, cy + dy, cz + dz
                
                # Bounds check
                if nx < 0 or nx >= D or ny < 0 or ny >= D or nz < 0 or nz >= D:
                    continue
                
                # Collision check
                if grid[nx, ny, nz] > 0.5:
                    continue
                
                # Cost (Euclidean)
                # We can precompute neighbor costs? 
                # neighbor magnitude: is it 1, sqrt(2), sqrt(3)?
                # We can compute it on fly or lookup
                dist_cost = float(np.sqrt(dx*dx + dy*dy + dz*dz))
                
                tentative_g = curr_g + dist_cost
                
                if tentative_g < g_score[nx, ny, nz]:
                    parents[nx, ny, nz] = [cx, cy, cz]
                    g_score[nx, ny, nz] = tentative_g
                    
                    # Heuristic
                    h_val = np.sqrt((nx - goal_idx[0])**2 + (ny - goal_idx[1])**2 + (nz - goal_idx[2])**2)
                    f_val = tentative_g + h_val
                    
                    heapq.heappush(open_set, (f_val, h_val, nx, ny, nz))
                    
        return None

    def _reconstruct_path(self, parents, current, res):
        path = [current]
        curr_idx = current
        
        while True:
            px, py, pz = parents[curr_idx[0], curr_idx[1], curr_idx[2]]
            if px == -1:
                break
            prev_idx = (px, py, pz)
            path.append(prev_idx)
            curr_idx = prev_idx
            
        path.reverse()
        
        # Convert to world coordinates
        world_path = []
        for idx in path:
            pt = (np.array(idx) + 0.5) * res
            world_path.append(pt)
            
        return np.array(world_path, dtype=np.float32)

if __name__ == "__main__":
    # Quick Test
    env = SpaceEnv()
    env.reset(seed=42)
    print("Environment Created. Obstacles:", len(env.obstacles))
    print("Start:", env.start)
    print("Goal:", env.goal)
    
    planner = AStarPlanner(env)
    path = planner.solve(env.start, env.goal)
    
    if path is not None:
        print(f"Path found! Length: {len(path)} waypoints")
    else:
        print("No path found.")
