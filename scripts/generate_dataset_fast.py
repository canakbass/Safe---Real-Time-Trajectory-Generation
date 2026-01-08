"""
Fast Expert Trajectory Generator (A* Based)
===========================================
Layer 0: High-speed training data generation.

Replaces slow RRT* with A* (A-Star) on a grid, followed by 
B-Spline smoothing to create realistic continuous trajectories.

Speedup: ~100x faster than RRT*.
"""

import argparse
import sys
import numpy as np
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import json
import shutil
import heapq
from scipy.interpolate import splprep, splev

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.environment import SpaceEnv, SpaceEnvConfig

class AStarSolver:
    """Helper class for fast grid-based pathfinding."""
    
    def __init__(self, resolution=64):
        self.resolution = resolution
        
    def solve(self, env_map, start_pos, target_pos, env_bounds=100.0):
        """
        Runs A* on the grid map.
        env_map: 2D numpy array (1=obstacle, 0=free)
        """
        # Convert continuous coords to grid indices
        scale = self.resolution / env_bounds
        start_idx = (int(start_pos[1] * scale), int(start_pos[0] * scale))
        target_idx = (int(target_pos[1] * scale), int(target_pos[0] * scale))
        
        # Grid bounds
        h, w = env_map.shape
        
        # Check if start or target are in obstacles (due to discretization)
        if env_map[start_idx] == 1 or env_map[target_idx] == 1:
            return None
            
        # A* Algorithm
        # Priority Queue: (f_score, g_score, (y, x), path)
        open_set = []
        heapq.heappush(open_set, (0, 0, start_idx, [start_idx]))
        
        visited = set()
        visited.add(start_idx)
        
        # Directions (8-connected)
        neighbors = [(-1,0), (1,0), (0,-1), (0,1), (-1,-1), (-1,1), (1,-1), (1,1)]
        
        while open_set:
            _, g, current, path = heapq.heappop(open_set)
            
            if current == target_idx:
                return self._smooth_path(path, scale)
            
            for dy, dx in neighbors:
                ny, nx = current[0] + dy, current[1] + dx
                
                if 0 <= ny < h and 0 <= nx < w:
                    if env_map[ny, nx] == 0 and (ny, nx) not in visited:
                        visited.add((ny, nx))
                        
                        # Cost: 1.0 for straight, 1.414 for diagonal
                        move_cost = 1.414 if dy != 0 and dx != 0 else 1.0
                        new_g = g + move_cost
                        
                        # Heuristic: Euclidean distance
                        dist = np.sqrt((target_idx[0]-ny)**2 + (target_idx[1]-nx)**2)
                        f = new_g + dist
                        
                        new_path = list(path)
                        new_path.append((ny, nx))
                        heapq.heappush(open_set, (f, new_g, (ny, nx), new_path))
                        
        return None # No path found

    def _smooth_path(self, grid_path, scale, n_waypoints=50):
        """Convert grid path to continuous smooth trajectory."""
        # Convert back to continuous coordinates
        # Swap y,x back to x,y and scale up
        coords = np.array([(p[1], p[0]) for p in grid_path]) / scale
        
        # If path is too short for spline, linear interpolate
        if len(coords) < 4:
            return self._linear_interp(coords, n_waypoints)
            
        try:
            # B-Spline Interpolation for smoothing
            # s=smoothing factor (0=pass through all points)
            tck, u = splprep(coords.T, s=2.0, k=3) 
            u_new = np.linspace(u.min(), u.max(), n_waypoints)
            smooth_path = np.column_stack(splev(u_new, tck))
            return smooth_path.astype(np.float32)
        except:
            return self._linear_interp(coords, n_waypoints)

    def _linear_interp(self, coords, n_points):
        """Simple linear interpolation if Spline fails."""
        x = coords[:, 0]
        y = coords[:, 1]
        t = np.linspace(0, 1, len(coords))
        t_new = np.linspace(0, 1, n_points)
        x_new = np.interp(t_new, t, x)
        y_new = np.interp(t_new, t, y)
        return np.column_stack((x_new, y_new)).astype(np.float32)


def generate_expert_dataset(
    n_samples: int = 1000,
    n_waypoints: int = 50,
    map_resolution: int = 64,
    seed: int = 42,
    output_dir: str = "./data",
    difficulty: str = "medium",
) -> str:
    
    print("=" * 70)
    print("FAST EXPERT TRAJECTORY GENERATOR (A* Version)")
    print("Layer 0: High-Speed Training Data Pipeline")
    print("=" * 70)
    
    # Initialize
    rng = np.random.default_rng(seed)
    astar = AStarSolver(resolution=map_resolution)
    
    # Difficulty settings
    difficulty_params = {
        'easy': {'n_obstacles': 8, 'min_r': 3.0, 'max_r': 6.0},
        'medium': {'n_obstacles': 12, 'min_r': 3.0, 'max_r': 8.0},
        'hard': {'n_obstacles': 18, 'min_r': 4.0, 'max_r': 9.0},
    }
    params = difficulty_params.get(difficulty, difficulty_params['medium'])
    
    # Storage
    trajectories = []
    obstacle_maps = []
    start_positions = []
    target_positions = []
    
    # Statistics
    stats = {'total_attempts': 0, 'successful': 0, 'failed': 0}
    
    pbar = tqdm(total=n_samples, desc="Generating trajectories")
    
    while len(trajectories) < n_samples:
        stats['total_attempts'] += 1
        
        # 1. Create Environment
        env_seed = int(rng.integers(0, 2**31))
        config = SpaceEnvConfig(
            n_obstacles=params['n_obstacles'],
            min_obstacle_radius=params['min_r'],
            max_obstacle_radius=params['max_r'],
            seed=env_seed,
        )
        env = SpaceEnv(config)
        env.reset()
        
        # 2. Get Map & Solve with A*
        # Use a slightly higher resolution map for pathfinding to be safe
        grid_map = env.get_obstacle_map(map_resolution)
        
        path = astar.solve(grid_map, env.start_pos, env.target_pos)
        
        if path is None:
            stats['failed'] += 1
            continue
            
        # 3. Double Check Collision (Safety)
        # Because Spline smoothing might cut corners, we verify with strict physics
        has_collision, _ = env.check_trajectory_collision(path, safety_margin=1.0)
        
        if has_collision:
            stats['failed'] += 1
            continue
            
        # SUCCESS
        trajectories.append(path)
        obstacle_maps.append(grid_map)
        start_positions.append(env.start_pos.copy())
        target_positions.append(env.target_pos.copy())
        
        stats['successful'] += 1
        pbar.update(1)
        
    pbar.close()
    
    # Convert & Save (Same as before)
    trajectories = np.array(trajectories, dtype=np.float32)
    obstacle_maps = np.array(obstacle_maps, dtype=np.float32)
    start_positions = np.array(start_positions, dtype=np.float32)
    target_positions = np.array(target_positions, dtype=np.float32)
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"expert_trajectories_{n_samples}_{timestamp}.npz"
    filepath = output_path / filename
    
    np.savez_compressed(
        filepath,
        trajectories=trajectories,
        obstacle_maps=obstacle_maps,
        start_positions=start_positions,
        target_positions=target_positions,
        metadata=json.dumps({
            'generator': 'A-Star-Spline',
            'n_samples': n_samples,
            'stats': stats
        })
    )
    
    print(f"\n✓ Saved fast dataset to: {filepath}")
    
    # Copy as latest
    latest_path = output_path / "expert_trajectories_latest.npz"
    if latest_path.exists(): latest_path.unlink()
    shutil.copy(filepath, latest_path)
    
    return str(filepath)

if __name__ == "__main__":
    generate_expert_dataset()