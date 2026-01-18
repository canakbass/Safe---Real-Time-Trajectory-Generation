
"""
Data Generation Script
======================
Generates training data for 3D Trajectory Model.
Pipeline:
1. Generate Random Environment (SpaceEnv).
2. Solve with A* (AStarPlanner).
3. Post-process:
    - Downsample Grid (Input).
    - Interpolate Path to 20 points (Label).
    - Augment (Random Rotations).
4. Save to .npz
"""

import sys
import numpy as np
import time
from pathlib import Path
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.environment.space_env import SpaceEnv, EnvironmentConfig, AStarPlanner

def interpolate_path(path, num_points=20):
    """Interpolate 3D path to fixed number of points."""
    if len(path) < 2:
        return np.resize(path, (num_points, 3))
    
    # Calculate cumulative distance
    dists = np.linalg.norm(path[1:] - path[:-1], axis=1)
    cum_dist = np.insert(np.cumsum(dists), 0, 0)
    total_dist = cum_dist[-1]
    
    # Sample linearly spaced distances
    target_dists = np.linspace(0, total_dist, num_points)
    
    # Interpolate for each dimension
    new_path = np.zeros((num_points, 3))
    for i in range(3):
        new_path[:, i] = np.interp(target_dists, cum_dist, path[:, i])
        
    return new_path

def augment_data(grid, start, goal, path):
    """
    Augment data by rotating 90 degrees around Z axis.
    Returns list of (grid, start, goal, path) tuples (Original + 3 Rotations).
    """
    augmented = []
    
    # Original
    augmented.append((grid.copy(), start.copy(), goal.copy(), path.copy()))
    
    # Rotations (90, 180, 270)
    # Grid: (100, 100, 100) -> Rotate around Z (axis 2? or 0,1 plane)
    # Let's assume Z is up (index 2). We rotate in XY plane.
    
    curr_grid = grid
    curr_start = start
    curr_goal = goal
    curr_path = path
    
    center = np.array([500, 500, 500]) # Approx center for point rotation
    
    for _ in range(3):
        # 1. Rotate Grid (90 deg counter-clockwise in XY)
        # NumPy rot90 rotates first two axes by default.
        # grid shape (X, Y, Z). rot90(k=1) rotates X->Y.
        curr_grid = np.rot90(curr_grid, k=1, axes=(0, 1))
        
        # 2. Rotate Points
        # 90 deg rotation matrix around Z:
        # [0 -1  0]
        # [1  0  0]
        # [0  0  1]
        
        # We need to rotate around center (500,500,500) ideally, 
        # but our grid rotation is around array center.
        # Let's stick to array center logic.
        
        # Simple coordinate swap for 90 deg: (x, y) -> (-y, x)
        # But we must map back to [0, 1000].
        # x_new = 1000 - y
        # y_new = x
        
        def rot_point(p):
            # p: (N, 3) or (3,)
            x, y, z = p[..., 0], p[..., 1], p[..., 2]
            new_x = 1000.0 - y
            new_y = x
            new_z = z
            return np.stack([new_x, new_y, new_z], axis=-1)

        curr_start = rot_point(curr_start)
        curr_goal = rot_point(curr_goal)
        curr_path = rot_point(curr_path)
        
        augmented.append((curr_grid.copy(), curr_start, curr_goal, curr_path))
        
    return augmented

def generate_dataset(n_samples=1000, save_path="data/train_data.npz"):
    config = EnvironmentConfig(grid_dim=100, n_obstacles=12)
    env = SpaceEnv(config)
    planner = AStarPlanner(env)
    
    data_grids = []
    data_starts = []
    data_goals = []
    data_paths = []
    
    pbar = tqdm(total=n_samples)
    attempts = 0
    
    while len(data_grids) < n_samples:
        attempts += 1
        env.reset(seed=int(time.time() * 1000) % 100000)
        
        # Solve
        path = planner.solve(env.start, env.goal, timeout_steps=50000)
        
        if path is not None:
            # Success
            grid = env.get_downsampled_grid()
            fixed_path = interpolate_path(path, 20)
            
            # Augment (4x multiplier)
            aug_samples = augment_data(grid, env.start, env.goal, fixed_path)
            
            for g, s, e, p in aug_samples:
                if len(data_grids) >= n_samples:
                    break
                data_grids.append(g)
                data_starts.append(s)
                data_goals.append(e)
                data_paths.append(p)
                pbar.update(1)
                
    pbar.close()
    
    # Save
    print(f"Saving {len(data_grids)} samples to {save_path}...")
    np.savez_compressed(
        save_path,
        grids=np.array(data_grids, dtype=np.bool_), # Save space
        starts=np.array(data_starts, dtype=np.float32),
        goals=np.array(data_goals, dtype=np.float32),
        paths=np.array(data_paths, dtype=np.float32)
    )
    print("Done.")

if __name__ == "__main__":
    Path("data").mkdir(exist_ok=True)
    # Generate small batch for Verification but name it train_data so train.py finds it immediately
    generate_dataset(n_samples=400, save_path="data/train_data.npz")
