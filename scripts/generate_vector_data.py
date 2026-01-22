
"""
Vector Data Generation Script
=============================
Generates compressed vector training data for PointNet Model (V5).
Pipeline:
1. Generate Random Environment (SpaceEnv).
2. Solve with A* (AStarPlanner).
3. Extract Vector State (Egocentric Obstacles).
4. Save to .npz
"""

import sys
import numpy as np
import time
from pathlib import Path
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.environment.space_env import SpaceEnv, EnvironmentConfig, AStarPlanner

def interpolate_path(path, num_points=20):
    """Interpolate 3D path to fixed number of points."""
    if len(path) < 2:
        return np.resize(path, (num_points, 3))
    
    dists = np.linalg.norm(path[1:] - path[:-1], axis=1)
    cum_dist = np.insert(np.cumsum(dists), 0, 0)
    total_dist = cum_dist[-1]
    
    target_dists = np.linspace(0, total_dist, num_points)
    
    new_path = np.zeros((num_points, 3))
    for i in range(3):
        new_path[:, i] = np.interp(target_dists, cum_dist, path[:, i])
        
    return new_path

def augment_data(obs, start, goal, path):
    """
    Augment data by rotating 90 degrees around Z axis.
    obs: (N, 4) [x, y, z, r] (Relative to START)
    start: (3,) (Absolute)
    goal: (3,) (Absolute)
    path: (K, 3) (Absolute)
    """
    augmented = []
    
    # Original
    augmented.append((obs.copy(), start.copy(), goal.copy(), path.copy()))
    
    curr_obs = obs.copy()
    curr_start = start.copy()
    curr_goal = goal.copy()
    curr_path = path.copy()
    
    # Rotation logic
    for _ in range(3):
        # Rotate vectors (x, y) -> (-y, x)
        
        # 1. Rotate Obstacles (Relative vectors)
        # obs positions are [x, y, z]. Z stays same.
        new_obs_x = -curr_obs[:, 1]
        new_obs_y = curr_obs[:, 0]
        curr_obs[:, 0] = new_obs_x
        curr_obs[:, 1] = new_obs_y
        
        # 2. Rotate Absolute Points (around center 500,500,500 like in original logic?)
        # Or simple rotation? The original logic rotated around 500,500,500 roughly.
        # Let's match the original logic: x_new = 1000 - y, y_new = x
        
        def rot_abs(p):
            x, y, z = p[..., 0], p[..., 1], p[..., 2]
            new_x = 1000.0 - y
            new_y = x
            new_z = z
            return np.stack([new_x, new_y, new_z], axis=-1)

        curr_start = rot_abs(curr_start)
        curr_goal = rot_abs(curr_goal)
        curr_path = rot_abs(curr_path)
        
        # Re-calc relative obs? No, we rotated the relative vector already.
        # But wait. If we rotate the World, the Relative Vector (Obs - Start) rotates naturally.
        # Let's verify:
        # P_obs_new = R * P_obs_old
        # P_start_new = R * P_start_old
        # Rel_new = P_obs_new - P_start_new = R(P_obs_old - P_start_old) = R * Rel_old
        # So yes, simply applying the rotation matrix to the relative vector is correct.
        
        augmented.append((curr_obs.copy(), curr_start, curr_goal, curr_path))
        
    return augmented

def generate_dataset(n_samples=2000, save_path="data/train_data_vector.npz"):
    config = EnvironmentConfig(grid_dim=100, n_obstacles=12) # Grid dim irrelevant but needed for config
    env = SpaceEnv(config)
    planner = AStarPlanner(env)
    
    data_obs = []
    data_starts = []
    data_goals = []
    data_paths = []
    
    pbar = tqdm(total=n_samples)
    
    while len(data_obs) < n_samples:
        env.reset(seed=int(time.time() * 10000) % 100000)
        
        path = planner.solve(env.start, env.goal, timeout_steps=50000)
        
        if path is not None:
            # Success
            # Get Vector State (Relative to Start)
            # n_obs=20 covers all 12 obstacles + padding
            obs_state = env.get_vector_state(env.start, n_obs=20, relative=True)
            
            fixed_path = interpolate_path(path, 20)
            
            # Augment
            aug_samples = augment_data(obs_state, env.start, env.goal, fixed_path)
            
            for o, s, g, p in aug_samples:
                if len(data_obs) >= n_samples:
                    break
                data_obs.append(o)
                data_starts.append(s)
                data_goals.append(g)
                data_paths.append(p)
                pbar.update(1)
                
    pbar.close()
    
    print(f"Saving {len(data_obs)} samples to {save_path}...")
    np.savez_compressed(
        save_path,
        obstacles=np.array(data_obs, dtype=np.float32),
        starts=np.array(data_starts, dtype=np.float32),
        goals=np.array(data_goals, dtype=np.float32),
        paths=np.array(data_paths, dtype=np.float32)
    )
    print("Done.")

if __name__ == "__main__":
    Path("data").mkdir(exist_ok=True)
    generate_dataset(n_samples=500, save_path="data/train_data_vector.npz")
