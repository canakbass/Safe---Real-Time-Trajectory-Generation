
"""
Benchmark Script - Step 1
=========================
Validates the performance of the 3D Space Environment and A* Solver.
Target: Jetson Nano (Embedded) -> Check Memory & Speed.
"""

import time
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.environment.space_env import SpaceEnv, EnvironmentConfig, AStarPlanner

def benchmark_step1(n_iter=10):
    print(f"Benchmarking 3D Environment (N={n_iter})...")
    print("-" * 60)
    
    times_reset = []
    times_grid = []
    times_astar = []
    success_count = 0
    total_steps = 0
    
    config = EnvironmentConfig(
        grid_dim=100,
        n_obstacles=15
    )
    env = SpaceEnv(config)
    planner = AStarPlanner(env)
    
    for i in range(n_iter):
        # 1. Reset Time
        t0 = time.perf_counter()
        env.reset(seed=i)
        t1 = time.perf_counter()
        times_reset.append((t1 - t0) * 1000)
        
        # 2. Grid Downsampling Time (Critical for Model Input)
        t0 = time.perf_counter()
        grid = env.get_downsampled_grid()
        t1 = time.perf_counter()
        times_grid.append((t1 - t0) * 1000)
        
        # 3. A* Solver Time
        t0 = time.perf_counter()
        path = planner.solve(env.start, env.goal, timeout_steps=100000)
        t1 = time.perf_counter()
        times_astar.append((t1 - t0) * 1000)
        
        if path is not None:
            success_count += 1
            total_steps += len(path)
            
    print(f"\nResults over {n_iter} iterations:")
    print(f"1. Env Reset (Analytic Obstacles): {np.mean(times_reset):.2f} ms ± {np.std(times_reset):.2f}")
    print(f"2. Grid Downsampling (100^3):    {np.mean(times_grid):.2f} ms ± {np.std(times_grid):.2f}")
    print(f"3. A* Solve Time:                  {np.mean(times_astar):.2f} ms ± {np.std(times_astar):.2f}")
    
    print(f"\nA* Success Rate: {success_count}/{n_iter} ({success_count/n_iter*100:.1f}%)")
    if success_count > 0:
        print(f"Avg Path Length: {total_steps/success_count:.1f} steps")

if __name__ == "__main__":
    benchmark_step1()
