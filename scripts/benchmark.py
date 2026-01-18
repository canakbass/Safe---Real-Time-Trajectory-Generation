
"""
Benchmark Script - Memory & Speed
=================================
Validates the performance of the 3D Space Environment and A* Solver.
Target: Jetson Nano (Embedded).
"""

import time
import numpy as np
import sys
from pathlib import Path
import tracemalloc
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path('.').resolve()))

from src.environment.space_env import SpaceEnv, EnvironmentConfig, AStarPlanner
from src.inference.pipeline import InferencePipeline

def run_benchmark(n_iter=10): # Reduced N for quick memory check
    print(f"🥊 BENCHMARK BAŞLIYOR (N={n_iter}) - Memory Check...")
    print("-" * 60)
    
    config = EnvironmentConfig(grid_dim=100, n_obstacles=12)
    env = SpaceEnv(config)
    
    # 1. Klasik A* Başlat
    planner = AStarPlanner(env)
    
    # 2. Bizim Modeli Başlat
    model_path = "checkpoints/best_model_3d.pth"
    pipeline = InferencePipeline(model_path=model_path, device="cpu")
    
    results = {
        'astar': {'success': 0, 'time': [], 'memory': []},
        'model': {'success': 0, 'time': [], 'memory': []}
    }
    
    for i in tqdm(range(n_iter)):
        env.reset(seed=1000+i)
        
        # --- YARIŞMACI 1: A* ---
        tracemalloc.start()
        t0 = time.perf_counter()
        
        # NOTE: A* memory grows with search depth.
        # To see "Peak", we need a tough problem.
        # But for speed, we limit this benchmark run.
        astar_path = planner.solve(env.start, env.goal, timeout_steps=500000) 
        
        t1 = time.perf_counter()
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        results['astar']['memory'].append(peak / 1024 / 1024) # MB
        
        if astar_path is not None:
            results['astar']['success'] += 1
            results['astar']['time'].append((t1-t0)*1000)
            
        # --- YARIŞMACI 2: Hybrid Model ---
        tracemalloc.start()
        t0 = time.perf_counter()
        out = pipeline.predict(env.get_downsampled_grid(), env.start, env.goal, env.obstacles)
        t1 = time.perf_counter()
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        results['model']['memory'].append(peak / 1024 / 1024) # MB
        
        path = out['path']
        t_total = out['metrics']['total_time_ms']
        
        # Check Collision
        is_valid = True
        for point in path:
            if env.check_collision(point, margin=5.0):
                is_valid = False
                break
        
        if is_valid:
            results['model']['success'] += 1
        
        results['model']['time'].append(t_total)

    print("\n📊 SONUÇLAR (Ortalamalar):")
    
    # Calculate means (handle cases where astar had 0 success for empty list)
    astar_time = np.mean(results['astar']['time']) if results['astar']['time'] else 0.0
    astar_mem = np.mean(results['astar']['memory'])
    
    model_time = np.mean(results['model']['time'])
    model_mem = np.mean(results['model']['memory'])
    
    print(f"A*    -> Süre: {astar_time:.1f} ms | RAM: {astar_mem:.2f} MB")
    print(f"Model -> Süre: {model_time:.1f} ms | RAM: {model_mem:.2f} MB")
    
    if model_mem > 0:
        ratio = astar_mem / model_mem
        print(f"💾 Bellek Farkı: Model, A*'dan {ratio:.1f}x daha az/fazla bellek harcıyor.")

if __name__ == "__main__":
    run_benchmark()
