
"""
Version 4: Comprehensive Benchmark (IAC 2026)
=============================================
Compares:
1. Classical A* (Optimal, Slow, High Memory)
2. RRT* (Sampling-based, High Variance)
3. APF (Reactive, Local Minima prone)
4. Hybrid Model (Ours: Fast, Low Memory, Safe)

Metrics:
- Success Rate (%)
- Mean Runtime (ms)
- Path Length (m) (Optimality)
- Peak Memory (MB)
- Est. Energy (Joule)
"""

import sys
import time
import numpy as np
import torch
import tracemalloc
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path('.').resolve()))

from src.environment.space_env import SpaceEnv, EnvironmentConfig, AStarPlanner
from src.planning.rrt_star import RRTStarPlanner
from src.planning.apf import PotentialFieldPlanner
from src.inference.pipeline import InferencePipeline

# Energy Model (Jetson Nano Profile)
POWER_CPU_WATTS = 5.0
POWER_GPU_WATTS = 10.0 # Peak for inference

def generate_scenarios(n=10, save_path="data/benchmark_scenarios.npz"):
    print(f"Generating {n} reproducible scenarios...")
    env = SpaceEnv(EnvironmentConfig(dynamic_obstacles=False)) # Static for fairness
    
    seeds = []
    starts = []
    goals = []
    
    for i in range(n):
        seed = 2026 + i
        seeds.append(seed)
        env.reset(seed=seed)
        starts.append(env.start)
        goals.append(env.goal)
        
    Path("data").mkdir(exist_ok=True)
    np.savez(save_path, seeds=seeds, starts=starts, goals=goals)
    print(f"Saved to {save_path}")

def run_benchmark():
    # Force regeneration to n=10
    generate_scenarios()
        
    scenario_path = "data/benchmark_scenarios.npz"
    data = np.load(scenario_path)
    seeds = data['seeds']
    
    # Init Planners
    env = SpaceEnv(EnvironmentConfig(dynamic_obstacles=False)) # Static environment for standard benchmark
    astar = AStarPlanner(env)
    rrt = RRTStarPlanner(env, max_iter=2000)
    apf = PotentialFieldPlanner(env, max_iter=3000)
    hybrid = InferencePipeline(model_path="checkpoints/best_model_3d.pth", device="cpu") # CPU for fair comparison on laptop
    
    # Results Container
    results = {
        'A*': {'success': 0, 'time': [], 'length': [], 'memory': [], 'energy': []},
        'RRT*': {'success': 0, 'time': [], 'length': [], 'memory': [], 'energy': []},
        'APF': {'success': 0, 'time': [], 'length': [], 'memory': [], 'energy': []},
        'Hybrid': {'success': 0, 'time': [], 'length': [], 'memory': [], 'energy': []}
    }
    
    n_total = len(seeds)
    
    print(f"🚀 Running Benchmark on {n_total} scenarios...")
    
    for i in tqdm(range(n_total)):
        seed = int(seeds[i])
        env.reset(seed=seed)
        start, goal = env.start, env.goal
        
        # --- 1. A* ---
        tracemalloc.start()
        t0 = time.time()
        path_astar = astar.solve(start, goal, timeout_steps=1000000)
        t_astar = (time.time() - t0)
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        if path_astar is not None:
            results['A*']['success'] += 1
            results['A*']['time'].append(t_astar * 1000) # ms
            results['A*']['length'].append(calculate_length(path_astar))
            results['A*']['memory'].append(peak / 1024 / 1024) # MB
            results['A*']['energy'].append(t_astar * POWER_CPU_WATTS) # J
        
        # --- 2. RRT* ---
        tracemalloc.start()
        t0 = time.time()
        path_rrt = rrt.solve(start, goal)
        t_rrt = (time.time() - t0)
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        if path_rrt is not None:
             results['RRT*']['success'] += 1
             results['RRT*']['time'].append(t_rrt * 1000)
             results['RRT*']['length'].append(calculate_length(path_rrt))
             results['RRT*']['memory'].append(peak / 1024 / 1024)
             results['RRT*']['energy'].append(t_rrt * POWER_CPU_WATTS)

        # --- 3. APF ---
        tracemalloc.start()
        t0 = time.time()
        path_apf = apf.solve(start, goal)
        t_apf = (time.time() - t0)
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        if path_apf is not None:
             results['APF']['success'] += 1
             results['APF']['time'].append(t_apf * 1000)
             results['APF']['length'].append(calculate_length(path_apf))
             results['APF']['memory'].append(peak / 1024 / 1024)
             results['APF']['energy'].append(t_apf * POWER_CPU_WATTS)

        # --- 4. Hybrid ---
        tracemalloc.start()
        t0 = time.time()
        # Input need grid
        grid = env.get_downsampled_grid()
        out = hybrid.predict(grid, start, goal, env.obstacles) # Predict + Repair
        path_hybrid = out['path']
        t_hybrid = (time.time() - t0)
        current, peak_hybrid = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        # Hybrid always returns a path, but is it collision free? 
        # Repair guarantees safety usually.
        # Let's assume 100% success for now as shown in V2, or check collision?
        # Let's check collision to be rigorous V4.
        is_safe = True
        for pt in path_hybrid:
            if env.check_collision(pt): is_safe = False; break
            
        if is_safe:
            results['Hybrid']['success'] += 1
            results['Hybrid']['time'].append(t_hybrid * 1000)
            results['Hybrid']['length'].append(calculate_length(path_hybrid))
            results['Hybrid']['memory'].append(peak_hybrid / 1024 / 1024)
            # Energy: Model runs on "GPU" power profile (simulated)
            results['Hybrid']['energy'].append(t_hybrid * POWER_GPU_WATTS)

    print_results(results, n_total)
    plot_results(results)

def calculate_length(path):
    length = 0
    for i in range(len(path)-1):
        length += np.linalg.norm(path[i+1] - path[i])
    return length

def print_results(results, n):
    output_str = ""
    output_str += "\n" + "="*80 + "\n"
    output_str += f"{'Algorithm':<10} | {'Success':<8} | {'Time (ms)':<12} | {'Length (m)':<12} | {'Mem (MB)':<10} | {'Energy (J)':<10}\n"
    output_str += "-" * 80 + "\n"
    
    for algo, res in results.items():
        succ_rate = (res['success'] / n) * 100
        time_mean = np.mean(res['time']) if res['time'] else 0
        len_mean = np.mean(res['length']) if res['length'] else 0
        mem_mean = np.mean(res['memory']) if res['memory'] else 0
        erg_mean = np.mean(res['energy']) if res['energy'] else 0
        
        output_str += f"{algo:<10} | {succ_rate:>7.1f}% | {time_mean:>12.1f} | {len_mean:>12.1f} | {mem_mean:>10.1f} | {erg_mean:>10.4f}\n"
    output_str += "="*80 + "\n"
    
    print(output_str) 
    
    Path("results").mkdir(exist_ok=True)
    with open("results/benchmark_table.txt", "w") as f:
        f.write(output_str)
    print("Saved table to results/benchmark_table.txt")

def plot_results(results):
    metrics = ['Time (ms)', 'Memory (MB)', 'Energy (J)']
    algos = list(results.keys())
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Data prep
    times = [np.mean(results[a]['time']) if results[a]['time'] else 0 for a in algos]
    mems = [np.mean(results[a]['memory']) if results[a]['memory'] else 0 for a in algos]
    energies = [np.mean(results[a]['energy']) if results[a]['energy'] else 0 for a in algos]
    
    # 1. Time
    axes[0].bar(algos, times, color=['gray', 'orange', 'red', 'green'])
    axes[0].set_title('Mean Runtime (Lower is Better)')
    axes[0].set_ylabel('ms')
    
    # 2. Memory
    axes[1].bar(algos, mems, color=['gray', 'orange', 'red', 'green'])
    axes[1].set_title('Peak Memory (Lower is Better)')
    axes[1].set_ylabel('MB')
    
    # 3. Energy
    axes[2].bar(algos, energies, color=['gray', 'orange', 'red', 'green'])
    axes[2].set_title('Est. Energy Consumption (Lower is Better)')
    axes[2].set_ylabel('Joules')
    
    plt.tight_layout()
    Path("results").mkdir(exist_ok=True)
    plt.savefig('results/benchmark_v4_metrics.png')
    print("Saved plot to results/benchmark_v4_metrics.png")

if __name__ == "__main__":
    run_benchmark()
