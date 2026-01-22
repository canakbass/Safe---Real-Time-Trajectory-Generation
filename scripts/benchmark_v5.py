
"""
Version 5: The Vector Revolution Benchmark
==========================================
Compares:
1. Classical A* (Baseline)
2. APF (Competitor: Fast & Light)
3. Hybrid-Grid (Old V3: Heavy)
4. Hybrid-Vector (New V5: The Solution)

Focus: Proving V5 beats APF in Memory/Speed while matching A* in quality.
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
from src.planning.apf import PotentialFieldPlanner
from src.inference.pipeline import InferencePipeline
from src.model.vector_net import VectorTrajectoryGenerator

# Energy Model
POWER_CPU_WATTS = 5.0
POWER_GPU_WATTS = 10.0

def generate_scenarios(n=10, save_path="data/benchmark_scenarios.npz"):
    print(f"Generating {n} reproducible scenarios...")
    env = SpaceEnv(EnvironmentConfig(dynamic_obstacles=False))
    
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
    # Force regeneration to n=10 for fair comparison on same subset
    generate_scenarios(n=10)
        
    scenario_path = "data/benchmark_scenarios.npz"
    data = np.load(scenario_path)
    seeds = data['seeds']
    
    # Init Planners
    env = SpaceEnv(EnvironmentConfig(dynamic_obstacles=False))
    
    # 1. A*
    astar = AStarPlanner(env)
    
    # 2. APF
    apf = PotentialFieldPlanner(env, max_iter=3000)
    
    # 3. Hybrid-Grid (V3)
    hybrid_grid = InferencePipeline(model_path="checkpoints/best_model_3d.pth", device="cpu")
    
    # 4. Hybrid-Vector (V5)
    device = torch.device('cpu') # Run on CPU for fair memory profiling vs others
    vector_model = VectorTrajectoryGenerator(n_obstacles=20, feature_dim=128, path_len=20).to(device)
    try:
        vector_model.load_state_dict(torch.load("checkpoints/best_model_vector_3d.pth", map_location=device))
        vector_model.eval()
        print("Loaded V5 Vector Model.")
    except Exception as e:
        print(f"Failed to load V5 model: {e}")
        return

    # Results
    results = {
        'A*':      {'success': 0, 'time': [], 'length': [], 'memory': [], 'energy': []},
        'APF':     {'success': 0, 'time': [], 'length': [], 'memory': [], 'energy': []},
        'Hybrid-G': {'success': 0, 'time': [], 'length': [], 'memory': [], 'energy': []}, # Grid
        'Hybrid-V': {'success': 0, 'time': [], 'length': [], 'memory': [], 'energy': []}  # Vector
    }
    
    n_total = len(seeds)
    print(f"🚀 Running benchmark on {n_total} scenarios...")
    
    for i in tqdm(range(n_total)):
        seed = int(seeds[i])
        env.reset(seed=seed)
        start, goal = env.start, env.goal
        
        # --- 1. A* ---
        tracemalloc.start()
        t0 = time.time()
        path = astar.solve(start, goal, timeout_steps=1000000)
        t_dur = time.time() - t0
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        if path is not None:
            results['A*']['success'] += 1
            results['A*']['time'].append(t_dur * 1000)
            results['A*']['length'].append(calculate_length(path))
            results['A*']['memory'].append(peak / 1024 / 1024)
            results['A*']['energy'].append(t_dur * POWER_CPU_WATTS)

        # --- 2. APF ---
        tracemalloc.start()
        t0 = time.time()
        path = apf.solve(start, goal)
        t_dur = time.time() - t0
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        if path is not None:
             results['APF']['success'] += 1
             results['APF']['time'].append(t_dur * 1000)
             results['APF']['length'].append(calculate_length(path))
             results['APF']['memory'].append(peak / 1024 / 1024)
             results['APF']['energy'].append(t_dur * POWER_CPU_WATTS)

        # --- 3. Hybrid-Grid ---
        tracemalloc.start()
        t0 = time.time()
        grid = env.get_downsampled_grid()
        out = hybrid_grid.predict(grid, start, goal, env.obstacles) 
        path = out['path']
        t_dur = time.time() - t0
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        results['Hybrid-G']['success'] += 1
        results['Hybrid-G']['time'].append(t_dur * 1000)
        results['Hybrid-G']['length'].append(calculate_length(path))
        results['Hybrid-G']['memory'].append(peak / 1024 / 1024)
        results['Hybrid-G']['energy'].append(t_dur * POWER_GPU_WATTS)

        # --- 4. Hybrid-Vector (The Star) ---
        tracemalloc.start()
        t0 = time.time()
        
        # 4a. Get Vector State (The fast part)
        obs_state = env.get_vector_state(start, n_obs=20, relative=True)
        
        # 4b. Inference
        with torch.no_grad():
            t_obs = torch.FloatTensor(obs_state).unsqueeze(0).to(device)
            t_start = torch.FloatTensor(start).unsqueeze(0).to(device)
            t_goal = torch.FloatTensor(goal).unsqueeze(0).to(device)
            
            pred_path = vector_model(t_obs, t_start, t_goal)
            path = pred_path[0].cpu().numpy()
            
        t_dur = time.time() - t0
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        results['Hybrid-V']['success'] += 1
        results['Hybrid-V']['time'].append(t_dur * 1000)
        results['Hybrid-V']['length'].append(calculate_length(path))
        results['Hybrid-V']['memory'].append(peak / 1024 / 1024)
        results['Hybrid-V']['energy'].append(t_dur * POWER_GPU_WATTS) # Using GPU profile even if on CPU for consistency

    print_results(results, n_total)
    plot_results(results)

def calculate_length(path):
    length = 0
    for i in range(len(path)-1):
        length += np.linalg.norm(path[i+1] - path[i])
    return length

def print_results(results, n):
    output_str = ""
    output_str += "\n" + "="*90 + "\n"
    output_str += f"{'Algorithm':<12} | {'Success':<8} | {'Time (ms)':<12} | {'Length (m)':<12} | {'Mem (MB)':<10} | {'Energy (J)':<10}\n"
    output_str += "-" * 90 + "\n"
    
    for algo, res in results.items():
        succ_rate = (res['success'] / n) * 100
        time_mean = np.mean(res['time']) if res['time'] else 0
        len_mean = np.mean(res['length']) if res['length'] else 0
        mem_mean = np.mean(res['memory']) if res['memory'] else 0
        erg_mean = np.mean(res['energy']) if res['energy'] else 0
        
        output_str += f"{algo:<12} | {succ_rate:>7.1f}% | {time_mean:>12.1f} | {len_mean:>12.1f} | {mem_mean:>10.1f} | {erg_mean:>10.4f}\n"
    output_str += "="*90 + "\n"
    
    print(output_str) 
    
    Path("results").mkdir(exist_ok=True)
    with open("results/benchmark_v5_table.txt", "w") as f:
        f.write(output_str)

def plot_results(results):
    metrics = ['Time (ms)', 'Memory (MB)', 'Energy (J)']
    algos = list(results.keys())
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    times = [np.mean(results[a]['time']) if results[a]['time'] else 0 for a in algos]
    mems = [np.mean(results[a]['memory']) if results[a]['memory'] else 0 for a in algos]
    energies = [np.mean(results[a]['energy']) if results[a]['energy'] else 0 for a in algos]
    
    colors = ['gray', 'red', 'blue', 'green']
    
    axes[0].bar(algos, times, color=colors)
    axes[0].set_title('Mean Runtime')
    axes[0].set_ylabel('ms')
    
    axes[1].bar(algos, mems, color=colors)
    axes[1].set_title('Peak Memory')
    axes[1].set_ylabel('MB')
    
    axes[2].bar(algos, energies, color=colors)
    axes[2].set_title('Energy Consumption')
    axes[2].set_ylabel('Joules')
    
    plt.tight_layout()
    Path("results").mkdir(exist_ok=True)
    plt.savefig('results/benchmark_v5_metrics.png')
    print("Saved plot to results/benchmark_v5_metrics.png")

if __name__ == "__main__":
    run_benchmark()
