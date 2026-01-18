"""
Quick Timing Test - GPU vs CPU Inference
=========================================
Tests existing trained models without retraining.

Usage:
    python scripts/timing_test.py
"""

import numpy as np
import torch
import time
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.environment.space_env import SpaceEnvironment, EnvironmentConfig
from scripts.benchmark import (
    SimpleTrajectoryMLP, DeepCNNTrajectory, UNetTrajectory, AttentionTrajectory
)


def time_inference(model, env, device, n_runs=20):
    """Time inference on a specific device."""
    model = model.to(device)
    model.eval()
    
    # Prepare input
    obs_map = torch.FloatTensor(env.get_obstacle_map(64)).unsqueeze(0).unsqueeze(0).to(device)
    start = torch.FloatTensor(env.start_pos / env.config.width).unsqueeze(0).to(device)
    target = torch.FloatTensor(env.target_pos / env.config.width).unsqueeze(0).to(device)
    
    # Warmup
    for _ in range(5):
        with torch.no_grad():
            _ = model(obs_map, start, target)
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    # Timed runs
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        
        with torch.no_grad():
            pred = model(obs_map, start, target)
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        # Include denormalization and collision repair
        traj = pred.cpu().numpy()[0] * env.config.width
        traj[0] = env.start_pos
        traj[-1] = env.target_pos
        
        # Simple collision repair
        for iteration in range(5):
            collision_found = False
            for j in range(1, len(traj) - 1):
                point = traj[j]
                for obs in env.obstacles:
                    dist = np.linalg.norm(point - obs.center)
                    safe_dist = obs.radius + 10.0
                    if dist < safe_dist:
                        collision_found = True
                        direction = (point - obs.center) / max(dist, 1e-6)
                        traj[j] = obs.center + direction * (safe_dist + 5.0)
                        traj[j] = np.clip(traj[j], 10, env.config.width - 10)
            if not collision_found:
                break
        
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)
    
    return np.mean(times), np.std(times)


def main():
    print("=" * 70)
    print("QUICK TIMING TEST: GPU vs CPU")
    print("=" * 70)
    
    # Setup environment
    config = EnvironmentConfig(seed=1000)
    env = SpaceEnvironment(config)
    env.reset(seed=1000)
    
    # A* baseline
    print("\n📊 A* Baseline:")
    times = []
    for i in range(20):
        env.reset(seed=1000 + i)
        t0 = time.perf_counter()
        path = env.solve_astar()
        if path is not None:
            traj = env.interpolate_trajectory(path, n_points=50)
        times.append((time.perf_counter() - t0) * 1000)
    
    astar_time = np.mean(times)
    print(f"   A* avg time: {astar_time:.2f} ms (±{np.std(times):.2f})")
    
    # Models to test
    models_info = [
        ('SimpleMLP', SimpleTrajectoryMLP, './checkpoints/simplemlp_best.pth'),
        ('DeepCNN', DeepCNNTrajectory, './checkpoints/deepcnn_best.pth'),
        ('UNet', UNetTrajectory, './checkpoints/unet_best.pth'),
        ('Attention', AttentionTrajectory, './checkpoints/attention_best.pth'),
    ]
    
    print("\n📊 Model Inference Times (Full Hybrid Pipeline):")
    print("-" * 70)
    print(f"{'Model':<12} {'Params':>10} {'GPU (ms)':>12} {'CPU (ms)':>12} {'Best':>8} {'vs A*':>10}")
    print("-" * 70)
    
    env.reset(seed=1000)
    
    for name, model_class, checkpoint in models_info:
        # Load model
        model = model_class(n_waypoints=50)
        
        if Path(checkpoint).exists():
            ckpt = torch.load(checkpoint, map_location='cpu', weights_only=True)
            model.load_state_dict(ckpt['model_state_dict'])
        
        n_params = sum(p.numel() for p in model.parameters())
        
        # GPU timing
        if torch.cuda.is_available():
            gpu_time, gpu_std = time_inference(model, env, torch.device('cuda'))
        else:
            gpu_time, gpu_std = float('inf'), 0
        
        # CPU timing
        cpu_time, cpu_std = time_inference(model, env, torch.device('cpu'))
        
        # Best
        if gpu_time < cpu_time:
            best = 'GPU'
            best_time = gpu_time
        else:
            best = 'CPU'
            best_time = cpu_time
        
        # vs A*
        if best_time < astar_time:
            speedup = f"{astar_time/best_time:.1f}x faster"
        else:
            speedup = f"{best_time/astar_time:.1f}x slower"
        
        gpu_str = f"{gpu_time:.2f}±{gpu_std:.1f}" if gpu_time != float('inf') else "N/A"
        cpu_str = f"{cpu_time:.2f}±{cpu_std:.1f}"
        
        print(f"{name:<12} {n_params:>10,} {gpu_str:>12} {cpu_str:>12} {best:>8} {speedup:>10}")
    
    print("-" * 70)
    print(f"\n💡 A* baseline: {astar_time:.2f} ms")
    
    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)
    print("""
Neden ML yavaş?
1. GPU overhead: CPU→GPU data transfer ~1-2ms
2. Single sample: Batch=1 GPU'yu verimli kullanmıyor
3. PyTorch eager mode: Her forward pass graph oluşturuyor

Çözümler:
1. ✅ CPU inference (küçük modeller için daha hızlı)
2. 🔧 TorchScript/ONNX (JIT compilation)
3. 🔧 Batch inference (aynı anda N trajectory)
4. 🔧 Quantization (INT8)
""")


if __name__ == "__main__":
    main()
