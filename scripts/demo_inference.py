
"""
Demo Inference Script
=====================
Visualizes the 3D Trajectory Generation Pipeline.
Shows:
1. Obstacles (Spheres)
2. Raw Model Output (Red)
3. Repaired & Smoothed Output (Green)
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.environment.space_env import SpaceEnv, EnvironmentConfig
from src.inference.pipeline import InferencePipeline

def plot_sphere(ax, center, radius):
    """Draw a sphere on 3D axis."""
    u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
    x = radius * np.cos(u) * np.sin(v) + center[0]
    y = radius * np.sin(u) * np.sin(v) + center[1]
    z = radius * np.cos(v) + center[2]
    ax.plot_wireframe(x, y, z, color="gray", alpha=0.3)

def run_demo():
    print("Initializing Environment...")
    config = EnvironmentConfig(n_obstacles=8)
    env = SpaceEnv(config)
    env.reset(seed=101) # Seed for reproducibility
    
    print("Initializing Pipeline...")
    # Using CPU for demo compatibility
    model_path = "checkpoints/best_model_3d.pth"
    if Path(model_path).exists():
        pipeline = InferencePipeline(model_path=model_path, device="cpu")
    else:
        print(f"Warning: {model_path} not found. Using Random Weights!")
        pipeline = InferencePipeline(device="cpu")
    
    print("Running Prediction...")
    # NOTE: Since model is random, the "Raw" path will be nonsense (random noise).
    # But the "Repair" step should theoretically pull it out of obstacles if it collides.
    # To demonstrate Repair better, we can artificially inject a collision path?
    # Let's trust the pipeline: Random path might hit obstacles.
    
    # Run pipeline
    result = pipeline.predict(
        grid=env.get_downsampled_grid(), 
        start=env.start, 
        goal=env.goal,
        obstacles=env.obstacles
    )
    
    path = result['path']
    raw = result['raw_path']
    repaired = result['repaired_path']
    
    metrics = result['metrics']
    print(f"\nInference Complete:")
    print(f"Total Time: {metrics['total_time_ms']:.2f} ms")
    print(f"Model: {metrics['model_time_ms']:.2f} ms")
    print(f"Repair: {metrics['repair_time_ms']:.2f} ms")
    print(f"Smooth: {metrics['smooth_time_ms']:.2f} ms")
    
    # Visualization
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title("3D Trajectory Generation (Step 3: Robust Inference)")
    
    # 1. Plot Obstacles
    for obs in env.obstacles:
        plot_sphere(ax, obs.center, obs.radius)
        
    # 2. Plot Start/Goal
    ax.scatter(*env.start, color='green', s=100, label='Start')
    ax.scatter(*env.goal, color='red', s=100, label='Goal')
    
    # 3. Plot Paths
    # Raw
    ax.plot(raw[:,0], raw[:,1], raw[:,2], 'r--', label='Raw Model Output', alpha=0.5)
    
    # Repaired (Intermediate)
    ax.plot(repaired[:,0], repaired[:,1], repaired[:,2], 'b.-', label='Repaired (APF)', alpha=0.3)
    
    # Final Smoothed
    ax.plot(path[:,0], path[:,1], path[:,2], 'g-', linewidth=2, label='Final Smoothed')
    
    ax.set_xlim(0, 1000)
    ax.set_ylim(0, 1000)
    ax.set_zlim(0, 1000)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    
    output_file = "results/demo_step3.png"
    Path("results").mkdir(exist_ok=True)
    plt.savefig(output_file)
    print(f"\nSaved visualization to {output_file}")

if __name__ == "__main__":
    run_demo()
