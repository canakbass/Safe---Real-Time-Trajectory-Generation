
"""
Version 3: Dynamic Demo (Reactive Replanning)
=============================================
Simulates a real-time control loop:
1. World Updates (Obstacles Move)
2. Agent Senses (Get Grid)
3. Hybrid Model Plans (Fast Inference)
4. Agent Moves (One step along path)
5. Repeat
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import torch
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path('.').resolve()))

from src.environment.space_env import SpaceEnv, EnvironmentConfig
from src.inference.pipeline import InferencePipeline

def run_dynamic_demo():
    print("🚀 STARTING V3 DYNAMIC DEMO (Reactive Replanning)...")
    
    # 1. Setup Dynamic Environment
    config = EnvironmentConfig(
        n_obstacles=15, 
        dynamic_obstacles=True, 
        max_velocity=80.0 # Fast moving obstacles!
    )
    env = SpaceEnv(config)
    env.reset(seed=2026) # IAC Year 
    
    # 2. Load Fast Model
    pipeline = InferencePipeline(model_path="checkpoints/best_model_3d.pth", device="cpu")
    
    # Simulation Parameters
    dt = 0.1 # 100ms time step (10Hz control loop)
    max_steps = 400 # Longer duration for demo
    
    # Tracking History for Animation
    history = {
        'agent': [],
        'obstacles': [], # List of list of (center, radius)
        'plan': [] # The planned path at each step
    }
    
    current_pos = env.start.copy()
    history['agent'].append(current_pos.copy())
    
    print("running simulation loop...")
    for step in range(max_steps):
        # A. Sense & Plan (Reactive)
        # In a real system, we'd use current_pos as start, but our model expects 'env.start'
        # So we temporarily hack: We act as if 'env.start' = current_pos for the planner
        env.start = current_pos # Update start for planner
        
        # Get Plan
        out = pipeline.predict(env.get_downsampled_grid(), env.start, env.goal, env.obstacles)
        path = out['path'] # Repaired, safe path
        
        # B. Act (Move 1 step along the plan)
        # We take the 1st point after start.
        if len(path) > 1:
            next_pos = path[1] # Move towards first waypoint
        else:
            next_pos = path[0]
            
        # Physics Move of Agent (Simple Teleport to waypoint for demo)
        current_pos = next_pos
        
        # C. Update World
        env.step(dt)
        
        # D. Record Stats
        history['agent'].append(current_pos.copy())
        # Store all obstacle positions at this frame
        obs_snap = [(o.center.copy(), o.radius) for o in env.obstacles]
        history['obstacles'].append(obs_snap)
        history['plan'].append(path.copy())
        
        # Check Goal
        dist_to_goal = np.linalg.norm(current_pos - env.goal)
        if step % 10 == 0:
            print(f"Step {step}: Dist to Goal: {dist_to_goal:.1f}m")
            
        if dist_to_goal < 20.0:
            print("🏆 GOAL REACHED!")
            break
            
    # --- VISUALIZATION (MULTI-ANGLE GIFS) ---
    print("Generating Multi-Angle Animations...")
    
    # Ensure consistency
    min_len = min(len(history['agent']), len(history['obstacles']), len(history['plan']))
    print(f"Total Frames: {min_len}")
    
    # Viewpoints: (Elev, Azim, Name)
    views = [
        (30, 45, 'iso'),  # Isometric (Standard)
        (90, -90, 'top'), # Top-Down (Like a Map)
        (0, 90, 'side')   # Side View
    ]
    
    for (elev, azim, name) in views:
        print(f"Rendering {name} view...")
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        def update(frame_idx):
            if frame_idx >= min_len: return
            
            ax.clear()
            ax.set_title(f"Dynamic Dodge ({name.upper()}) - T={frame_idx*dt:.1f}s")
            ax.set_xlim(0, 1000); ax.set_ylim(0, 1000); ax.set_zlim(0, 1000)
            ax.view_init(elev=elev, azim=azim) # Set Camera Angle
            
            # 1. Obstacles
            obs_list = history['obstacles'][frame_idx]
            for (center, radius) in obs_list:
                u, v = np.mgrid[0:2*np.pi:8j, 0:np.pi:4j]
                x = radius * np.cos(u) * np.sin(v) + center[0]
                y = radius * np.sin(u) * np.sin(v) + center[1]
                z = radius * np.cos(v) + center[2]
                ax.plot_wireframe(x, y, z, color="red", alpha=0.3)
                
            # 2. Agent Trail
            agent_hist = np.array(history['agent'][:frame_idx+1])
            ax.plot(agent_hist[:,0], agent_hist[:,1], agent_hist[:,2], 'b-', linewidth=2)
            ax.scatter(*agent_hist[-1], color='blue', s=80, edgecolors='white', label='Agent') 
            
            # 3. Plan (Faint)
            if frame_idx < len(history['plan']):
                 p = history['plan'][frame_idx]
                 ax.plot(p[:,0], p[:,1], p[:,2], 'g--', alpha=0.3)
                
            ax.scatter(*env.goal, color='green', marker='*', s=150, label='Goal')
            
            # Remove Panes for cleaner look
            ax.xaxis.pane.fill = False
            ax.yaxis.pane.fill = False
            ax.zaxis.pane.fill = False
            
            # Simple Legend (only once to avoid clutter)
            if frame_idx == 0:
                ax.legend()

        anim = animation.FuncAnimation(fig, update, frames=min_len, interval=100)
        
        Path("results").mkdir(exist_ok=True)
        anim.save(f'results/dynamic_{name}.gif', writer='pillow', fps=10)
        plt.close(fig) # Free memory
        
    print("All GIFs saved!")

if __name__ == "__main__":
    run_dynamic_demo()
