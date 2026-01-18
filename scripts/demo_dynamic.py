
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
    max_steps = 200
    
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
        temp_env_start = env.start.copy() 
        env.start = current_pos # Update start for planner
        
        # Get Plan
        out = pipeline.predict(env.get_downsampled_grid(), env.start, env.goal, env.obstacles)
        path = out['path'] # Repaired, safe path
        
        # Restore env.start (cleanliness, though technically we moved)
        # Actually, for the simulation logic, updating env.start IS the right way.
        
        # B. Act (Move 1 step along the plan)
        # We take the point that corresponds to t + dt? 
        # The path has 20 points. Let's take the 1st point after start.
        if len(path) > 1:
            next_pos = path[1] # Move towards first waypoint
        else:
            next_pos = path[0]
            
        # Physics Move of Agent (Simple Teleport to waypoint for demo)
        # In reality: Apply Force -> Accel -> Vel -> Pos
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
        print(f"Step {step}: Dist to Goal: {dist_to_goal:.1f}m")
        if dist_to_goal < 20.0:
            print("🏆 GOAL REACHED!")
            break
            
    # --- VISUALIZATION (GIF) ---
    print("Generating Animation...")
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    def update(frame_idx):
        ax.clear()
        ax.set_title(f"V3 Dynamic Replanning - T={frame_idx*dt:.1f}s")
        ax.set_xlim(0, 1000); ax.set_ylim(0, 1000); ax.set_zlim(0, 1000)
        
        # 1. Draw Obstacles (Moving)
        obs_list = history['obstacles'][frame_idx]
        for (center, radius) in obs_list:
            u, v = np.mgrid[0:2*np.pi:10j, 0:np.pi:6j]
            x = radius * np.cos(u) * np.sin(v) + center[0]
            y = radius * np.sin(u) * np.sin(v) + center[1]
            z = radius * np.cos(v) + center[2]
            ax.plot_wireframe(x, y, z, color="red", alpha=0.2)
            
        # 2. Draw Agent History (Trail)
        agent_hist = np.array(history['agent'][:frame_idx+1])
        ax.plot(agent_hist[:,0], agent_hist[:,1], agent_hist[:,2], 'b-', linewidth=1, label='Flown Path')
        
        # 3. Draw Current Agent
        curr = history['agent'][frame_idx]
        ax.scatter(*curr, color='blue', s=50, label='Agent')
        
        # 4. Draw Current Intention (The Plan) - Optional, maybe too cluttered?
        # Let's show it faintly to show "Thinking"
        if frame_idx < len(history['plan']):
            plan = history['plan'][frame_idx]
            ax.plot(plan[:,0], plan[:,1], plan[:,2], 'g--', alpha=0.5, label='Current Plan')
            
        # Goal
        ax.scatter(*env.goal, color='green', s=100, marker='*', label='Goal')
        
        ax.legend(loc='upper left')

    frames = len(history['agent'])
    anim = animation.FuncAnimation(fig, update, frames=frames, interval=100) # 10fps
    
    Path("results").mkdir(exist_ok=True)
    anim.save('results/dynamic_chase.gif', writer='pillow', fps=10)
    print("Done! Saved to results/dynamic_chase.gif")

if __name__ == "__main__":
    run_dynamic_demo()
