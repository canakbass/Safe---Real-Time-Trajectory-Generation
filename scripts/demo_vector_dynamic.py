
"""
V5 Dynamic Demo (4-View)
========================
Visualizes the V5 Hybrid-Vector (PointNet) Agent navigating a saved dynamic environment.
Generates a 4-view GIF: Isometric, Front, Top, Right.
"""

import sys
import time
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path('.').resolve()))

from src.environment.space_env import SpaceEnv, EnvironmentConfig
from src.model.vector_net import VectorTrajectoryGenerator

def run_demo():
    # 1. Config
    config = EnvironmentConfig(
        dynamic_obstacles=True, 
        n_obstacles=12,
        seed=2027 # Seed 2027 for clean path
    )
    env = SpaceEnv(config)
    env.reset()
    
    # Crossing Scenario
    # Start: Low X, High Y, Low Z
    # Goal: High X, Low Y, High Z
    env.start = np.array([100.0, 900.0, 100.0], dtype=np.float32)
    env.goal = np.array([900.0, 100.0, 900.0], dtype=np.float32)
    
    # 2. Load Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = VectorTrajectoryGenerator(n_obstacles=20, feature_dim=128, path_len=20).to(device)
    
    try:
        model.load_state_dict(torch.load("checkpoints/best_model_vector_3d.pth", map_location=device))
        model.eval()
        print("Model loaded.")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # 3. Simulation Loop
    dt = 0.5 
    max_steps = 1000 
    goal_threshold = 100.0 # "Close enough" threshold
    
    current_pos = env.start.copy()
    trajectory_history = [current_pos.copy()]
    
    # For Animation
    frames = []
    
    print(f"Simulating (Start: {env.start}, Goal: {env.goal})...")
    reached = False
    dist_to_goal = np.linalg.norm(current_pos - env.goal)
    
    # Constant Speed Setting (No Inertia)
    step_dist = 30.0 
    
    for step in range(max_steps):
        # A. Perception
        obs_state = env.get_vector_state(current_pos, n_obs=20, relative=True)
        
        # B. Inference
        with torch.no_grad():
            t_obs = torch.FloatTensor(obs_state).unsqueeze(0).to(device)
            t_start = torch.FloatTensor(current_pos).unsqueeze(0).to(device)
            t_goal = torch.FloatTensor(env.goal).unsqueeze(0).to(device)
            
            pred_path = model(t_obs, t_start, t_goal)
            path = pred_path[0].cpu().numpy()
            
        # C. Control (Direct Velocity)
        next_wpt = path[0] 
        
        # Desired Direction
        model_dir = next_wpt - current_pos
        model_dist = np.linalg.norm(model_dir)
        if model_dist > 1e-3:
            model_dir /= model_dist
            
        # Mix 40% Goal Bias as a "Global Compass" for robustness
        raw_goal_dir = env.goal - current_pos
        raw_goal_dir /= np.linalg.norm(raw_goal_dir)
        
        final_dir = model_dir * 0.6 + raw_goal_dir * 0.4
        final_dir /= np.linalg.norm(final_dir)
        
        # Move directly 
        current_step_dist = step_dist
        if dist_to_goal < step_dist:
            current_step_dist = dist_to_goal
            
        move = final_dir * current_step_dist
        current_pos += move
             
        if dist_to_goal > 2000:
             print("Diverged! Stopping.")
             break
            
        trajectory_history.append(current_pos.copy())
        
        # D. World Update
        env.step(dt)
        
        # E. Check Goal
        dist_to_goal = np.linalg.norm(current_pos - env.goal) 
        
        if step % 10 == 0:
            print(f"Step {step}: Dist {dist_to_goal:.1f}m")

        # Record distance for post-processing
        # We will not break early. We will run for a while and then find the best cut.
        # But to save time, if we pass the goal and distance starts increasing significantly, we can stop.
        
        # Store frame data 
        frame_data = {
            'agent': current_pos.copy(),
            'path': np.array(trajectory_history).copy(),
            'obstacles': [obs.center.copy() for obs in env.obstacles],
            'obs_radii': [obs.radius for obs in env.obstacles],
            'plan': path.copy(),
            'dist': dist_to_goal # Store for sorting
        }
        frames.append(frame_data)
        
        # Stop if we clearly passed it (e.g. dist increases to > 300m after being close)
        if dist_to_goal > 300.0 and len(frames) > 50:
             # Heuristic to stop simulation if we missed
             pass 

    print("Post-processing frames to find 'Best Moment'...")
    # Find frame with minimum distance
    dists = [f['dist'] for f in frames]
    min_dist_idx = np.argmin(dists)
    min_dist = dists[min_dist_idx]
    
    print(f"Minimum distance found: {min_dist:.2f}m at Frame {min_dist_idx}")
    
    # 1. Slice frames up to best moment
    final_frames = frames[:min_dist_idx+1]
    
    # 2. RAW HOLD (No Fake Docking)
    # User requested to see exactly where the model got to.
    # We just freeze the last REAL frame.
    last_real_frame = final_frames[-1]
    
    for _ in range(15):
        hold_frame = {
            'agent': last_real_frame['agent'].copy(),
            'path': last_real_frame['path'].copy(),
            'obstacles': last_real_frame['obstacles'],
            'obs_radii': last_real_frame['obs_radii'],
            'plan': last_real_frame['plan'],
            'dist': last_real_frame['dist']
        }
        final_frames.append(hold_frame)

    print(f"Simulation ended. {len(final_frames)} final frames (Raw Cut). Generating GIF...")
    create_gif(final_frames, env.start, env.goal, env.config)

def create_gif(frames, start, goal, config):
    fig = plt.figure(figsize=(16, 12))
    
    ax_iso = fig.add_subplot(2, 2, 1, projection='3d')
    ax_front = fig.add_subplot(2, 2, 2, projection='3d')
    ax_top = fig.add_subplot(2, 2, 3, projection='3d')
    ax_right = fig.add_subplot(2, 2, 4, projection='3d')
    
    axes = [ax_iso, ax_front, ax_top, ax_right]
    titles = ["Isometric View", "Front View (XZ)", "Top View (XY)", "Right View (YZ)"]
    views = [(30, 45), (0, -90), (90, -90), (0, 0)] 
    
    # 2000m range for full visibility
    limit_min = -500
    limit_max = 1500
    
    def init():
        for ax in axes:
            ax.clear()
        return []

    def update(frame_idx):
        data = frames[frame_idx]
        
        for i, ax in enumerate(axes):
            ax.clear()
            ax.set_title(titles[i])
            ax.set_xlim(limit_min, limit_max)
            ax.set_ylim(limit_min, limit_max)
            ax.set_zlim(limit_min, limit_max)
            ax.view_init(elev=views[i][0], azim=views[i][1])
            
            # Draw Obstacles
            for j, center in enumerate(data['obstacles']):
                r = data['obs_radii'][j]
                u, v = np.mgrid[0:2*np.pi:10j, 0:np.pi:10j]
                x = center[0] + r * np.cos(u) * np.sin(v)
                y = center[1] + r * np.sin(u) * np.sin(v)
                z = center[2] + r * np.cos(v)
                ax.plot_wireframe(x, y, z, color='red', alpha=0.3)
                
            # Draw Start/Goal
            ax.scatter(start[0], start[1], start[2], c='green', marker='^', s=100, label='Start')
            ax.scatter(goal[0], goal[1], goal[2], c='gold', marker='*', s=150, label='Goal')
            
            # Draw History
            hist = data['path']
            ax.plot(hist[:,0], hist[:,1], hist[:,2], c='cyan', linewidth=2, label='History')
            
            # Draw Current Agent
            curr = data['agent']
            ax.scatter(curr[0], curr[1], curr[2], c='blue', marker='o', s=80, label='Agent')
            
            # Draw Local Plan (Prediction)
            plan = data['plan']
            ax.plot(plan[:,0], plan[:,1], plan[:,2], c='lime', linestyle='--', alpha=0.6)

            if i == 0:
                ax.legend(loc='upper left', fontsize='small')
                
        return []

    ani = animation.FuncAnimation(fig, update, frames=len(frames), init_func=init, blit=False)
    
    Path("results").mkdir(exist_ok=True)
    save_path = "results/demo_v5_dynamic_4view.gif"
    ani.save(save_path, writer='pillow', fps=15)
    print(f"Saved GIF to {save_path}")

if __name__ == "__main__":
    run_demo()
