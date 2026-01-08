"""
Expert Trajectory Dataset Generator (Improved)
==============================================
Layer 0: Generates high-quality expert trajectories for training.

This improved version:
- Uses higher RRT* iterations for better paths
- Only saves SUCCESSFUL collision-free trajectories
- Includes progress tracking and quality metrics
- Outputs PyTorch-ready .npz format

Usage:
    python scripts/generate_dataset.py --n-samples 1000 --seed 42
"""

import argparse
import sys
import numpy as np
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import json
import shutil

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.environment import SpaceEnv, SpaceEnvConfig
from src.solvers import RRTSolver, RRTConfig


def generate_expert_dataset(
    n_samples: int = 1000,
    n_waypoints: int = 50,
    map_resolution: int = 64,
    seed: int = 42,
    output_dir: str = "./data",
    rrt_iterations: int = 5000,
    difficulty: str = "medium",
) -> str:
    """
    Generate expert trajectory dataset using RRT*.
    
    Only saves successful, collision-free trajectories.
    """
    print("=" * 70)
    print("EXPERT TRAJECTORY DATASET GENERATOR")
    print("Layer 0: Training Data Pipeline")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Target samples: {n_samples}")
    print(f"  Waypoints: {n_waypoints}")
    print(f"  Map resolution: {map_resolution}x{map_resolution}")
    print(f"  RRT* iterations: {rrt_iterations}")
    print(f"  Difficulty: {difficulty}")
    print(f"  Seed: {seed}")
    print()
    
    # Initialize
    rng = np.random.default_rng(seed)
    
    # High-quality RRT* config
    rrt_config = RRTConfig(
        max_iterations=rrt_iterations,
        step_size=4.0,
        goal_bias=0.15,
        goal_tolerance=3.0,
        rewire_radius=20.0,
        collision_resolution=0.3,
    )
    
    solver = RRTSolver(config=rrt_config, n_waypoints=n_waypoints)
    
    # Difficulty settings
    difficulty_params = {
        'easy': {'n_obstacles': 6, 'min_r': 2.0, 'max_r': 5.0},
        'medium': {'n_obstacles': 10, 'min_r': 3.0, 'max_r': 7.0},
        'hard': {'n_obstacles': 15, 'min_r': 3.0, 'max_r': 8.0},
    }
    params = difficulty_params.get(difficulty, difficulty_params['medium'])
    
    # Storage
    trajectories = []
    obstacle_maps = []
    start_positions = []
    target_positions = []
    
    # Statistics
    stats = {
        'total_attempts': 0,
        'successful': 0,
        'failed_rrt': 0,
        'failed_collision': 0,
    }
    
    solve_times = []
    path_lengths = []
    
    # Progress bar
    pbar = tqdm(total=n_samples, desc="Generating expert trajectories")
    
    while len(trajectories) < n_samples:
        stats['total_attempts'] += 1
        
        # Create random environment
        env_seed = int(rng.integers(0, 2**31))
        config = SpaceEnvConfig(
            n_obstacles=params['n_obstacles'],
            min_obstacle_radius=params['min_r'],
            max_obstacle_radius=params['max_r'],
            seed=env_seed,
        )
        env = SpaceEnv(config)
        env.reset()
        
        # Solve with RRT*
        result = solver.solve(env)
        
        if not result.success:
            stats['failed_rrt'] += 1
            continue
        
        # Verify collision-free
        has_collision, _ = env.check_trajectory_collision(
            result.trajectory, safety_margin=0.5
        )
        
        if has_collision:
            stats['failed_collision'] += 1
            continue
        
        # Verify reaches target
        final_dist = np.linalg.norm(result.trajectory[-1] - env.target_pos)
        if final_dist > 5.0:
            stats['failed_rrt'] += 1
            continue
        
        # SUCCESS - Store data
        trajectories.append(result.trajectory)
        obstacle_maps.append(env.get_obstacle_map(map_resolution))
        start_positions.append(env.start_pos.copy())
        target_positions.append(env.target_pos.copy())
        
        solve_times.append(result.timing.total_ms)
        path_lengths.append(result.path_length)
        
        stats['successful'] += 1
        pbar.update(1)
    
    pbar.close()
    
    # Stats
    stats['avg_solve_time_ms'] = float(np.mean(solve_times))
    stats['avg_path_length'] = float(np.mean(path_lengths))
    stats['success_rate'] = stats['successful'] / stats['total_attempts'] * 100
    
    print(f"\n✓ Generation complete!")
    print(f"  Success rate: {stats['success_rate']:.1f}%")
    print(f"  Total attempts: {stats['total_attempts']}")
    print(f"  Avg solve time: {stats['avg_solve_time_ms']:.0f} ms")
    print(f"  Avg path length: {stats['avg_path_length']:.1f} m")
    
    # Convert to arrays
    trajectories = np.array(trajectories, dtype=np.float32)
    obstacle_maps = np.array(obstacle_maps, dtype=np.float32)
    start_positions = np.array(start_positions, dtype=np.float32)
    target_positions = np.array(target_positions, dtype=np.float32)
    
    print(f"\nDataset shapes:")
    print(f"  trajectories: {trajectories.shape}")
    print(f"  obstacle_maps: {obstacle_maps.shape}")
    
    # Save
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"expert_trajectories_{n_samples}_{timestamp}.npz"
    filepath = output_path / filename
    
    np.savez_compressed(
        filepath,
        trajectories=trajectories,
        obstacle_maps=obstacle_maps,
        start_positions=start_positions,
        target_positions=target_positions,
        metadata=json.dumps({
            'n_samples': n_samples,
            'n_waypoints': n_waypoints,
            'map_resolution': map_resolution,
            'stats': stats,
            'timestamp': datetime.now().isoformat(),
        })
    )
    
    print(f"\n✓ Saved to: {filepath}")
    
    # Copy as "latest"
    latest_path = output_path / "expert_trajectories_latest.npz"
    if latest_path.exists():
        latest_path.unlink()
    shutil.copy(filepath, latest_path)
    print(f"✓ Also saved as: {latest_path}")
    
    return str(filepath)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate Expert Trajectory Dataset"
    )
    
    parser.add_argument(
        '--n-samples', '-n',
        type=int,
        default=1000,
        help='Number of expert trajectories (default: 1000)'
    )
    
    parser.add_argument(
        '--seed', '-s',
        type=int,
        default=42,
        help='Random seed (default: 42)'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='./data',
        help='Output directory (default: ./data)'
    )
    
    parser.add_argument(
        '--n-waypoints', '-w',
        type=int,
        default=50,
        help='Waypoints per trajectory (default: 50)'
    )
    
    parser.add_argument(
        '--rrt-iterations',
        type=int,
        default=5000,
        help='RRT* max iterations (default: 5000)'
    )
    
    parser.add_argument(
        '--difficulty',
        type=str,
        choices=['easy', 'medium', 'hard'],
        default='medium',
        help='Environment difficulty (default: medium)'
    )
    
    args = parser.parse_args()
    
    generate_expert_dataset(
        n_samples=args.n_samples,
        n_waypoints=args.n_waypoints,
        seed=args.seed,
        output_dir=args.output,
        rrt_iterations=args.rrt_iterations,
        difficulty=args.difficulty,
    )


if __name__ == "__main__":
    main()
