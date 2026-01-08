"""
Dataset Generation Script
=========================
Layer 0: Offline Training Data Pipeline

Generates expert trajectory datasets for training the Diffusion Model.

Usage:
    python generate_dataset.py --n-samples 10000 --seed 42 --output data/
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.training import DataGenerator, DataGeneratorConfig
from src.solvers import RRTConfig


def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Expert Trajectory Dataset Generator - Layer 0"
    )
    
    parser.add_argument(
        '--n-samples', '-n',
        type=int,
        default=10000,
        help='Number of expert trajectories to generate (default: 10000)'
    )
    
    parser.add_argument(
        '--seed', '-s',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='./data/expert_trajectories',
        help='Output directory (default: ./data/expert_trajectories)'
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
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("EXPERT TRAJECTORY DATASET GENERATOR")
    print("Layer 0: Offline Training Data Pipeline")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Target samples: {args.n_samples}")
    print(f"  Waypoints: {args.n_waypoints}")
    print(f"  RRT* iterations: {args.rrt_iterations}")
    print(f"  Seed: {args.seed}")
    print(f"  Output: {args.output}")
    print()
    
    # Configure generator
    config = DataGeneratorConfig(
        n_samples=args.n_samples,
        n_waypoints=args.n_waypoints,
        seed=args.seed,
        output_dir=args.output,
        rrt_config=RRTConfig(max_iterations=args.rrt_iterations)
    )
    
    # Generate
    generator = DataGenerator(config)
    generator.generate(show_progress=True)
    
    # Save
    filepath = generator.save()
    
    print("\nDataset generation complete!")
    print(f"Output: {filepath}")
    
    # Print statistics
    stats = generator.get_statistics()
    print(f"\nStatistics:")
    print(f"  Success rate: {stats['successful'] / stats['total_attempts'] * 100:.1f}%")
    print(f"  By difficulty: {stats['by_difficulty']}")


if __name__ == "__main__":
    main()
