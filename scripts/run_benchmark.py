"""
Benchmark Runner Script
=======================
Main entry point for running trajectory solver benchmarks.

Usage:
    python run_benchmark.py --n-trials 100 --seed 42 --output results/

This script orchestrates the full benchmark pipeline:
    1. Initialize environment and solvers
    2. Run N trials for each solver
    3. Collect metrics via EnergyAuditor
    4. Generate comparison report
"""

import argparse
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Tuple
import json
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.environment import SpaceEnv, SpaceEnvConfig, create_random_env
from src.solvers import RRTSolver, RRTConfig, HybridSolver, HybridSolverConfig
from src.auditor import EnergyAuditor, PowerProfile


def run_benchmark(
    n_trials: int = 100,
    seed: int = 42,
    output_dir: str = "./results"
) -> dict:
    """
    Run complete benchmark comparing RRT* vs Hybrid solver.
    
    Args:
        n_trials: Number of trials per solver
        seed: Random seed for reproducibility
        output_dir: Directory for output files
    
    Returns:
        Summary statistics dictionary
    """
    print("=" * 70)
    print("TRAJECTORY SOLVER BENCHMARK")
    print("Safe & Real-Time Trajectory Generation")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Trials: {n_trials}")
    print(f"  Seed: {seed}")
    print(f"  Output: {output_dir}")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Initialize random generator
    rng = np.random.default_rng(seed)
    
    # Initialize solvers
    print("\n[1/4] Initializing solvers...")
    
    rrt_solver = RRTSolver(
        config=RRTConfig(max_iterations=3000, step_size=5.0),
        n_waypoints=50
    )
    
    hybrid_solver = HybridSolver(
        config=HybridSolverConfig(
            n_diffusion_steps=50,
            slsqp_max_iter=100,
            safety_margin=1.0
        ),
        n_waypoints=50
    )
    
    # Initialize auditor with Jetson Nano power profile
    auditor = EnergyAuditor(
        power_profile=PowerProfile.jetson_nano(),
        target_tolerance=5.0,
        safety_margin=0.5
    )
    
    print(f"  RRT* Solver: {rrt_solver.name}")
    print(f"  Hybrid Solver: {hybrid_solver.name}")
    print(f"  Power Profile: {auditor.power_profile.name}")
    
    # Generate test environments
    print("\n[2/4] Generating test environments...")
    
    environments: List[SpaceEnv] = []
    for i in range(n_trials):
        env = create_random_env(
            n_obstacles=12,
            seed=int(rng.integers(0, 2**31)),
            difficulty='medium'
        )
        env.reset()
        environments.append(env)
    
    print(f"  Generated {len(environments)} environments")
    
    # Run RRT* benchmark
    print("\n[3/4] Running RRT* benchmark...")
    
    for i, env in enumerate(environments):
        if (i + 1) % 10 == 0 or i == 0:
            print(f"  Trial {i+1}/{n_trials}...", end='\r')
        
        result = rrt_solver.solve(env)
        auditor.evaluate(env, result)
    
    print(f"  Completed {n_trials} RRT* trials    ")
    
    # Run Hybrid benchmark
    print("\n[4/4] Running Hybrid solver benchmark...")
    
    for i, env in enumerate(environments):
        if (i + 1) % 10 == 0 or i == 0:
            print(f"  Trial {i+1}/{n_trials}...", end='\r')
        
        # Clone environment to ensure fair comparison
        env_clone = env.clone()
        result = hybrid_solver.solve(env_clone)
        auditor.evaluate(env_clone, result)
    
    print(f"  Completed {n_trials} Hybrid trials    ")
    
    # Generate reports
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    
    auditor.print_comparison()
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    csv_path = output_path / f"benchmark_results_{timestamp}.csv"
    auditor.save_results(str(csv_path))
    print(f"\nDetailed results saved to: {csv_path}")
    
    json_path = output_path / f"benchmark_summary_{timestamp}.json"
    auditor.save_summary(str(json_path))
    print(f"Summary saved to: {json_path}")
    
    return auditor.get_summary_statistics()


def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Trajectory Solver Benchmark - IAC 2026 Research"
    )
    
    parser.add_argument(
        '--n-trials', '-n',
        type=int,
        default=100,
        help='Number of trials per solver (default: 100)'
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
        default='./results',
        help='Output directory for results (default: ./results)'
    )
    
    args = parser.parse_args()
    
    summary = run_benchmark(
        n_trials=args.n_trials,
        seed=args.seed,
        output_dir=args.output
    )
    
    print("\nBenchmark complete!")


if __name__ == "__main__":
    main()
