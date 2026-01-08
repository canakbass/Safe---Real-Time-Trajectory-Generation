"""
Full Pipeline Runner
====================
End-to-end pipeline for IAC 2026 Paper:
"Safe & Real-Time Trajectory Generation: A Hybrid Approach"

This script runs the complete pipeline:
    1. Generate expert trajectory dataset (RRT*)
    2. Train diffusion model
    3. Run benchmark comparison
    4. Generate publication-ready results

Usage:
    python scripts/run_full_pipeline.py --full          # Full pipeline
    python scripts/run_full_pipeline.py --benchmark     # Only benchmark (if model exists)
    python scripts/run_full_pipeline.py --quick         # Quick test (smaller dataset)

Author: H. Can Akbas
"""

import argparse
import sys
from pathlib import Path
import time
import subprocess
import shutil

# Project root
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def print_header(title: str, char: str = "="):
    """Print a formatted header."""
    print()
    print(char * 70)
    print(f"  {title}")
    print(char * 70)
    print()


def run_command(cmd: list, description: str) -> bool:
    """Run a subprocess command with error handling."""
    print(f"\n>>> Running: {description}")
    print(f"    Command: {' '.join(cmd)}\n")
    
    try:
        result = subprocess.run(cmd, check=True, cwd=str(PROJECT_ROOT))
        print(f"\n✓ {description} completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ {description} failed with code {e.returncode}")
        return False
    except FileNotFoundError:
        print(f"\n✗ Command not found: {cmd[0]}")
        return False


def check_prerequisites() -> dict:
    """Check if required files and dependencies exist."""
    status = {
        'dataset_exists': (PROJECT_ROOT / "data" / "expert_trajectories_latest.npz").exists(),
        'model_exists': (PROJECT_ROOT / "checkpoints" / "diffusion_best.pth").exists() or 
                       (PROJECT_ROOT / "checkpoints" / "diffusion_latest.pth").exists(),
        'results_dir': (PROJECT_ROOT / "results").exists(),
    }
    
    # Check Python dependencies
    try:
        import torch
        status['torch_available'] = True
        status['cuda_available'] = torch.cuda.is_available()
    except ImportError:
        status['torch_available'] = False
        status['cuda_available'] = False
    
    try:
        import numpy
        import scipy
        status['numpy_scipy'] = True
    except ImportError:
        status['numpy_scipy'] = False
    
    return status


def print_status(status: dict):
    """Print prerequisite status."""
    print("\n📊 System Status:")
    print("-" * 40)
    print(f"  PyTorch:       {'✓' if status['torch_available'] else '✗'}")
    print(f"  CUDA GPU:      {'✓' if status['cuda_available'] else '✗ (CPU mode)'}")
    print(f"  NumPy/SciPy:   {'✓' if status['numpy_scipy'] else '✗'}")
    print(f"  Dataset:       {'✓' if status['dataset_exists'] else '✗ (will generate)'}")
    print(f"  Trained Model: {'✓' if status['model_exists'] else '✗ (will train)'}")
    print("-" * 40)


def step_generate_dataset(n_samples: int = 500) -> bool:
    """Step 1: Generate expert trajectory dataset."""
    print_header("STEP 1: GENERATE EXPERT DATASET", "═")
    print(f"Generating {n_samples} expert trajectories using high-quality RRT*...")
    
    return run_command(
        [sys.executable, "scripts/generate_dataset.py", 
         "-n", str(n_samples), "-w", "4"],
        f"Dataset generation ({n_samples} samples)"
    )


def step_train_model(n_epochs: int = 100) -> bool:
    """Step 2: Train diffusion model."""
    print_header("STEP 2: TRAIN DIFFUSION MODEL", "═")
    print(f"Training for {n_epochs} epochs...")
    
    return run_command(
        [sys.executable, "scripts/train_diffusion.py",
         "--epochs", str(n_epochs), "--batch-size", "32"],
        f"Model training ({n_epochs} epochs)"
    )


def step_run_benchmark(n_scenarios: int = 100) -> bool:
    """Step 3: Run benchmark comparison."""
    print_header("STEP 3: RUN BENCHMARK", "═")
    print(f"Comparing RRT* vs Hybrid on {n_scenarios} scenarios...")
    
    return run_command(
        [sys.executable, "scripts/run_benchmark.py",
         "-n", str(n_scenarios)],
        f"Benchmark comparison ({n_scenarios} scenarios)"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Full Pipeline for IAC 2026 Paper"
    )
    
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument('--full', action='store_true',
                           help='Run full pipeline (dataset + train + benchmark)')
    mode_group.add_argument('--benchmark', action='store_true',
                           help='Only run benchmark (requires trained model)')
    mode_group.add_argument('--quick', action='store_true',
                           help='Quick test with small dataset')
    mode_group.add_argument('--train', action='store_true',
                           help='Only train model (requires dataset)')
    
    parser.add_argument('--samples', type=int, default=500,
                       help='Number of training samples (default: 500)')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Training epochs (default: 100)')
    parser.add_argument('--scenarios', type=int, default=100,
                       help='Benchmark scenarios (default: 100)')
    
    args = parser.parse_args()
    
    # Default to full pipeline
    if not any([args.full, args.benchmark, args.quick, args.train]):
        args.full = True
    
    print_header("IAC 2026 PAPER: FULL PIPELINE", "█")
    print("Safe & Real-Time Trajectory Generation:")
    print("A Hybrid Approach using Diffusion Models and Convex Optimization")
    print("\nAuthor: H. Can Akbas")
    print(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Check prerequisites
    status = check_prerequisites()
    print_status(status)
    
    if not status['numpy_scipy']:
        print("\n❌ ERROR: NumPy and SciPy are required!")
        print("Run: pip install numpy scipy")
        return 1
    
    if args.quick:
        # Quick test with minimal resources
        args.samples = 50
        args.epochs = 10
        args.scenarios = 20
        print("\n⚡ QUICK MODE: Reduced dataset, epochs, and scenarios")
    
    start_time = time.time()
    results = {}
    
    # =========================================================================
    # EXECUTE PIPELINE
    # =========================================================================
    
    if args.full or args.quick:
        # Full pipeline
        results['dataset'] = step_generate_dataset(args.samples)
        
        if results['dataset']:
            results['training'] = step_train_model(args.epochs)
        else:
            print("\n⚠️ Skipping training due to dataset generation failure")
            results['training'] = False
        
        if results['training'] or status['model_exists']:
            results['benchmark'] = step_run_benchmark(args.scenarios)
        else:
            print("\n⚠️ Skipping benchmark due to training failure")
            results['benchmark'] = False
    
    elif args.train:
        if not status['dataset_exists']:
            print("\n⚠️ Dataset not found. Generating first...")
            results['dataset'] = step_generate_dataset(args.samples)
            if not results['dataset']:
                print("❌ Cannot train without dataset!")
                return 1
        results['training'] = step_train_model(args.epochs)
    
    elif args.benchmark:
        if not status['model_exists']:
            print("\n⚠️ WARNING: No trained model found!")
            print("   Benchmark will use fallback heuristic (not recommended)")
            response = input("   Continue anyway? [y/N]: ")
            if response.lower() != 'y':
                return 0
        results['benchmark'] = step_run_benchmark(args.scenarios)
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    
    elapsed = time.time() - start_time
    
    print_header("PIPELINE COMPLETE", "█")
    print(f"Total time: {elapsed/60:.1f} minutes\n")
    
    print("Results Summary:")
    print("-" * 40)
    for step, success in results.items():
        status_icon = "✓" if success else "✗"
        print(f"  {step.capitalize():15} [{status_icon}]")
    
    # Check for output files
    results_dir = PROJECT_ROOT / "results"
    if results_dir.exists():
        print("\n📁 Generated Files:")
        for f in results_dir.glob("*"):
            print(f"    {f.name}")
    
    checkpoint_dir = PROJECT_ROOT / "checkpoints"
    if checkpoint_dir.exists():
        print("\n🧠 Model Checkpoints:")
        for f in checkpoint_dir.glob("*.pth"):
            size_mb = f.stat().st_size / (1024 * 1024)
            print(f"    {f.name} ({size_mb:.1f} MB)")
    
    print("\n" + "=" * 70)
    if all(results.values()):
        print("✅ All steps completed successfully!")
        print("\nNext steps:")
        print("  1. Review results in ./results/")
        print("  2. Check benchmark_results_latest.csv for metrics")
        print("  3. Use visualization tools to generate figures for paper")
    else:
        print("⚠️ Some steps failed. Review output above.")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
