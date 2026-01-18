# Hybrid AI Trajectory Planning

ML-based spacecraft trajectory generation for IAC 2026 paper.

## Results

| Method | Success Rate | Latency | Speedup |
|--------|-------------|---------|---------|
| A* (Baseline) | 100% | 4.35 ms | 1x |
| **SimpleMLP Hybrid** | **96%** | **1.26 ms** | **3.5x** |
| Attention Hybrid | 96% | 1.99 ms | 2.2x |

**Trade-off:** 4% success rate loss for 3.5x speed gain.

## Usage

`ash
# Full benchmark (trains models + tests)
python scripts/benchmark.py

# Quick timing test (uses pre-trained models)
python scripts/timing_test.py
`

## Project Structure

`
src/environment/space_env.py  # 2D environment with A* solver
scripts/benchmark.py          # Training & benchmark
scripts/timing_test.py        # Quick timing test
checkpoints/                  # Model weights (.pth)
data/training_10k.npz         # Training data cache
results/                      # Benchmark results
`

## Environment

- 1000m x 1000m area
- 6 circular obstacles (20-50m radius)
- 50 waypoint trajectories
- A* baseline solver

## Models

| Model | Params | GPU Time |
|-------|--------|----------|
| SimpleMLP | 673K | 1.26 ms |
| Attention | 663K | 1.99 ms |
| UNet | 1.14M | 2.18 ms |
| DeepCNN | 1.33M | 2.57 ms |

## Requirements

- Python 3.10+
- PyTorch 2.0+ (CUDA recommended)
- NumPy, tqdm
