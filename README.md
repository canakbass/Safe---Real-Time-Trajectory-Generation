# Space Trajectory Benchmark
## Safe & Real-Time Trajectory Generation: A Hybrid Approach using Consistent Diffusion Models and Convex Optimization

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![IAC 2026](https://img.shields.io/badge/Conference-IAC%202026-green.svg)](https://www.iafastro.org/)

Research implementation for comparing classical (RRT*) vs. hybrid AI (Diffusion + SLSQP) trajectory generation approaches for spacecraft path planning.

---

## 🎯 Project Goal

Demonstrate that a **Hybrid AI** approach (Diffusion Model warm-starting a mathematical optimizer) outperforms Classical Methods (RRT*) in terms of:
- **Inference latency** (milliseconds)
- **Computational energy efficiency** (Joules)
- While maintaining **strict safety guarantees**

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                     LAYER 0: OFFLINE TRAINING                       │
│  DataGenerator → RRT* Solver → Expert Trajectories (.npz)           │
└─────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    LAYER 1: THE SIMULATION                          │
│  SpaceEnv (Gymnasium) - 2D Space, Kinematic Spacecraft, Obstacles   │
└─────────────────────────────────────────────────────────────────────┘
                                  │
                    ┌─────────────┴─────────────┐
                    ▼                           ▼
┌───────────────────────────┐   ┌───────────────────────────────────┐
│   SOLVER A: RRT*          │   │   SOLVER B: HYBRID (Proposed)     │
│   • CPU-only (5W)         │   │   • Diffusion Model (GPU, 10W)    │
│   • Reliable but slow     │   │   • SLSQP Refinement (CPU, 5W)    │
│                           │   │   • Fast AND safe                 │
└───────────────────────────┘   └───────────────────────────────────┘
                    │                           │
                    └─────────────┬─────────────┘
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    LAYER 3: THE AUDITOR                             │
│  EnergyAuditor - Success Rate, Delta-V, Energy (Joules)             │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/your-org/space-trajectory-benchmark.git
cd space-trajectory-benchmark

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: .\venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

```txt
numpy>=1.21.0
scipy>=1.7.0
gymnasium>=0.28.0
pandas>=1.3.0
matplotlib>=3.4.0
tqdm>=4.62.0
pyyaml>=5.4.0
torch>=2.0.0  # For production diffusion model
```

---

## 🚀 Quick Start

### 1. Generate Expert Trajectories (Layer 0)

```bash
python scripts/generate_dataset.py --n-samples 1000 --seed 42
```

### 2. Run Benchmark Comparison

```bash
python scripts/run_benchmark.py --n-trials 100 --seed 42
```

### 3. View Results

Results are saved to `./results/`:
- `benchmark_results_YYYYMMDD_HHMMSS.csv` - Per-trial metrics
- `benchmark_summary_YYYYMMDD_HHMMSS.json` - Aggregated statistics

---

## 📐 Key Metrics

| Metric | Formula | Description |
|--------|---------|-------------|
| **Success Rate** | `Σ(collision_free ∧ reaches_target) / N` | Percentage of valid trajectories |
| **Delta-V** | `∫ ||a(t)|| dt` | Fuel consumption (m/s) |
| **Energy (Joules)** | `E = t_cpu × P_cpu + t_gpu × P_gpu` | Computational energy cost |

### Power Profile (Jetson Nano)

| Component | Power (W) |
|-----------|-----------|
| CPU | 5.0 |
| GPU | 10.0 |
| Idle | 1.5 |

---

## 🔬 Technical Details

### Hybrid Solver Algorithm

```
HYBRID_SOLVE(env):
    // STEP 1: GPU Inference
    obstacle_map ← env.get_obstacle_map(64×64)
    condition ← encode(obstacle_map, start, target)
    τ_diff ← DiffusionModel.generate(condition, T=50)
    
    // STEP 2: CPU Refinement (SAFETY CRITICAL)
    x0 ← flatten(τ_diff)  // Warm-start
    constraints ← [collision_avoidance, dynamics_limits]
    result ← scipy.minimize(fuel_cost, x0, method='SLSQP')
    τ_safe ← reshape(result.x)
    
    RETURN τ_safe  // ALWAYS return SLSQP output
```

### Energy Calculation

```python
def calculate_energy(timing, power_profile):
    E_cpu = timing.cpu_seconds * power_profile.cpu_watts
    E_gpu = timing.gpu_seconds * power_profile.gpu_watts
    E_total = E_cpu + E_gpu
    return E_total  # Joules
```

---

## 📁 Project Structure

```
space_trajectory_benchmark/
├── docs/
│   └── SYSTEM_ARCHITECTURE.md
├── src/
│   ├── environment/
│   │   └── space_env.py        # Gymnasium environment
│   ├── solvers/
│   │   ├── base_solver.py      # Abstract interface
│   │   ├── rrt_solver.py       # Solver A (Baseline)
│   │   └── hybrid_solver.py    # Solver B (Proposed)
│   ├── auditor/
│   │   └── energy_auditor.py   # Benchmarking
│   └── training/
│       └── data_generator.py   # Layer 0
├── configs/
│   └── default.yaml            # Hyperparameters
├── scripts/
│   ├── generate_dataset.py
│   └── run_benchmark.py
├── data/                       # Generated datasets
├── checkpoints/                # Model weights
└── results/                    # Benchmark outputs
```

---

## 📊 Expected Results

Based on preliminary experiments:

| Solver | Success Rate | Avg. Time | Avg. Energy |
|--------|-------------|-----------|-------------|
| RRT* | ~85% | 500-2000 ms | 2.5-10 J |
| Hybrid | ~90% | 50-100 ms | 0.5-1.0 J |

*Note: Results depend on environment complexity and hardware.*

---

## 🔒 Safety Guarantee

The **Hybrid Solver** maintains strict safety through a two-stage approach:

1. **Diffusion Model** (probabilistic): Generates a rough trajectory that may violate constraints
2. **SLSQP Optimizer** (deterministic): Refines the trajectory to **explicitly satisfy** collision avoidance constraints

The AI output `τ_diff` is **NEVER** used directly. Only the SLSQP-refined `τ_safe` is returned.

---

## � Future Work

The following limitations are acknowledged and planned for future versions:

| Limitation | Current State | Planned Update |
|------------|---------------|----------------|
| **3D Space** | Currently limited to 2D for proof-of-concept | 3D kinematics and full SE(3) dynamics will be added in v2.0 |
| **Dynamic Obstacles** | Current obstacles are static | Moving debris prediction and tracking is planned for future iterations |

---

## 📝 Citation

```bibtex
@inproceedings{trajectory2026iac,
  title={Safe \& Real-Time Trajectory Generation: A Hybrid Approach using 
         Consistent Diffusion Models and Convex Optimization},
  author={Akbas, H. Can},
  booktitle={International Astronautical Congress (IAC)},
  year={2026}
}
```

---

## 👤 Author

**H. Can Akbas**  

---

**Target Conference:** International Astronautical Congress (IAC) 2026
