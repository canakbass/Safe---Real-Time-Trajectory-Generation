# Software Architecture & Design Document

## Project Title
**"Safe & Real-Time Trajectory Generation: A Hybrid Approach using Consistent Diffusion Models and Convex Optimization"**

**Document Version:** 1.0  
**Target Conference:** IAC 2026  
**Date:** January 8, 2026  

---

## 1. Executive Summary

This document defines the software architecture for a hybrid trajectory generation system that combines probabilistic AI inference (1D Temporal Diffusion Models) with deterministic mathematical optimization (SLSQP) to achieve real-time, energy-efficient, and provably safe spacecraft path planning.

---

## 2. System Context & Constraints

### 2.1 Hardware Power Model (Theoretical Embedded Profile)

| Component | Power Draw (W) | Use Case |
|-----------|---------------|----------|
| CPU (ARM Cortex-A57) | 5.0 | Classical solvers (RRT*), SLSQP refinement |
| GPU (Maxwell 128-core) | 10.0 | Diffusion model inference |
| Idle State | 1.5 | Baseline power consumption |

*Reference Hardware: NVIDIA Jetson Nano (space-analog embedded system)*

### 2.2 Safety Invariant

```
∀ τ_final : collision_check(τ_final, obstacles) = TRUE
```

The AI output τ_diff is **probabilistic** and **NOT flight-ready**. Only after SLSQP refinement is the trajectory certified safe.

---

## 3. Architectural Layers

```
┌─────────────────────────────────────────────────────────────────┐
│                    LAYER 0: OFFLINE TRAINING                    │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────────────┐  │
│  │ DataGenerator│───▶│ RRT* Solver │───▶│ Expert Trajectories │  │
│  │  (Random Maps)│    │  (N=10,000) │    │   (.npz Dataset)    │  │
│  └─────────────┘    └─────────────┘    └─────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                  LAYER 1: THE SIMULATION (WORLD)                │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                      SpaceEnv (Gymnasium)                │    │
│  │  • start_pos: np.ndarray[2]                              │    │
│  │  • target_pos: np.ndarray[2]                             │    │
│  │  • obstacles: List[CircularObstacle]                     │    │
│  │  • dynamics: KinematicPointMass                          │    │
│  └─────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                   LAYER 2: THE SOLVERS (BRAINS)                 │
│  ┌──────────────────┐         ┌────────────────────────────┐    │
│  │   RRTSolver      │         │      HybridSolver          │    │
│  │   (Baseline)     │         │      (Proposed)            │    │
│  │                  │         │  ┌──────────────────────┐  │    │
│  │  CPU-Only        │         │  │ DiffusionModel (GPU) │  │    │
│  │  Power: 5W       │         │  │    τ_diff output     │  │    │
│  │                  │         │  └──────────┬───────────┘  │    │
│  │                  │         │             │ warm-start   │    │
│  │                  │         │             ▼              │    │
│  │                  │         │  ┌──────────────────────┐  │    │
│  │                  │         │  │ SLSQP Optimizer(CPU) │  │    │
│  │                  │         │  │    τ_safe output     │  │    │
│  │                  │         │  └──────────────────────┘  │    │
│  └──────────────────┘         └────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                  LAYER 3: THE AUDITOR (BENCHMARKING)            │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                     EnergyAuditor                        │    │
│  │  • success_rate: float                                   │    │
│  │  • fuel_cost_delta_v: float                              │    │
│  │  • energy_joules: float = time_s × power_w               │    │
│  └─────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
```

---

## 4. Use Case Analysis

### 4.1 Actors

| Actor | Type | Description |
|-------|------|-------------|
| Researcher | Primary | Initiates benchmark runs, configures parameters |
| SimulationEngine | System | Generates randomized problem instances |
| SolverA (RRT*) | System | Baseline classical trajectory solver |
| SolverB (Hybrid) | System | Proposed AI-augmented trajectory solver |
| EnergyAuditor | System | Passive observer collecting metrics |

### 4.2 Primary Use Case: "Compare Trajectory Performance"

**Preconditions:**
- Diffusion model weights loaded (`.pt` file)
- Power profile configured
- Number of trials (N) specified

**Flow of Events (Sunny Day Scenario):**

```
1. [Researcher] → Invoke benchmark(n_trials=100)
2. [SimulationEngine] → FOR each trial i in [1, N]:
   2.1  Generate random SpaceEnv instance (start, target, obstacles)
   2.2  Clone environment state for fair comparison
   
3. [SolverA: RRT*] → 
   3.1  START CPU timer
   3.2  Execute RRT* pathfinding
   3.3  STOP CPU timer → record t_rrt
   3.4  Return τ_rrt
   
4. [SolverB: Hybrid] →
   4.1  START GPU timer
   4.2  Encode obstacle map → latent representation
   4.3  Run Diffusion Model reverse process (T=50 steps)
   4.4  STOP GPU timer → record t_diff
   4.5  START CPU timer
   4.6  Pass τ_diff as x0 to SLSQP optimizer
   4.7  Run constrained optimization (max_iter=100)
   4.8  STOP CPU timer → record t_opt
   4.9  Return τ_safe
   
5. [EnergyAuditor] →
   5.1  Compute collision_free(τ_rrt), collision_free(τ_safe)
   5.2  Compute delta_v(τ_rrt), delta_v(τ_safe)
   5.3  Compute E_rrt = t_rrt × P_cpu
   5.4  Compute E_hybrid = (t_diff × P_gpu) + (t_opt × P_cpu)
   5.5  Append metrics to results DataFrame
   
6. [Researcher] ← Receive aggregated benchmark report
```

**Postconditions:**
- CSV file with per-trial metrics generated
- Statistical summary (mean, std, 95% CI) computed

---

## 5. Data Flow Specification

### 5.1 Obstacle Map Encoding Pipeline

```
┌──────────────┐      ┌─────────────────┐      ┌──────────────────┐
│   SpaceEnv   │      │  ObstacleEncoder │      │ DiffusionModel   │
│              │      │                 │      │                  │
│ obstacles:   │      │ Binary Grid     │      │ Condition Vector │
│ List[Circle] │─────▶│ (64×64 px)      │─────▶│ c ∈ ℝ^128        │
│              │      │                 │      │                  │
└──────────────┘      └─────────────────┘      └──────────────────┘
                                                       │
                                                       ▼
┌──────────────┐      ┌─────────────────┐      ┌──────────────────┐
│ SLSQP Output │      │ SLSQP Optimizer │      │ Diffusion Output │
│              │      │                 │      │                  │
│ τ_safe:      │◀─────│ x0 = τ_diff     │◀─────│ τ_diff:          │
│ (N×2) safe   │      │ constraints=g() │      │ (N×2) raw        │
│              │      │                 │      │                  │
└──────────────┘      └─────────────────┘      └──────────────────┘
                              │
                              ▼
                      ┌─────────────────┐
                      │  EnergyAuditor  │
                      │                 │
                      │ • t_gpu, t_cpu  │
                      │ • E = Σ(t × P)  │
                      │ • collision_ok? │
                      └─────────────────┘
```

### 5.2 Data Structures

| Structure | Type | Shape | Description |
|-----------|------|-------|-------------|
| `obstacle_map` | `np.ndarray` | `(64, 64)` | Binary occupancy grid |
| `condition_vector` | `torch.Tensor` | `(128,)` | CNN-encoded obstacle latent |
| `trajectory_raw` | `np.ndarray` | `(H, 2)` | Diffusion output (H waypoints) |
| `trajectory_safe` | `np.ndarray` | `(H, 2)` | SLSQP-refined trajectory |
| `timing_record` | `dict` | - | `{gpu_ms, cpu_ms, total_ms}` |

---

## 6. Class Hierarchy Specification

See accompanying file: `src/architecture/class_hierarchy.py`

---

## 7. Key Algorithms

### 7.1 Diffusion Warm-Start Protocol

```
Algorithm: HYBRID_SOLVE(env, model, optimizer_config)
─────────────────────────────────────────────────────
Input:  env (SpaceEnv), model (DiffusionModel), config
Output: τ_safe (collision-free trajectory)

1.  c ← ENCODE_OBSTACLES(env.obstacles)           // GPU
2.  x_T ← sample_noise(shape=(H, 2))              // GPU
3.  FOR t = T down to 1:                          // GPU
4.      ε_θ ← model.predict_noise(x_t, t, c)
5.      x_{t-1} ← DDPM_step(x_t, ε_θ, t)
6.  τ_diff ← x_0                                  // Transfer to CPU
7.  
8.  // SLSQP Refinement (CPU)
9.  x0 ← flatten(τ_diff)
10. bounds ← [(0, env.width), (0, env.height)] × H
11. constraints ← [collision_constraint(env.obstacles),
                   dynamics_constraint(env.v_max)]
12. result ← scipy.optimize.minimize(
        fun=fuel_cost,
        x0=x0,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={maxiter: 100}
    )
13. τ_safe ← reshape(result.x, (H, 2))
14. RETURN τ_safe
```

### 7.2 Energy Cost Calculation

```
Algorithm: CALCULATE_ENERGY(timing, power_profile)
──────────────────────────────────────────────────
Input:  timing {t_gpu, t_cpu}, power {P_gpu, P_cpu}
Output: E_total (Joules)

1.  E_gpu ← t_gpu × P_gpu      // Diffusion inference
2.  E_cpu ← t_cpu × P_cpu      // SLSQP optimization
3.  E_total ← E_gpu + E_cpu
4.  RETURN E_total

// For RRT* baseline:
5. E_rrt ← t_rrt × P_cpu       // CPU-only operation
```

---

## 8. Interface Contracts

### 8.1 BaseSolver Interface

```python
class BaseSolver(ABC):
    @abstractmethod
    def solve(self, env: SpaceEnv) -> SolverResult:
        """
        Returns:
            SolverResult containing:
            - trajectory: np.ndarray of shape (H, 2)
            - timing: TimingRecord
            - success: bool
        """
        pass
```

### 8.2 EnergyAuditor Interface

```python
class EnergyAuditor:
    def evaluate(self, 
                 env: SpaceEnv, 
                 result: SolverResult,
                 power_profile: PowerProfile) -> AuditReport:
        """
        Returns:
            AuditReport containing:
            - collision_free: bool
            - delta_v: float (m/s)
            - energy_joules: float
            - path_length: float (m)
        """
        pass
```

---

## 9. File Structure

```
space_trajectory_benchmark/
├── docs/
│   └── SYSTEM_ARCHITECTURE.md          # This document
├── src/
│   ├── __init__.py
│   ├── environment/
│   │   ├── __init__.py
│   │   └── space_env.py                # SpaceEnv (Gymnasium)
│   ├── solvers/
│   │   ├── __init__.py
│   │   ├── base_solver.py              # BaseSolver ABC
│   │   ├── rrt_solver.py               # RRTSolver
│   │   └── hybrid_solver.py            # HybridSolver
│   ├── models/
│   │   ├── __init__.py
│   │   ├── diffusion_model.py          # 1D Temporal Diffusion
│   │   └── obstacle_encoder.py         # CNN Encoder
│   ├── training/
│   │   ├── __init__.py
│   │   ├── data_generator.py           # DataGenerator
│   │   └── train_diffusion.py          # Training loop
│   ├── auditor/
│   │   ├── __init__.py
│   │   └── energy_auditor.py           # EnergyAuditor
│   └── utils/
│       ├── __init__.py
│       ├── collision.py                # Collision detection
│       └── dynamics.py                 # Kinematic constraints
├── configs/
│   └── default.yaml                    # Hyperparameters
├── scripts/
│   ├── generate_dataset.py             # Layer 0 execution
│   └── run_benchmark.py                # Main experiment
├── data/
│   └── expert_trajectories/            # .npz files
├── checkpoints/
│   └── diffusion_model.pt              # Trained weights
└── results/
    └── benchmark_results.csv           # Output metrics
```

---

## 10. Revision History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2026-01-08 | GNC Team | Initial architecture specification |

