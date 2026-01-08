"""
Energy Auditor Module
=====================
Passive observer for benchmarking trajectory solvers.

Computes:
1. Success Rate (collision-free paths)
2. Fuel Cost (Delta-V)
3. Computational Energy Cost (Joules) - NOVELTY METRIC

Energy Model:
    E(J) = t_inference(s) × P_device(W)
    
    where:
    - CPU operations (RRT*, SLSQP): P_cpu = 5.0 W
    - GPU operations (Diffusion): P_gpu = 10.0 W

Reference Hardware: NVIDIA Jetson Nano (space-analog embedded system)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import json
from pathlib import Path

from ..solvers.base_solver import SolverResult, SolverType, TimingRecord
from ..environment.space_env import SpaceEnv


# =============================================================================
# POWER PROFILES
# =============================================================================

@dataclass
class PowerProfile:
    """
    Hardware power consumption profile.
    
    Models the power draw of embedded space computing hardware
    for energy consumption estimation.
    
    Reference: NVIDIA Jetson Nano Technical Specifications
    
    Attributes:
        cpu_watts: CPU power consumption in Watts
        gpu_watts: GPU power consumption in Watts
        idle_watts: Baseline idle power consumption
        name: Profile identifier
    """
    cpu_watts: float = 5.0
    gpu_watts: float = 10.0
    idle_watts: float = 1.5
    name: str = "jetson_nano"
    
    @classmethod
    def jetson_nano(cls) -> 'PowerProfile':
        """NVIDIA Jetson Nano power profile (5W mode)."""
        return cls(cpu_watts=5.0, gpu_watts=10.0, idle_watts=1.5, name="jetson_nano_5w")
    
    @classmethod
    def jetson_nano_maxn(cls) -> 'PowerProfile':
        """NVIDIA Jetson Nano MAXN power profile (10W mode)."""
        return cls(cpu_watts=7.5, gpu_watts=15.0, idle_watts=2.0, name="jetson_nano_10w")
    
    @classmethod
    def radiation_hardened(cls) -> 'PowerProfile':
        """Typical radiation-hardened space processor."""
        return cls(cpu_watts=3.0, gpu_watts=0.0, idle_watts=1.0, name="rad_hard_cpu")


# =============================================================================
# AUDIT REPORT
# =============================================================================

@dataclass
class AuditReport:
    """
    Comprehensive audit report for a single solver run.
    
    Contains all metrics required for benchmark comparison.
    
    Attributes:
        solver_type: Type of solver audited
        success: Whether trajectory is valid (collision-free + reaches target)
        collision_free: Trajectory avoids all obstacles
        reaches_target: Final waypoint within tolerance of target
        path_length: Total path length in meters
        delta_v: Total fuel consumption (m/s)
        energy_joules: Computational energy cost
        timing: Detailed timing breakdown
    """
    # Identification
    solver_type: SolverType
    trial_id: int = 0
    
    # Safety metrics
    success: bool = False
    collision_free: bool = False
    reaches_target: bool = False
    
    # Performance metrics
    path_length: float = 0.0
    delta_v: float = 0.0
    
    # Energy metrics (NOVELTY)
    energy_joules: float = 0.0
    energy_breakdown: Dict[str, float] = field(default_factory=dict)
    
    # Timing
    timing: Optional[TimingRecord] = None
    
    # Raw data reference
    trajectory: Optional[np.ndarray] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for DataFrame construction."""
        return {
            'trial_id': self.trial_id,
            'solver_type': self.solver_type.value,
            'success': self.success,
            'collision_free': self.collision_free,
            'reaches_target': self.reaches_target,
            'path_length_m': self.path_length,
            'delta_v_ms': self.delta_v,
            'energy_joules': self.energy_joules,
            'energy_cpu_j': self.energy_breakdown.get('cpu', 0.0),
            'energy_gpu_j': self.energy_breakdown.get('gpu', 0.0),
            'time_total_ms': self.timing.total_ms if self.timing else 0.0,
            'time_cpu_ms': self.timing.cpu_ms if self.timing else 0.0,
            'time_gpu_ms': self.timing.gpu_ms if self.timing else 0.0,
        }


# =============================================================================
# ENERGY AUDITOR
# =============================================================================

class EnergyAuditor:
    """
    Passive Observer for Trajectory Solver Benchmarking.
    
    The EnergyAuditor is responsible for:
        1. Evaluating solver outputs (collision check, target reached)
        2. Computing fuel cost (Delta-V)
        3. Computing computational energy cost (Joules)
        4. Aggregating results across multiple trials
    
    Energy Calculation Formula:
        ┌─────────────────────────────────────────────────────────────┐
        │  E_total = E_cpu + E_gpu                                    │
        │                                                             │
        │  E_cpu = t_cpu (s) × P_cpu (W)                              │
        │  E_gpu = t_gpu (s) × P_gpu (W)                              │
        │                                                             │
        │  For RRT* (CPU-only):                                       │
        │      E = t_rrt × 5.0 W                                      │
        │                                                             │
        │  For Hybrid (GPU + CPU):                                    │
        │      E = (t_diffusion × 10.0 W) + (t_slsqp × 5.0 W)         │
        └─────────────────────────────────────────────────────────────┘
    
    Example:
        >>> auditor = EnergyAuditor(power_profile=PowerProfile.jetson_nano())
        >>> report = auditor.evaluate(env, solver_result)
        >>> print(f"Energy: {report.energy_joules:.4f} J")
    """
    
    def __init__(
        self,
        power_profile: Optional[PowerProfile] = None,
        target_tolerance: float = 5.0,
        safety_margin: float = 0.5
    ):
        """
        Initialize the Energy Auditor.
        
        Args:
            power_profile: Hardware power consumption profile
            target_tolerance: Distance threshold for reaching target (m)
            safety_margin: Collision detection buffer (m)
        """
        self.power_profile = power_profile or PowerProfile.jetson_nano()
        self.target_tolerance = target_tolerance
        self.safety_margin = safety_margin
        
        # Results storage
        self._reports: List[AuditReport] = []
        self._trial_counter = 0
    
    def evaluate(
        self,
        env: SpaceEnv,
        result: SolverResult
    ) -> AuditReport:
        """
        Evaluate a solver result and generate audit report.
        
        This is the main entry point for benchmarking a single trial.
        
        Args:
            env: SpaceEnv instance used for the solve
            result: SolverResult from a trajectory solver
        
        Returns:
            AuditReport with all computed metrics
        """
        self._trial_counter += 1
        
        # 1. Safety evaluation
        collision_free, reaches_target = self._evaluate_safety(
            trajectory=result.trajectory,
            env=env
        )
        
        # 2. Performance metrics
        path_length = result.path_length
        delta_v = self._compute_delta_v(result.trajectory, env)
        
        # 3. Energy calculation (CORE NOVELTY)
        energy_joules, energy_breakdown = self.calculate_energy_cost(
            timing=result.timing,
            solver_type=result.solver_type
        )
        
        # 4. Assemble report
        report = AuditReport(
            solver_type=result.solver_type,
            trial_id=self._trial_counter,
            success=result.success and collision_free and reaches_target,
            collision_free=collision_free,
            reaches_target=reaches_target,
            path_length=path_length,
            delta_v=delta_v,
            energy_joules=energy_joules,
            energy_breakdown=energy_breakdown,
            timing=result.timing,
            trajectory=result.trajectory.copy() if result.trajectory is not None else None
        )
        
        self._reports.append(report)
        
        return report
    
    def calculate_energy_cost(
        self,
        timing: TimingRecord,
        solver_type: SolverType
    ) -> Tuple[float, Dict[str, float]]:
        """
        Calculate computational energy cost in Joules.
        
        ╔════════════════════════════════════════════════════════════════╗
        ║                    ENERGY CALCULATION                          ║
        ╠════════════════════════════════════════════════════════════════╣
        ║  Input:                                                        ║
        ║    timing.cpu_ms  = CPU time in milliseconds                   ║
        ║    timing.gpu_ms  = GPU time in milliseconds                   ║
        ║    P_cpu = 5.0 W  (from power profile)                         ║
        ║    P_gpu = 10.0 W (from power profile)                         ║
        ║                                                                ║
        ║  Calculation:                                                  ║
        ║    t_cpu = timing.cpu_ms / 1000  [seconds]                     ║
        ║    t_gpu = timing.gpu_ms / 1000  [seconds]                     ║
        ║                                                                ║
        ║    E_cpu = t_cpu × P_cpu  [Joules]                             ║
        ║    E_gpu = t_gpu × P_gpu  [Joules]                             ║
        ║                                                                ║
        ║    E_total = E_cpu + E_gpu  [Joules]                           ║
        ║                                                                ║
        ║  For RRT* (solver_type = CLASSICAL_RRT):                       ║
        ║    E_gpu = 0 (no GPU usage)                                    ║
        ║    E_total = t_rrt × 5.0 W                                     ║
        ║                                                                ║
        ║  For Hybrid (solver_type = HYBRID_DIFFUSION):                  ║
        ║    E_total = (t_diff × 10.0 W) + (t_slsqp × 5.0 W)             ║
        ╚════════════════════════════════════════════════════════════════╝
        
        Args:
            timing: TimingRecord with CPU and GPU times
            solver_type: Type of solver for profile selection
        
        Returns:
            Tuple of (total_energy_joules, breakdown_dict)
        """
        # Convert milliseconds to seconds
        t_cpu = timing.cpu_seconds
        t_gpu = timing.gpu_seconds
        
        # Get power values from profile
        P_cpu = self.power_profile.cpu_watts
        P_gpu = self.power_profile.gpu_watts
        
        # Calculate energy components
        E_cpu = t_cpu * P_cpu  # Joules = seconds × Watts
        E_gpu = t_gpu * P_gpu
        
        # Special handling by solver type
        if solver_type == SolverType.CLASSICAL_RRT:
            # RRT* is CPU-only, no GPU energy
            E_gpu = 0.0
            # All time is CPU time
            E_cpu = timing.total_seconds * P_cpu
        
        elif solver_type == SolverType.HYBRID_DIFFUSION:
            # Hybrid uses both GPU (diffusion) and CPU (SLSQP)
            # Use the detailed breakdown if available
            pass  # Already calculated correctly from timing
        
        E_total = E_cpu + E_gpu
        
        breakdown = {
            'cpu': E_cpu,
            'gpu': E_gpu,
            'total': E_total,
            't_cpu_s': t_cpu,
            't_gpu_s': t_gpu,
            'P_cpu_w': P_cpu,
            'P_gpu_w': P_gpu
        }
        
        return E_total, breakdown
    
    def _evaluate_safety(
        self,
        trajectory: np.ndarray,
        env: SpaceEnv
    ) -> Tuple[bool, bool]:
        """
        Evaluate trajectory safety.
        
        Returns:
            Tuple of (collision_free, reaches_target)
        """
        if trajectory is None or len(trajectory) < 2:
            return False, False
        
        # Check collisions
        has_collision, _ = env.check_trajectory_collision(
            trajectory,
            safety_margin=self.safety_margin
        )
        collision_free = not has_collision
        
        # Check target reached
        final_pos = trajectory[-1]
        dist_to_target = np.linalg.norm(final_pos - env.target_pos)
        reaches_target = dist_to_target <= self.target_tolerance
        
        return collision_free, reaches_target
    
    def _compute_delta_v(
        self,
        trajectory: np.ndarray,
        env: SpaceEnv
    ) -> float:
        """Compute Delta-V (fuel cost) for trajectory."""
        if trajectory is None or len(trajectory) < 3:
            return 0.0
        
        return env.compute_trajectory_delta_v(trajectory, dt=0.1)
    
    # =========================================================================
    # AGGREGATION METHODS
    # =========================================================================
    
    def get_summary_statistics(self) -> Dict[str, Any]:
        """
        Compute summary statistics across all trials.
        
        Returns:
            Dictionary with aggregated metrics
        """
        if not self._reports:
            return {}
        
        df = self.to_dataframe()
        
        summary = {}
        
        for solver_type in df['solver_type'].unique():
            solver_df = df[df['solver_type'] == solver_type]
            
            summary[solver_type] = {
                'n_trials': len(solver_df),
                'success_rate': solver_df['success'].mean() * 100,
                'collision_free_rate': solver_df['collision_free'].mean() * 100,
                'path_length': {
                    'mean': solver_df['path_length_m'].mean(),
                    'std': solver_df['path_length_m'].std(),
                },
                'delta_v': {
                    'mean': solver_df['delta_v_ms'].mean(),
                    'std': solver_df['delta_v_ms'].std(),
                },
                'energy_joules': {
                    'mean': solver_df['energy_joules'].mean(),
                    'std': solver_df['energy_joules'].std(),
                    'total': solver_df['energy_joules'].sum(),
                },
                'time_ms': {
                    'mean': solver_df['time_total_ms'].mean(),
                    'std': solver_df['time_total_ms'].std(),
                },
            }
        
        return summary
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert all reports to pandas DataFrame."""
        return pd.DataFrame([r.to_dict() for r in self._reports])
    
    def save_results(self, filepath: str) -> None:
        """Save results to CSV file."""
        df = self.to_dataframe()
        df.to_csv(filepath, index=False)
    
    def save_summary(self, filepath: str) -> None:
        """Save summary statistics to JSON file."""
        summary = self.get_summary_statistics()
        summary['metadata'] = {
            'timestamp': datetime.now().isoformat(),
            'power_profile': self.power_profile.name,
            'target_tolerance_m': self.target_tolerance,
            'safety_margin_m': self.safety_margin,
        }
        
        with open(filepath, 'w') as f:
            json.dump(summary, f, indent=2)
    
    def reset(self) -> None:
        """Clear all stored reports."""
        self._reports = []
        self._trial_counter = 0
    
    def print_comparison(self) -> None:
        """Print formatted comparison of solver performance."""
        summary = self.get_summary_statistics()
        
        print("\n" + "=" * 70)
        print("TRAJECTORY SOLVER BENCHMARK RESULTS")
        print(f"Power Profile: {self.power_profile.name}")
        print("=" * 70)
        
        for solver, stats in summary.items():
            if solver == 'metadata':
                continue
            
            print(f"\n[{solver.upper()}]")
            print(f"  Trials:         {stats['n_trials']}")
            print(f"  Success Rate:   {stats['success_rate']:.1f}%")
            print(f"  Path Length:    {stats['path_length']['mean']:.2f} ± {stats['path_length']['std']:.2f} m")
            print(f"  Delta-V:        {stats['delta_v']['mean']:.4f} ± {stats['delta_v']['std']:.4f} m/s")
            print(f"  Energy:         {stats['energy_joules']['mean']*1000:.4f} ± {stats['energy_joules']['std']*1000:.4f} mJ")
            print(f"  Time:           {stats['time_ms']['mean']:.2f} ± {stats['time_ms']['std']:.2f} ms")
        
        print("\n" + "=" * 70)


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    'EnergyAuditor',
    'AuditReport',
    'PowerProfile',
]
