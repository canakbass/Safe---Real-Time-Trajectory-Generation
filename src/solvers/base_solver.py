"""
Base Solver Module
==================
Abstract base class defining the solver interface for trajectory generation.

All solvers (RRT*, Hybrid AI) must implement this interface to ensure
consistent benchmarking and auditing.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from enum import Enum
import numpy as np
import time


# =============================================================================
# DATA STRUCTURES
# =============================================================================

class SolverType(Enum):
    """Enumeration of solver types for identification."""
    CLASSICAL_RRT = "rrt_star"
    HYBRID_DIFFUSION = "hybrid_diffusion"
    PURE_OPTIMIZATION = "optimization_only"


@dataclass
class TimingRecord:
    """
    Detailed timing breakdown for energy auditing.
    
    Attributes:
        total_ms: Total solve time in milliseconds
        cpu_ms: Time spent on CPU operations (ms)
        gpu_ms: Time spent on GPU operations (ms)
        breakdown: Optional detailed breakdown by component
    """
    total_ms: float
    cpu_ms: float
    gpu_ms: float = 0.0
    breakdown: Dict[str, float] = field(default_factory=dict)
    
    @property
    def total_seconds(self) -> float:
        return self.total_ms / 1000.0
    
    @property
    def cpu_seconds(self) -> float:
        return self.cpu_ms / 1000.0
    
    @property
    def gpu_seconds(self) -> float:
        return self.gpu_ms / 1000.0


@dataclass
class SolverResult:
    """
    Standard output format for all trajectory solvers.
    
    Attributes:
        trajectory: Array of waypoints, shape (N, 2)
        success: Whether a valid trajectory was found
        timing: Detailed timing information
        solver_type: Type of solver that produced this result
        metadata: Additional solver-specific information
    """
    trajectory: np.ndarray  # Shape: (N, 2)
    success: bool
    timing: TimingRecord
    solver_type: SolverType
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def n_waypoints(self) -> int:
        return len(self.trajectory) if self.trajectory is not None else 0
    
    @property
    def path_length(self) -> float:
        """Compute total path length in meters."""
        if self.trajectory is None or len(self.trajectory) < 2:
            return 0.0
        return float(np.sum(np.linalg.norm(np.diff(self.trajectory, axis=0), axis=1)))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'success': self.success,
            'n_waypoints': self.n_waypoints,
            'path_length': self.path_length,
            'timing_total_ms': self.timing.total_ms,
            'timing_cpu_ms': self.timing.cpu_ms,
            'timing_gpu_ms': self.timing.gpu_ms,
            'solver_type': self.solver_type.value,
            'metadata': self.metadata
        }


# =============================================================================
# TIMING CONTEXT MANAGER
# =============================================================================

class SolverTimer:
    """
    Context manager for precise timing of solver operations.
    
    Supports separate tracking of CPU and GPU time.
    
    Example:
        >>> timer = SolverTimer()
        >>> with timer.cpu_section("rrt_planning"):
        ...     result = rrt_algorithm()
        >>> print(timer.get_timing())
    """
    
    def __init__(self):
        self._start_time: Optional[float] = None
        self._cpu_total: float = 0.0
        self._gpu_total: float = 0.0
        self._breakdown: Dict[str, float] = {}
        self._current_section: Optional[str] = None
        self._section_start: Optional[float] = None
        self._section_type: Optional[str] = None
    
    def start(self) -> 'SolverTimer':
        """Start the overall timer."""
        self._start_time = time.perf_counter()
        return self
    
    def stop(self) -> float:
        """Stop the overall timer and return total time in ms."""
        if self._start_time is None:
            return 0.0
        elapsed = (time.perf_counter() - self._start_time) * 1000
        return elapsed
    
    class _SectionContext:
        """Inner context manager for timed sections."""
        def __init__(self, timer: 'SolverTimer', section_name: str, section_type: str):
            self.timer = timer
            self.section_name = section_name
            self.section_type = section_type
        
        def __enter__(self):
            self.timer._section_start = time.perf_counter()
            self.timer._current_section = self.section_name
            self.timer._section_type = self.section_type
            return self
        
        def __exit__(self, exc_type, exc_val, exc_tb):
            elapsed = (time.perf_counter() - self.timer._section_start) * 1000
            self.timer._breakdown[self.section_name] = elapsed
            
            if self.section_type == 'cpu':
                self.timer._cpu_total += elapsed
            elif self.section_type == 'gpu':
                self.timer._gpu_total += elapsed
            
            self.timer._current_section = None
            self.timer._section_type = None
            return False
    
    def cpu_section(self, name: str) -> _SectionContext:
        """Create a CPU timing section."""
        return self._SectionContext(self, name, 'cpu')
    
    def gpu_section(self, name: str) -> _SectionContext:
        """Create a GPU timing section."""
        return self._SectionContext(self, name, 'gpu')
    
    def get_timing(self) -> TimingRecord:
        """Get the timing record."""
        total = self.stop() if self._start_time else 0.0
        return TimingRecord(
            total_ms=total,
            cpu_ms=self._cpu_total,
            gpu_ms=self._gpu_total,
            breakdown=self._breakdown.copy()
        )


# =============================================================================
# ABSTRACT BASE CLASS
# =============================================================================

class BaseSolver(ABC):
    """
    Abstract Base Class for Trajectory Solvers.
    
    All trajectory solvers must inherit from this class and implement
    the `solve()` method. This ensures consistent interface for 
    benchmarking and energy auditing.
    
    Attributes:
        name: Human-readable solver name
        solver_type: Enum identifying the solver category
        n_waypoints: Number of waypoints in generated trajectories
    
    Methods to Implement:
        solve(env) -> SolverResult: Generate a trajectory for the given environment
    
    Example:
        >>> class MySolver(BaseSolver):
        ...     def solve(self, env):
        ...         # Implementation
        ...         return SolverResult(...)
    """
    
    def __init__(
        self,
        name: str,
        solver_type: SolverType,
        n_waypoints: int = 50
    ):
        """
        Initialize the base solver.
        
        Args:
            name: Human-readable name for logging
            solver_type: Category of solver
            n_waypoints: Target number of waypoints in output
        """
        self.name = name
        self.solver_type = solver_type
        self.n_waypoints = n_waypoints
        self._timer = SolverTimer()
    
    @abstractmethod
    def solve(self, env: 'SpaceEnv') -> SolverResult:
        """
        Generate a trajectory from start to target avoiding obstacles.
        
        This method must be implemented by all concrete solver classes.
        
        Args:
            env: SpaceEnv instance with defined start, target, and obstacles
        
        Returns:
            SolverResult containing the trajectory and timing information
        
        Contract:
            - Must return a SolverResult even on failure (with success=False)
            - timing must accurately reflect CPU vs GPU usage
            - trajectory should have shape (N, 2) where N >= 2
        """
        pass
    
    def validate_trajectory(
        self,
        trajectory: np.ndarray,
        env: 'SpaceEnv',
        safety_margin: float = 0.5
    ) -> Dict[str, Any]:
        """
        Validate a trajectory against environment constraints.
        
        Args:
            trajectory: Array of shape (N, 2)
            env: Environment to validate against
            safety_margin: Buffer around obstacles
        
        Returns:
            Dictionary with validation results
        """
        results = {
            'valid': True,
            'collision_free': True,
            'reaches_target': False,
            'within_bounds': True,
            'issues': []
        }
        
        if trajectory is None or len(trajectory) < 2:
            results['valid'] = False
            results['issues'].append("Trajectory too short or None")
            return results
        
        # Check collision
        collision, idx = env.check_trajectory_collision(trajectory, safety_margin)
        if collision:
            results['collision_free'] = False
            results['valid'] = False
            results['issues'].append(f"Collision at waypoint {idx}")
        
        # Check target reached
        final_dist = np.linalg.norm(trajectory[-1] - env.target_pos)
        results['reaches_target'] = final_dist < 5.0  # 5m tolerance
        if not results['reaches_target']:
            results['issues'].append(f"Final point {final_dist:.1f}m from target")
        
        # Check bounds
        if np.any(trajectory < 0) or \
           np.any(trajectory[:, 0] > env.config.width) or \
           np.any(trajectory[:, 1] > env.config.height):
            results['within_bounds'] = False
            results['valid'] = False
            results['issues'].append("Trajectory exits environment bounds")
        
        return results
    
    def interpolate_trajectory(
        self,
        waypoints: np.ndarray,
        target_points: int = None
    ) -> np.ndarray:
        """
        Interpolate waypoints to achieve target number of points.
        
        Uses linear interpolation between consecutive waypoints.
        
        Args:
            waypoints: Input waypoints, shape (M, 2)
            target_points: Desired number of output points (defaults to self.n_waypoints)
        
        Returns:
            Interpolated trajectory, shape (target_points, 2)
        """
        target_points = target_points or self.n_waypoints
        
        if len(waypoints) == target_points:
            return waypoints
        
        # Compute cumulative distances
        dists = np.zeros(len(waypoints))
        dists[1:] = np.cumsum(np.linalg.norm(np.diff(waypoints, axis=0), axis=1))
        total_dist = dists[-1]
        
        if total_dist < 1e-6:
            return np.tile(waypoints[0], (target_points, 1))
        
        # Generate uniform samples along path
        target_dists = np.linspace(0, total_dist, target_points)
        
        # Interpolate
        result = np.zeros((target_points, 2))
        for i, d in enumerate(target_dists):
            # Find segment
            idx = np.searchsorted(dists, d, side='right') - 1
            idx = np.clip(idx, 0, len(waypoints) - 2)
            
            # Local interpolation factor
            segment_len = dists[idx + 1] - dists[idx]
            if segment_len < 1e-6:
                t = 0.0
            else:
                t = (d - dists[idx]) / segment_len
            
            result[i] = waypoints[idx] + t * (waypoints[idx + 1] - waypoints[idx])
        
        return result
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', type={self.solver_type.value})"


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    'BaseSolver',
    'SolverResult',
    'SolverType',
    'TimingRecord',
    'SolverTimer',
]
