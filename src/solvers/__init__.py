"""
Solvers Module
==============
Contains all trajectory solver implementations for benchmark comparison.

- BaseSolver: Abstract interface for all solvers
- RRTSolver: Classical RRT* baseline (Solver A)
- HybridSolver: Diffusion + SLSQP hybrid (Solver B - Proposed)
"""

from .base_solver import (
    BaseSolver,
    SolverResult,
    SolverType,
    TimingRecord,
    SolverTimer,
)

from .rrt_solver import (
    RRTSolver,
    RRTConfig,
    RRTNode,
)

from .hybrid_solver import (
    HybridSolver,
    HybridSolverConfig,
    TemporalDiffusionModel,
    DiffusionModelConfig,
    TrajectoryOptimizer,
)

__all__ = [
    # Base
    'BaseSolver',
    'SolverResult',
    'SolverType',
    'TimingRecord',
    'SolverTimer',
    # RRT*
    'RRTSolver',
    'RRTConfig',
    'RRTNode',
    # Hybrid
    'HybridSolver',
    'HybridSolverConfig',
    'TemporalDiffusionModel',
    'DiffusionModelConfig',
    'TrajectoryOptimizer',
]
