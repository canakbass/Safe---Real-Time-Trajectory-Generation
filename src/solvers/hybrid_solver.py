"""
Hybrid Solver Module
====================
Proposed hybrid approach combining Diffusion Model inference with 
SLSQP trajectory optimization.

This is Solver B (Proposed/Novelty) in the benchmark comparison.
Combines probabilistic AI warm-start with deterministic refinement.

Architecture:
    ┌─────────────────────┐     ┌──────────────────────┐
    │  DiffusionModel     │────▶│  SLSQP Optimizer     │
    │  (GPU Inference)    │     │  (CPU Refinement)    │
    │  τ_diff (rough)     │     │  τ_safe (certified)  │
    └─────────────────────┘     └──────────────────────┘
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize, OptimizeResult
from typing import Optional, Dict, Any, Tuple, Callable, TYPE_CHECKING
from dataclasses import dataclass, field
import warnings
from pathlib import Path

from .base_solver import (
    BaseSolver,
    SolverResult,
    SolverType,
    SolverTimer,
    TimingRecord,
)

# PyTorch imports (optional - for trained model)
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class HybridSolverConfig:
    """
    Configuration for Hybrid Diffusion + SLSQP Solver.
    
    Attributes:
        n_diffusion_steps: Number of reverse diffusion steps (T)
        diffusion_noise_scale: Initial noise scale for diffusion
        slsqp_max_iter: Maximum SLSQP iterations
        slsqp_ftol: Function tolerance for SLSQP convergence
        safety_margin: Collision safety buffer (meters)
        dynamics_weight: Weight for dynamics smoothness in objective
    """
    # Diffusion parameters
    n_diffusion_steps: int = 50
    diffusion_noise_scale: float = 1.0
    
    # SLSQP parameters
    slsqp_max_iter: int = 100
    slsqp_ftol: float = 1e-6
    
    # Safety parameters
    safety_margin: float = 1.0
    
    # Objective weights
    dynamics_weight: float = 0.1
    fuel_weight: float = 1.0


@dataclass
class DiffusionModelConfig:
    """
    Configuration for the 1D Temporal Diffusion Model.
    
    Attributes:
        hidden_dim: Hidden layer dimension
        n_layers: Number of transformer/conv layers
        condition_dim: Dimension of obstacle condition vector
        beta_start: Starting noise schedule value
        beta_end: Ending noise schedule value
    """
    hidden_dim: int = 128
    n_layers: int = 4
    condition_dim: int = 128
    beta_start: float = 1e-4
    beta_end: float = 0.02


# =============================================================================
# DIFFUSION MODEL (Production Implementation with PyTorch Support)
# =============================================================================

class TemporalDiffusionModel:
    """
    1D Temporal Diffusion Model for Trajectory Generation.
    
    This implementation supports both:
        1. Trained PyTorch model (when checkpoint available)
        2. Fallback NumPy heuristic (for testing without training)
    
    The model learns to denoise trajectories conditioned on:
        - Start position
        - Target position  
        - Obstacle map encoding
    
    Forward Process (Training):
        q(x_t | x_{t-1}) = N(x_t; √(1-β_t) x_{t-1}, β_t I)
    
    Reverse Process (Inference):
        p_θ(x_{t-1} | x_t, c) = N(x_{t-1}; μ_θ(x_t, t, c), σ_t² I)
    """
    
    def __init__(
        self,
        config: Optional[DiffusionModelConfig] = None,
        n_waypoints: int = 50
    ):
        """
        Initialize the Diffusion Model.
        
        Args:
            config: Model configuration
            n_waypoints: Number of trajectory waypoints to generate
        """
        self.config = config or DiffusionModelConfig()
        self.n_waypoints = n_waypoints
        
        # Noise schedule (linear)
        self.betas = np.linspace(
            self.config.beta_start,
            self.config.beta_end,
            50  # Standard diffusion steps
        )
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = np.cumprod(self.alphas)
        
        # Model state
        self._weights_loaded = False
        self._pytorch_model = None
        self._device = None
        self._rng = np.random.default_rng()
        
        # Statistics for normalization (will be loaded from checkpoint)
        self._traj_mean = None
        self._traj_std = None
    
    def load_weights(self, checkpoint_path: str) -> bool:
        """
        Load trained model weights from PyTorch checkpoint.
        
        Args:
            checkpoint_path: Path to .pth checkpoint file
            
        Returns:
            True if successfully loaded, False otherwise
        """
        if not TORCH_AVAILABLE:
            warnings.warn("PyTorch not available. Using fallback heuristic.")
            self._weights_loaded = False
            return False
        
        checkpoint_file = Path(checkpoint_path)
        if not checkpoint_file.exists():
            warnings.warn(f"Checkpoint not found: {checkpoint_path}. Using fallback.")
            self._weights_loaded = False
            return False
        
        try:
            # Import the trained model class
            from ..training.diffusion_model import TrajectoryDiffusionModel, DiffusionConfig
            
            # Load checkpoint
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            checkpoint = torch.load(checkpoint_path, map_location=self._device)
            
            # Reconstruct config
            saved_config = checkpoint.get('config', {})
            model_config = DiffusionConfig(
                n_waypoints=saved_config.get('n_waypoints', self.n_waypoints),
                hidden_dim=saved_config.get('hidden_dim', 256),
                n_layers=saved_config.get('n_layers', 4),
                n_timesteps=saved_config.get('n_timesteps', 100),
            )
            
            # Create and load model
            self._pytorch_model = TrajectoryDiffusionModel(model_config)
            self._pytorch_model.load_state_dict(checkpoint['model_state_dict'])
            self._pytorch_model.to(self._device)
            self._pytorch_model.eval()
            
            # Update waypoint count
            self.n_waypoints = model_config.n_waypoints
            
            self._weights_loaded = True
            print(f"✓ Loaded trained diffusion model from {checkpoint_path}")
            print(f"  Device: {self._device}")
            
            return True
            
        except Exception as e:
            warnings.warn(f"Failed to load checkpoint: {e}. Using fallback.")
            self._weights_loaded = False
            return False
    
    def encode_obstacles(
        self,
        obstacle_map: np.ndarray,
        start_pos: np.ndarray,
        target_pos: np.ndarray
    ) -> np.ndarray:
        """
        Encode obstacles and endpoints into condition vector.
        
        Args:
            obstacle_map: Binary occupancy grid (H, W)
            start_pos: Start position (2,)
            target_pos: Target position (2,)
        
        Returns:
            Condition vector (obstacle_map for PyTorch, flat vector for fallback)
        """
        if self._weights_loaded and self._pytorch_model is not None:
            # For PyTorch model, return resized obstacle map
            # The CNN encoder will handle it
            from scipy.ndimage import zoom
            
            # Resize to 64x64 for CNN
            target_size = 64
            zoom_factor = target_size / max(obstacle_map.shape)
            resized = zoom(obstacle_map.astype(np.float32), zoom_factor, order=0)
            
            # Pad/crop to exact size
            result = np.zeros((target_size, target_size), dtype=np.float32)
            h, w = min(resized.shape[0], target_size), min(resized.shape[1], target_size)
            result[:h, :w] = resized[:h, :w]
            
            return result
        
        # Fallback: flatten and project
        h, w = obstacle_map.shape
        downsampled = obstacle_map[::4, ::4].flatten()
        
        # Concatenate with normalized positions
        norm_start = start_pos / 100.0
        norm_target = target_pos / 100.0
        
        # Create condition vector
        condition = np.zeros(self.config.condition_dim)
        available_space = self.config.condition_dim - 4
        copy_len = min(len(downsampled), available_space)
        condition[:copy_len] = downsampled[:copy_len]
        condition[-4:-2] = norm_start
        condition[-2:] = norm_target
        
        return condition.astype(np.float32)
    
    def generate_trajectory(
        self,
        condition: np.ndarray,
        start_pos: np.ndarray,
        target_pos: np.ndarray,
        n_steps: int = 50
    ) -> np.ndarray:
        """
        Generate trajectory using reverse diffusion process.
        
        Args:
            condition: Encoded condition (obstacle map or flat vector)
            start_pos: Start position for anchoring
            target_pos: Target position for anchoring
            n_steps: Number of diffusion steps
        
        Returns:
            Generated trajectory of shape (n_waypoints, 2)
        """
        if self._weights_loaded and self._pytorch_model is not None:
            return self._generate_pytorch(condition, start_pos, target_pos)
        else:
            return self._generate_fallback(condition, start_pos, target_pos, n_steps)
    
    def _generate_pytorch(
        self,
        obstacle_map: np.ndarray,
        start_pos: np.ndarray,
        target_pos: np.ndarray
    ) -> np.ndarray:
        """Generate trajectory using trained PyTorch model."""
        with torch.no_grad():
            # Prepare inputs
            obs_tensor = torch.from_numpy(obstacle_map).unsqueeze(0).unsqueeze(0).float()
            start_tensor = torch.from_numpy(start_pos / 100.0).unsqueeze(0).float()
            target_tensor = torch.from_numpy(target_pos / 100.0).unsqueeze(0).float()
            
            # Move to device
            obs_tensor = obs_tensor.to(self._device)
            start_tensor = start_tensor.to(self._device)
            target_tensor = target_tensor.to(self._device)
            
            # Generate
            generated = self._pytorch_model.generate(
                obstacle_maps=obs_tensor,
                start_positions=start_tensor,
                target_positions=target_tensor,
                n_samples=1
            )
            
            # Convert back to numpy and denormalize
            trajectory = generated[0].cpu().numpy() * 100.0  # Denormalize
            
            # Anchor endpoints exactly
            trajectory[0] = start_pos
            trajectory[-1] = target_pos
            
            return trajectory
    
    def _generate_fallback(
        self,
        condition: np.ndarray,
        start_pos: np.ndarray,
        target_pos: np.ndarray,
        n_steps: int
    ) -> np.ndarray:
        """Generate trajectory using heuristic fallback (for untrained model)."""
        # Initialize with noise around linear interpolation
        linear_traj = np.linspace(start_pos, target_pos, self.n_waypoints)
        
        x_t = self._rng.normal(
            size=(self.n_waypoints, 2)
        ).astype(np.float32) * self.config.beta_end
        x_t = x_t + linear_traj
        
        # Reverse diffusion (simplified)
        for t in reversed(range(n_steps)):
            # Predict noise (simplified: random perturbation toward linear)
            noise_pred = self._predict_noise(x_t, t, condition, linear_traj)
            
            # Denoise step
            x_t = x_t - 0.1 * noise_pred
            
            # Add small noise (except final step)
            if t > 0:
                noise = self._rng.normal(size=x_t.shape).astype(np.float32)
                x_t = x_t + 0.01 * noise
        
        # Anchor endpoints
        x_t[0] = start_pos
        x_t[-1] = target_pos
        
        return x_t
    
    def _predict_noise(
        self,
        x_t: np.ndarray,
        t: int,
        condition: np.ndarray,
        prior: np.ndarray
    ) -> np.ndarray:
        """Predict noise for denoising step (fallback heuristic)."""
        deviation = x_t - prior
        return deviation * 0.5 + self._rng.normal(size=x_t.shape) * 0.1
    
    @property
    def is_trained(self) -> bool:
        """Check if using trained model or fallback."""
        return self._weights_loaded and self._pytorch_model is not None


# =============================================================================
# SLSQP TRAJECTORY OPTIMIZER
# =============================================================================

class TrajectoryOptimizer:
    """
    SLSQP-based Trajectory Optimizer for safety refinement.
    
    Takes a rough trajectory (from diffusion model) and refines it to:
        1. Satisfy collision constraints
        2. Respect kinematic limits
        3. Minimize fuel consumption (Delta-V)
    
    Optimization Problem:
        minimize    Σ ||a_i||²  (fuel proxy)
        subject to  d(x_i, obs_j) ≥ safety_margin  ∀ i, j
                    ||v_i|| ≤ v_max  ∀ i
                    x_0 = start, x_N = target
    """
    
    def __init__(
        self,
        config: HybridSolverConfig,
        env: 'SpaceEnv'
    ):
        """
        Initialize the optimizer.
        
        Args:
            config: Solver configuration
            env: Space environment with obstacles
        """
        self.config = config
        self.env = env
        self.n_waypoints = 50  # Will be set from input
    
    def refine(
        self,
        initial_trajectory: np.ndarray,
        start_pos: np.ndarray,
        target_pos: np.ndarray
    ) -> Tuple[np.ndarray, OptimizeResult]:
        """
        Refine trajectory using SLSQP optimization.
        
        This is the critical safety layer that converts the probabilistic
        AI output into a certified collision-free trajectory.
        
        Args:
            initial_trajectory: Warm-start from diffusion model, shape (N, 2)
            start_pos: Fixed start position
            target_pos: Fixed target position
        
        Returns:
            Tuple of (refined_trajectory, scipy_result)
        """
        self.n_waypoints = len(initial_trajectory)
        
        # Flatten for scipy
        x0 = initial_trajectory.flatten()
        
        # Define bounds (environment limits)
        bounds = []
        for i in range(self.n_waypoints):
            bounds.append((0.0, self.env.config.width))   # x
            bounds.append((0.0, self.env.config.height))  # y
        
        # Fix start and target
        bounds[0] = (start_pos[0], start_pos[0])
        bounds[1] = (start_pos[1], start_pos[1])
        bounds[-2] = (target_pos[0], target_pos[0])
        bounds[-1] = (target_pos[1], target_pos[1])
        
        # Build constraints
        constraints = self._build_constraints()
        
        # Run SLSQP
        result = minimize(
            fun=self._objective,
            x0=x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={
                'maxiter': self.config.slsqp_max_iter,
                'ftol': self.config.slsqp_ftol,
                'disp': False
            }
        )
        
        refined_trajectory = result.x.reshape(-1, 2)
        
        return refined_trajectory, result
    
    def _objective(self, x: np.ndarray) -> float:
        """
        Objective function: minimize fuel consumption (acceleration magnitude).
        
        J = Σ ||a_i||² + λ Σ ||jerk_i||²
        """
        trajectory = x.reshape(-1, 2)
        
        # Velocity estimates (finite differences)
        dt = 0.1
        velocities = np.diff(trajectory, axis=0) / dt
        
        # Acceleration estimates
        accelerations = np.diff(velocities, axis=0) / dt
        
        # Fuel cost (L2 norm of accelerations)
        fuel_cost = np.sum(np.linalg.norm(accelerations, axis=1) ** 2)
        
        # Smoothness penalty (jerk)
        if len(accelerations) > 1:
            jerks = np.diff(accelerations, axis=0) / dt
            smoothness_cost = np.sum(np.linalg.norm(jerks, axis=1) ** 2)
        else:
            smoothness_cost = 0.0
        
        return self.config.fuel_weight * fuel_cost + \
               self.config.dynamics_weight * smoothness_cost
    
    def _build_constraints(self) -> list:
        """Build SLSQP constraint dictionaries."""
        constraints = []
        
        # Collision constraints for each waypoint
        for i in range(self.n_waypoints):
            for j, obs in enumerate(self.env.obstacles):
                constraints.append({
                    'type': 'ineq',
                    'fun': self._collision_constraint,
                    'args': (i, obs.center, obs.radius)
                })
        
        return constraints
    
    def _collision_constraint(
        self,
        x: np.ndarray,
        waypoint_idx: int,
        obs_center: np.ndarray,
        obs_radius: float
    ) -> float:
        """
        Collision avoidance constraint.
        
        g(x) = ||p_i - c_j|| - r_j - margin ≥ 0
        
        Returns positive value if constraint satisfied.
        """
        trajectory = x.reshape(-1, 2)
        point = trajectory[waypoint_idx]
        
        distance = np.linalg.norm(point - obs_center)
        return distance - obs_radius - self.config.safety_margin


# =============================================================================
# HYBRID SOLVER IMPLEMENTATION
# =============================================================================

class HybridSolver(BaseSolver):
    """
    Hybrid Diffusion + SLSQP Trajectory Solver.
    
    This is the proposed novel approach combining:
        1. Fast AI inference (Diffusion Model) for initial trajectory guess
        2. Deterministic optimization (SLSQP) for safety certification
    
    Key Innovation:
        The diffusion model provides a good warm-start that allows SLSQP
        to converge quickly to a safe solution. This is faster than
        pure optimization from scratch, and safer than pure AI inference.
    
    Safety Guarantee:
        The final trajectory τ_safe is ALWAYS the output of SLSQP, which
        explicitly enforces collision constraints. The AI output τ_diff
        is NEVER used directly.
    
    Energy Profile:
        - GPU: Diffusion inference (10W)
        - CPU: SLSQP refinement (5W)
    
    Example:
        >>> solver = HybridSolver(checkpoint_path="checkpoints/diffusion_best.pth")
        >>> result = solver.solve(env)
        >>> assert result.metadata['slsqp_success'], "Safety layer must succeed"
    """
    
    # Default checkpoint paths to search
    DEFAULT_CHECKPOINTS = [
        "./checkpoints/diffusion_best.pth",
        "./checkpoints/diffusion_latest.pth",
        "../checkpoints/diffusion_best.pth",
    ]
    
    def __init__(
        self,
        config: Optional[HybridSolverConfig] = None,
        diffusion_config: Optional[DiffusionModelConfig] = None,
        n_waypoints: int = 50,
        checkpoint_path: Optional[str] = None,
        auto_load: bool = True
    ):
        """
        Initialize Hybrid Solver.
        
        Args:
            config: Solver configuration
            diffusion_config: Diffusion model configuration
            n_waypoints: Number of waypoints in trajectory
            checkpoint_path: Path to trained diffusion model weights
            auto_load: If True, automatically search for checkpoint
        """
        super().__init__(
            name="Hybrid Diffusion + SLSQP Solver",
            solver_type=SolverType.HYBRID_DIFFUSION,
            n_waypoints=n_waypoints
        )
        
        self.config = config or HybridSolverConfig()
        
        # Initialize diffusion model
        self.diffusion_model = TemporalDiffusionModel(
            config=diffusion_config,
            n_waypoints=n_waypoints
        )
        
        # Try to load trained weights
        loaded = False
        if checkpoint_path:
            loaded = self.diffusion_model.load_weights(checkpoint_path)
        elif auto_load:
            # Search for checkpoint
            for default_path in self.DEFAULT_CHECKPOINTS:
                if Path(default_path).exists():
                    loaded = self.diffusion_model.load_weights(default_path)
                    if loaded:
                        break
        
        if not loaded:
            warnings.warn(
                "No trained diffusion model loaded. Using fallback heuristic. "
                "Run 'python scripts/train_diffusion.py' to train the model."
            )
        
        # Optimizer initialized per-solve (needs env reference)
        self.optimizer: Optional[TrajectoryOptimizer] = None
    
    @property
    def is_using_trained_model(self) -> bool:
        """Check if solver is using trained AI model."""
        return self.diffusion_model.is_trained
    
    def solve(self, env: 'SpaceEnv') -> SolverResult:
        """
        Generate trajectory using hybrid Diffusion + SLSQP approach.
        
        ╔══════════════════════════════════════════════════════════════════╗
        ║                    HYBRID SOLVE ALGORITHM                        ║
        ╠══════════════════════════════════════════════════════════════════╣
        ║  STEP 1 (GPU): DIFFUSION INFERENCE                               ║
        ║    1.1  obstacle_map ← env.get_obstacle_map(resolution=64)       ║
        ║    1.2  condition ← encode(obstacle_map, start, target)          ║
        ║    1.3  τ_diff ← diffusion_model.generate(condition)             ║
        ║                                                                  ║
        ║  STEP 2 (CPU): SLSQP REFINEMENT                                  ║
        ║    2.1  x0 ← flatten(τ_diff)  // Warm-start from AI              ║
        ║    2.2  constraints ← [collision_avoidance, dynamics_limits]     ║
        ║    2.3  result ← scipy.optimize.minimize(                        ║
        ║             fun=fuel_cost,                                       ║
        ║             x0=x0,                                               ║
        ║             method='SLSQP',                                      ║
        ║             constraints=constraints                              ║
        ║         )                                                        ║
        ║    2.4  τ_safe ← reshape(result.x)                               ║
        ║                                                                  ║
        ║  RETURN: τ_safe (NOT τ_diff!)                                    ║
        ╚══════════════════════════════════════════════════════════════════╝
        
        Args:
            env: SpaceEnv instance with start, target, obstacles
        
        Returns:
            SolverResult containing certified safe trajectory
        """
        timer = SolverTimer()
        timer.start()
        
        # =====================================================================
        # STEP 1: DIFFUSION MODEL INFERENCE (GPU)
        # =====================================================================
        with timer.gpu_section("diffusion_inference"):
            # 1.1 Encode obstacle map
            obstacle_map = env.get_obstacle_map(resolution=64)
            
            # 1.2 Create condition vector
            condition = self.diffusion_model.encode_obstacles(
                obstacle_map=obstacle_map,
                start_pos=env.start_pos,
                target_pos=env.target_pos
            )
            
            # 1.3 Generate rough trajectory via reverse diffusion
            tau_diff = self.diffusion_model.generate_trajectory(
                condition=condition,
                start_pos=env.start_pos,
                target_pos=env.target_pos,
                n_steps=self.config.n_diffusion_steps
            )
        
        # =====================================================================
        # STEP 2: SLSQP REFINEMENT (CPU)
        # =====================================================================
        with timer.cpu_section("slsqp_optimization"):
            # 2.1 Initialize optimizer with environment
            self.optimizer = TrajectoryOptimizer(
                config=self.config,
                env=env
            )
            
            # 2.2-2.4 Refine trajectory using SLSQP
            # τ_diff serves as warm-start (x0)
            tau_safe, opt_result = self.optimizer.refine(
                initial_trajectory=tau_diff,
                start_pos=env.start_pos,
                target_pos=env.target_pos
            )
        
        # =====================================================================
        # RESULT ASSEMBLY
        # =====================================================================
        timing = timer.get_timing()
        
        # Validate final trajectory
        collision_free, collision_idx = env.check_trajectory_collision(
            tau_safe, 
            safety_margin=0.1
        )
        
        success = opt_result.success and not collision_free
        # Note: check_trajectory_collision returns True if collision detected
        success = opt_result.success and (collision_idx is None)
        
        return SolverResult(
            trajectory=tau_safe,  # ALWAYS return SLSQP output, NOT diffusion output
            success=success,
            timing=timing,
            solver_type=self.solver_type,
            metadata={
                'diffusion_time_ms': timing.breakdown.get('diffusion_inference', 0),
                'slsqp_time_ms': timing.breakdown.get('slsqp_optimization', 0),
                'slsqp_success': opt_result.success,
                'slsqp_iterations': opt_result.nit if hasattr(opt_result, 'nit') else None,
                'slsqp_message': opt_result.message if hasattr(opt_result, 'message') else None,
                'collision_free': collision_idx is None,
                'warm_start_source': 'diffusion_model',
                'using_trained_model': self.diffusion_model.is_trained
            }
        )
    
    def solve_without_warmstart(self, env: 'SpaceEnv') -> SolverResult:
        """
        Solve using SLSQP only (no diffusion warm-start).
        
        Useful for ablation studies comparing warm-start benefit.
        
        Args:
            env: SpaceEnv instance
        
        Returns:
            SolverResult from pure SLSQP (linear initialization)
        """
        timer = SolverTimer()
        timer.start()
        
        with timer.cpu_section("slsqp_cold_start"):
            # Cold start: linear interpolation
            tau_linear = np.linspace(
                env.start_pos,
                env.target_pos,
                self.n_waypoints
            )
            
            self.optimizer = TrajectoryOptimizer(
                config=self.config,
                env=env
            )
            
            tau_safe, opt_result = self.optimizer.refine(
                initial_trajectory=tau_linear,
                start_pos=env.start_pos,
                target_pos=env.target_pos
            )
        
        timing = timer.get_timing()
        collision_free, collision_idx = env.check_trajectory_collision(tau_safe, 0.1)
        
        return SolverResult(
            trajectory=tau_safe,
            success=opt_result.success and (collision_idx is None),
            timing=timing,
            solver_type=SolverType.PURE_OPTIMIZATION,
            metadata={
                'slsqp_success': opt_result.success,
                'warm_start_source': 'linear_interpolation'
            }
        )


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    'HybridSolver',
    'HybridSolverConfig',
    'TemporalDiffusionModel',
    'DiffusionModelConfig',
    'TrajectoryOptimizer',
]
