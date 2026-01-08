"""
Space Environment Module
========================
OpenAI Gymnasium-based 2D space environment for trajectory planning benchmarks.

This module implements Layer 1 (The Simulation/World) of the system architecture.
"""

from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any
import warnings


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class CircularObstacle:
    """
    Represents a circular obstacle (space debris) in the environment.
    
    Attributes:
        center: (x, y) position of obstacle center in meters
        radius: Radius of the obstacle in meters
        id: Unique identifier for tracking
    """
    center: np.ndarray  # Shape: (2,)
    radius: float
    id: Optional[str] = None
    
    def __post_init__(self):
        self.center = np.asarray(self.center, dtype=np.float32)
        if self.id is None:
            self.id = f"obs_{id(self)}"
    
    def contains_point(self, point: np.ndarray, safety_margin: float = 0.0) -> bool:
        """Check if a point lies within the obstacle (with optional safety margin)."""
        distance = np.linalg.norm(point - self.center)
        return distance < (self.radius + safety_margin)
    
    def distance_to_point(self, point: np.ndarray) -> float:
        """Compute signed distance from point to obstacle surface (negative = inside)."""
        return np.linalg.norm(point - self.center) - self.radius


@dataclass
class KinematicConstraints:
    """
    Kinematic constraints for point-mass spacecraft.
    
    Attributes:
        v_max: Maximum velocity magnitude (m/s)
        a_max: Maximum acceleration magnitude (m/s²)
        delta_v_budget: Total available Delta-V (m/s)
    """
    v_max: float = 10.0          # m/s
    a_max: float = 2.0           # m/s²
    delta_v_budget: float = 50.0  # m/s


@dataclass
class SpaceEnvConfig:
    """
    Configuration parameters for SpaceEnv.
    
    Attributes:
        width: Environment width in meters
        height: Environment height in meters
        n_obstacles: Number of obstacles to generate
        min_obstacle_radius: Minimum obstacle radius (m)
        max_obstacle_radius: Maximum obstacle radius (m)
        min_clearance: Minimum clearance between start/goal and obstacles (m)
        seed: Random seed for reproducibility
    """
    width: float = 100.0
    height: float = 100.0
    n_obstacles: int = 10
    min_obstacle_radius: float = 2.0
    max_obstacle_radius: float = 8.0
    min_clearance: float = 5.0
    seed: Optional[int] = None
    kinematics: KinematicConstraints = field(default_factory=KinematicConstraints)


# =============================================================================
# GYMNASIUM ENVIRONMENT
# =============================================================================

class SpaceEnv(gym.Env):
    """
    2D Space Environment for Trajectory Planning Benchmarks.
    
    This Gymnasium environment simulates a 2D space domain with:
    - Kinematic point-mass spacecraft dynamics
    - Static circular obstacles (space debris)
    - Configurable start and target positions
    
    Observation Space:
        Box(low=0, high=max(width,height), shape=(4,), dtype=float32)
        [spacecraft_x, spacecraft_y, target_x, target_y]
    
    Action Space:
        Box(low=-a_max, high=a_max, shape=(2,), dtype=float32)
        [acceleration_x, acceleration_y]
    
    Reward:
        - Distance reduction to target: +1.0 per meter
        - Collision: -100.0
        - Target reached: +500.0
    
    Example:
        >>> config = SpaceEnvConfig(n_obstacles=15, seed=42)
        >>> env = SpaceEnv(config)
        >>> obs, info = env.reset()
        >>> print(f"Start: {info['start_pos']}, Target: {info['target_pos']}")
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}
    
    def __init__(
        self, 
        config: Optional[SpaceEnvConfig] = None,
        render_mode: Optional[str] = None
    ):
        """
        Initialize the Space Environment.
        
        Args:
            config: Environment configuration. Uses defaults if None.
            render_mode: Rendering mode ('human', 'rgb_array', or None)
        """
        super().__init__()
        
        self.config = config or SpaceEnvConfig()
        self.render_mode = render_mode
        
        # Initialize random generator
        self._rng = np.random.default_rng(self.config.seed)
        
        # Environment state
        self.start_pos: np.ndarray = None
        self.target_pos: np.ndarray = None
        self.current_pos: np.ndarray = None
        self.current_vel: np.ndarray = None
        self.obstacles: List[CircularObstacle] = []
        self.timestep: int = 0
        self.max_timesteps: int = 500
        
        # Define spaces
        max_dim = max(self.config.width, self.config.height)
        self.observation_space = spaces.Box(
            low=0.0,
            high=max_dim,
            shape=(4,),
            dtype=np.float32
        )
        
        self.action_space = spaces.Box(
            low=-self.config.kinematics.a_max,
            high=self.config.kinematics.a_max,
            shape=(2,),
            dtype=np.float32
        )
        
        # Rendering
        self._fig = None
        self._ax = None

    def reset(
        self, 
        seed: Optional[int] = None, 
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset the environment to initial state.
        
        Args:
            seed: Random seed for this episode
            options: Additional options dict with optional keys:
                - 'start_pos': Override start position
                - 'target_pos': Override target position
                - 'obstacles': Override obstacle list
        
        Returns:
            observation: Current observation
            info: Dictionary with additional information
        """
        super().reset(seed=seed)
        
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        
        options = options or {}
        
        # Generate or set obstacles
        if 'obstacles' in options:
            self.obstacles = options['obstacles']
        else:
            self._generate_obstacles()
        
        # Generate or set start position
        if 'start_pos' in options:
            self.start_pos = np.asarray(options['start_pos'], dtype=np.float32)
        else:
            self.start_pos = self._sample_valid_position()
        
        # Generate or set target position
        if 'target_pos' in options:
            self.target_pos = np.asarray(options['target_pos'], dtype=np.float32)
        else:
            self.target_pos = self._sample_valid_position(
                min_distance_from=self.start_pos,
                min_distance=30.0  # Ensure non-trivial problem
            )
        
        # Initialize spacecraft state
        self.current_pos = self.start_pos.copy()
        self.current_vel = np.zeros(2, dtype=np.float32)
        self.timestep = 0
        
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one timestep of dynamics.
        
        Args:
            action: Acceleration vector [a_x, a_y]
        
        Returns:
            observation: New observation
            reward: Reward for this step
            terminated: True if episode ended (collision or goal reached)
            truncated: True if max timesteps exceeded
            info: Additional information
        """
        action = np.clip(action, -self.config.kinematics.a_max, 
                        self.config.kinematics.a_max)
        
        dt = 0.1  # Time step in seconds
        
        # Store previous distance for reward calculation
        prev_distance = np.linalg.norm(self.current_pos - self.target_pos)
        
        # Kinematic update (Euler integration)
        self.current_vel += action * dt
        
        # Velocity clipping
        speed = np.linalg.norm(self.current_vel)
        if speed > self.config.kinematics.v_max:
            self.current_vel = (self.current_vel / speed) * self.config.kinematics.v_max
        
        self.current_pos += self.current_vel * dt
        
        # Boundary clipping
        self.current_pos = np.clip(
            self.current_pos,
            [0, 0],
            [self.config.width, self.config.height]
        )
        
        self.timestep += 1
        
        # Check termination conditions
        collision = self._check_collision(self.current_pos)
        current_distance = np.linalg.norm(self.current_pos - self.target_pos)
        goal_reached = current_distance < 2.0
        
        terminated = collision or goal_reached
        truncated = self.timestep >= self.max_timesteps
        
        # Calculate reward
        reward = self._calculate_reward(
            prev_distance, current_distance, collision, goal_reached, action
        )
        
        observation = self._get_observation()
        info = self._get_info()
        info['collision'] = collision
        info['goal_reached'] = goal_reached
        
        return observation, reward, terminated, truncated, info

    # =========================================================================
    # HELPER METHODS
    # =========================================================================
    
    def _generate_obstacles(self) -> None:
        """Generate random circular obstacles."""
        self.obstacles = []
        
        for i in range(self.config.n_obstacles):
            max_attempts = 100
            for _ in range(max_attempts):
                center = self._rng.uniform(
                    low=[self.config.max_obstacle_radius, 
                         self.config.max_obstacle_radius],
                    high=[self.config.width - self.config.max_obstacle_radius,
                          self.config.height - self.config.max_obstacle_radius]
                )
                radius = self._rng.uniform(
                    self.config.min_obstacle_radius,
                    self.config.max_obstacle_radius
                )
                
                # Check overlap with existing obstacles
                valid = True
                for obs in self.obstacles:
                    dist = np.linalg.norm(center - obs.center)
                    if dist < (radius + obs.radius + 2.0):  # 2m buffer
                        valid = False
                        break
                
                if valid:
                    self.obstacles.append(CircularObstacle(
                        center=center,
                        radius=radius,
                        id=f"debris_{i}"
                    ))
                    break
            else:
                warnings.warn(f"Could not place obstacle {i} after {max_attempts} attempts")

    def _sample_valid_position(
        self, 
        min_distance_from: Optional[np.ndarray] = None,
        min_distance: float = 0.0
    ) -> np.ndarray:
        """Sample a position that is not inside any obstacle."""
        max_attempts = 1000
        
        for _ in range(max_attempts):
            pos = self._rng.uniform(
                low=[self.config.min_clearance, self.config.min_clearance],
                high=[self.config.width - self.config.min_clearance,
                      self.config.height - self.config.min_clearance]
            ).astype(np.float32)
            
            # Check obstacle clearance
            valid = True
            for obs in self.obstacles:
                if obs.distance_to_point(pos) < self.config.min_clearance:
                    valid = False
                    break
            
            # Check distance constraint
            if valid and min_distance_from is not None:
                if np.linalg.norm(pos - min_distance_from) < min_distance:
                    valid = False
            
            if valid:
                return pos
        
        raise RuntimeError(f"Could not find valid position after {max_attempts} attempts")

    def _check_collision(self, position: np.ndarray) -> bool:
        """Check if position collides with any obstacle."""
        for obs in self.obstacles:
            if obs.contains_point(position):
                return True
        return False

    def _get_observation(self) -> np.ndarray:
        """Construct observation vector."""
        return np.concatenate([self.current_pos, self.target_pos]).astype(np.float32)

    def _get_info(self) -> Dict[str, Any]:
        """Construct info dictionary."""
        return {
            'start_pos': self.start_pos.copy(),
            'target_pos': self.target_pos.copy(),
            'current_pos': self.current_pos.copy(),
            'current_vel': self.current_vel.copy(),
            'n_obstacles': len(self.obstacles),
            'timestep': self.timestep
        }

    def _calculate_reward(
        self,
        prev_distance: float,
        current_distance: float,
        collision: bool,
        goal_reached: bool,
        action: np.ndarray
    ) -> float:
        """Calculate step reward."""
        reward = 0.0
        
        # Progress reward
        reward += (prev_distance - current_distance) * 1.0
        
        # Fuel penalty (proportional to acceleration magnitude)
        reward -= np.linalg.norm(action) * 0.01
        
        # Terminal rewards
        if collision:
            reward -= 100.0
        elif goal_reached:
            reward += 500.0
        
        return reward

    # =========================================================================
    # UTILITY METHODS FOR SOLVERS
    # =========================================================================

    def get_obstacle_map(self, resolution: int = 64) -> np.ndarray:
        """
        Generate binary occupancy grid for obstacle encoding.
        
        Args:
            resolution: Grid resolution (pixels per side)
        
        Returns:
            Binary array of shape (resolution, resolution)
            1 = occupied, 0 = free
        """
        grid = np.zeros((resolution, resolution), dtype=np.float32)
        
        cell_width = self.config.width / resolution
        cell_height = self.config.height / resolution
        
        for i in range(resolution):
            for j in range(resolution):
                # Cell center
                x = (j + 0.5) * cell_width
                y = (i + 0.5) * cell_height
                point = np.array([x, y])
                
                for obs in self.obstacles:
                    if obs.contains_point(point, safety_margin=cell_width/2):
                        grid[i, j] = 1.0
                        break
        
        return grid

    def get_problem_vector(self) -> np.ndarray:
        """
        Get flattened problem representation for model input.
        
        Returns:
            Array containing [start_x, start_y, target_x, target_y]
        """
        return np.concatenate([self.start_pos, self.target_pos])

    def check_trajectory_collision(
        self, 
        trajectory: np.ndarray,
        safety_margin: float = 0.5
    ) -> Tuple[bool, Optional[int]]:
        """
        Check if a trajectory collides with any obstacle.
        
        Args:
            trajectory: Array of shape (N, 2) containing waypoints
            safety_margin: Additional safety buffer around obstacles
        
        Returns:
            Tuple of (collision_occurred, first_collision_index)
        """
        for i, point in enumerate(trajectory):
            for obs in self.obstacles:
                if obs.contains_point(point, safety_margin=safety_margin):
                    return True, i
        return False, None

    def compute_trajectory_delta_v(self, trajectory: np.ndarray, dt: float = 0.1) -> float:
        """
        Compute total Delta-V (fuel cost) for a trajectory.
        
        Uses finite differences to estimate velocity changes.
        
        Args:
            trajectory: Array of shape (N, 2) containing waypoints
            dt: Time step between waypoints
        
        Returns:
            Total Delta-V in m/s
        """
        if len(trajectory) < 3:
            return 0.0
        
        # Velocity estimates
        velocities = np.diff(trajectory, axis=0) / dt
        
        # Acceleration estimates
        accelerations = np.diff(velocities, axis=0) / dt
        
        # Total Delta-V = integral of |a| * dt
        delta_v = np.sum(np.linalg.norm(accelerations, axis=1)) * dt
        
        return float(delta_v)

    def clone(self) -> 'SpaceEnv':
        """Create a deep copy of the environment with identical state."""
        new_env = SpaceEnv(config=self.config, render_mode=self.render_mode)
        new_env.start_pos = self.start_pos.copy()
        new_env.target_pos = self.target_pos.copy()
        new_env.current_pos = self.current_pos.copy()
        new_env.current_vel = self.current_vel.copy()
        new_env.obstacles = [
            CircularObstacle(obs.center.copy(), obs.radius, obs.id)
            for obs in self.obstacles
        ]
        new_env.timestep = self.timestep
        return new_env

    # =========================================================================
    # RENDERING
    # =========================================================================

    def render(self):
        """Render the environment."""
        if self.render_mode is None:
            return None
        
        try:
            import matplotlib.pyplot as plt
            from matplotlib.patches import Circle
        except ImportError:
            warnings.warn("matplotlib required for rendering")
            return None
        
        if self._fig is None:
            self._fig, self._ax = plt.subplots(1, 1, figsize=(8, 8))
        
        self._ax.clear()
        self._ax.set_xlim(0, self.config.width)
        self._ax.set_ylim(0, self.config.height)
        self._ax.set_aspect('equal')
        self._ax.set_xlabel('X (m)')
        self._ax.set_ylabel('Y (m)')
        self._ax.set_title('Space Trajectory Environment')
        
        # Draw obstacles
        for obs in self.obstacles:
            circle = Circle(obs.center, obs.radius, color='gray', alpha=0.7)
            self._ax.add_patch(circle)
        
        # Draw start, target, current positions
        self._ax.plot(*self.start_pos, 'go', markersize=12, label='Start')
        self._ax.plot(*self.target_pos, 'r*', markersize=15, label='Target')
        self._ax.plot(*self.current_pos, 'b^', markersize=10, label='Spacecraft')
        
        self._ax.legend(loc='upper right')
        
        if self.render_mode == 'human':
            plt.pause(0.01)
            return None
        elif self.render_mode == 'rgb_array':
            self._fig.canvas.draw()
            img = np.frombuffer(self._fig.canvas.tostring_rgb(), dtype=np.uint8)
            img = img.reshape(self._fig.canvas.get_width_height()[::-1] + (3,))
            return img

    def close(self):
        """Clean up resources."""
        if self._fig is not None:
            import matplotlib.pyplot as plt
            plt.close(self._fig)
            self._fig = None
            self._ax = None


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

def create_random_env(
    n_obstacles: int = 10,
    seed: Optional[int] = None,
    difficulty: str = 'medium'
) -> SpaceEnv:
    """
    Factory function to create environment with preset difficulty.
    
    Args:
        n_obstacles: Number of obstacles
        seed: Random seed
        difficulty: 'easy', 'medium', or 'hard'
    
    Returns:
        Configured SpaceEnv instance
    """
    difficulty_presets = {
        'easy': {'n_obstacles': 5, 'min_obstacle_radius': 2.0, 'max_obstacle_radius': 5.0},
        'medium': {'n_obstacles': 10, 'min_obstacle_radius': 3.0, 'max_obstacle_radius': 8.0},
        'hard': {'n_obstacles': 20, 'min_obstacle_radius': 4.0, 'max_obstacle_radius': 10.0},
    }
    
    preset = difficulty_presets.get(difficulty, difficulty_presets['medium'])
    preset['n_obstacles'] = n_obstacles  # Override with explicit value
    preset['seed'] = seed
    
    config = SpaceEnvConfig(**preset)
    return SpaceEnv(config)


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    'SpaceEnv',
    'SpaceEnvConfig',
    'CircularObstacle',
    'KinematicConstraints',
    'create_random_env',
]
