"""
Data Generator Module
=====================
Layer 0: Offline Training Data Pipeline

Generates expert trajectory datasets by running the classical RRT* solver
on randomized environment configurations. This data is used to train
the Diffusion Model for trajectory generation.

Output Format:
    .npz files containing:
    - trajectories: (N, H, 2) array of expert paths
    - obstacle_maps: (N, 64, 64) binary occupancy grids
    - start_positions: (N, 2) start coordinates
    - target_positions: (N, 2) target coordinates
    - metadata: dict with generation parameters
"""

from __future__ import annotations

import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Tuple
from tqdm import tqdm
import json
from datetime import datetime
import warnings

from ..environment.space_env import SpaceEnv, SpaceEnvConfig, create_random_env
from ..solvers.rrt_solver import RRTSolver, RRTConfig


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class DataGeneratorConfig:
    """
    Configuration for expert trajectory data generation.
    
    Attributes:
        n_samples: Total number of trajectories to generate
        n_waypoints: Number of waypoints per trajectory
        map_resolution: Resolution of obstacle maps
        env_config: Environment configuration
        rrt_config: RRT* solver configuration
        difficulty_distribution: Mix of easy/medium/hard environments
        seed: Random seed for reproducibility
        output_dir: Directory to save generated data
    """
    n_samples: int = 10000
    n_waypoints: int = 50
    map_resolution: int = 64
    env_config: Optional[SpaceEnvConfig] = None
    rrt_config: Optional[RRTConfig] = None
    difficulty_distribution: Dict[str, float] = None
    seed: int = 42
    output_dir: str = "./data/expert_trajectories"
    
    def __post_init__(self):
        if self.difficulty_distribution is None:
            self.difficulty_distribution = {
                'easy': 0.2,
                'medium': 0.5,
                'hard': 0.3
            }


# =============================================================================
# DATA GENERATOR
# =============================================================================

class DataGenerator:
    """
    Expert Trajectory Data Generator for Diffusion Model Training.
    
    This class implements Layer 0 of the system architecture:
    automated generation of expert trajectories using RRT*.
    
    Pipeline:
        1. Generate random environment (start, target, obstacles)
        2. Run RRT* solver to find valid trajectory
        3. Store trajectory + environment encoding
        4. Repeat N times
        5. Save to .npz format
    
    Quality Control:
        - Only successful (collision-free) trajectories are saved
        - Environments with no valid solution are skipped
        - Statistics tracked for generation success rate
    
    Example:
        >>> config = DataGeneratorConfig(n_samples=1000, seed=42)
        >>> generator = DataGenerator(config)
        >>> generator.generate()
        >>> generator.save("expert_data.npz")
    """
    
    def __init__(self, config: Optional[DataGeneratorConfig] = None):
        """
        Initialize the Data Generator.
        
        Args:
            config: Generation configuration parameters
        """
        self.config = config or DataGeneratorConfig()
        
        # Initialize RRT* solver
        self.solver = RRTSolver(
            config=self.config.rrt_config or RRTConfig(max_iterations=5000),
            n_waypoints=self.config.n_waypoints
        )
        
        # Random generator
        self._rng = np.random.default_rng(self.config.seed)
        
        # Data storage
        self._trajectories: List[np.ndarray] = []
        self._obstacle_maps: List[np.ndarray] = []
        self._start_positions: List[np.ndarray] = []
        self._target_positions: List[np.ndarray] = []
        self._metadata: List[Dict[str, Any]] = []
        
        # Statistics
        self._stats = {
            'total_attempts': 0,
            'successful': 0,
            'failed': 0,
            'by_difficulty': {'easy': 0, 'medium': 0, 'hard': 0}
        }
    
    def generate(self, show_progress: bool = True) -> None:
        """
        Generate expert trajectory dataset.
        
        Args:
            show_progress: Whether to show progress bar
        """
        n_target = self.config.n_samples
        
        # Pre-compute difficulty samples
        difficulties = self._sample_difficulties(n_target * 2)  # Extra for failures
        diff_idx = 0
        
        iterator = tqdm(total=n_target, desc="Generating trajectories") if show_progress else None
        
        while len(self._trajectories) < n_target and diff_idx < len(difficulties):
            difficulty = difficulties[diff_idx]
            diff_idx += 1
            self._stats['total_attempts'] += 1
            
            try:
                # Create random environment
                env = self._create_environment(difficulty)
                
                # Run RRT* solver
                result = self.solver.solve(env)
                
                # Validate result
                if result.success:
                    collision_free, _ = env.check_trajectory_collision(
                        result.trajectory,
                        safety_margin=0.5
                    )
                    
                    if collision_free is False:  # No collision
                        # Store data
                        self._trajectories.append(result.trajectory)
                        self._obstacle_maps.append(
                            env.get_obstacle_map(self.config.map_resolution)
                        )
                        self._start_positions.append(env.start_pos.copy())
                        self._target_positions.append(env.target_pos.copy())
                        self._metadata.append({
                            'difficulty': difficulty,
                            'n_obstacles': len(env.obstacles),
                            'path_length': result.path_length,
                            'rrt_tree_size': result.metadata.get('tree_size', 0),
                            'solve_time_ms': result.timing.total_ms
                        })
                        
                        self._stats['successful'] += 1
                        self._stats['by_difficulty'][difficulty] += 1
                        
                        if iterator:
                            iterator.update(1)
                    else:
                        self._stats['failed'] += 1
                else:
                    self._stats['failed'] += 1
                    
            except Exception as e:
                warnings.warn(f"Generation failed: {e}")
                self._stats['failed'] += 1
        
        if iterator:
            iterator.close()
        
        print(f"\nGeneration complete:")
        print(f"  Successful: {self._stats['successful']}")
        print(f"  Failed: {self._stats['failed']}")
        print(f"  Success rate: {self._stats['successful'] / max(1, self._stats['total_attempts']) * 100:.1f}%")
    
    def _sample_difficulties(self, n: int) -> List[str]:
        """Sample difficulty levels according to distribution."""
        difficulties = []
        dist = self.config.difficulty_distribution
        
        for difficulty, prob in dist.items():
            count = int(n * prob)
            difficulties.extend([difficulty] * count)
        
        self._rng.shuffle(difficulties)
        return difficulties
    
    def _create_environment(self, difficulty: str) -> SpaceEnv:
        """Create random environment with specified difficulty."""
        difficulty_params = {
            'easy': {'n_obstacles': 5, 'min_r': 2.0, 'max_r': 5.0},
            'medium': {'n_obstacles': 12, 'min_r': 3.0, 'max_r': 7.0},
            'hard': {'n_obstacles': 20, 'min_r': 4.0, 'max_r': 9.0},
        }
        
        params = difficulty_params.get(difficulty, difficulty_params['medium'])
        
        config = SpaceEnvConfig(
            n_obstacles=params['n_obstacles'],
            min_obstacle_radius=params['min_r'],
            max_obstacle_radius=params['max_r'],
            seed=int(self._rng.integers(0, 2**31))
        )
        
        env = SpaceEnv(config)
        env.reset()
        
        return env
    
    def save(self, filename: Optional[str] = None) -> str:
        """
        Save generated data to .npz file.
        
        Args:
            filename: Output filename (uses default if None)
        
        Returns:
            Path to saved file
        """
        if not self._trajectories:
            raise ValueError("No data to save. Run generate() first.")
        
        # Create output directory
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"expert_trajectories_{timestamp}.npz"
        
        filepath = output_dir / filename
        
        # Convert to arrays
        trajectories = np.array(self._trajectories)  # (N, H, 2)
        obstacle_maps = np.array(self._obstacle_maps)  # (N, 64, 64)
        start_positions = np.array(self._start_positions)  # (N, 2)
        target_positions = np.array(self._target_positions)  # (N, 2)
        
        # Save
        np.savez_compressed(
            filepath,
            trajectories=trajectories,
            obstacle_maps=obstacle_maps,
            start_positions=start_positions,
            target_positions=target_positions,
            metadata=json.dumps({
                'n_samples': len(trajectories),
                'n_waypoints': self.config.n_waypoints,
                'map_resolution': self.config.map_resolution,
                'generation_stats': self._stats,
                'timestamp': datetime.now().isoformat()
            })
        )
        
        print(f"Saved {len(trajectories)} trajectories to {filepath}")
        
        return str(filepath)
    
    @staticmethod
    def load(filepath: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict]:
        """
        Load generated data from .npz file.
        
        Args:
            filepath: Path to .npz file
        
        Returns:
            Tuple of (trajectories, obstacle_maps, start_positions, 
                     target_positions, metadata)
        """
        data = np.load(filepath, allow_pickle=True)
        
        return (
            data['trajectories'],
            data['obstacle_maps'],
            data['start_positions'],
            data['target_positions'],
            json.loads(str(data['metadata']))
        )
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get generation statistics."""
        return self._stats.copy()


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    'DataGenerator',
    'DataGeneratorConfig',
]
