"""
Diffusion Model Architecture
============================
1D Temporal Diffusion Model for Trajectory Generation.

This module implements a conditional diffusion model that learns to
generate spacecraft trajectories given:
- Start position
- Target position  
- Obstacle map encoding

Architecture:
    Conditional U-Net style MLP with sinusoidal time embeddings.

Reference:
    Ho et al., "Denoising Diffusion Probabilistic Models", NeurIPS 2020
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class DiffusionConfig:
    """Configuration for the Diffusion Model."""
    # Trajectory parameters
    n_waypoints: int = 50
    waypoint_dim: int = 2  # (x, y)
    
    # Condition parameters
    obstacle_map_size: int = 64
    condition_dim: int = 256
    
    # Model architecture
    hidden_dim: int = 512
    n_layers: int = 6
    dropout: float = 0.1
    
    # Diffusion parameters
    n_timesteps: int = 100
    beta_start: float = 1e-4
    beta_end: float = 0.02
    
    # Training
    learning_rate: float = 1e-4
    batch_size: int = 64
    n_epochs: int = 200


# =============================================================================
# BUILDING BLOCKS
# =============================================================================

class SinusoidalPositionEmbedding(nn.Module):
    """
    Sinusoidal embeddings for diffusion timesteps.
    
    Same as Transformer positional encoding but for time.
    """
    
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
    
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        device = t.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = t[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class ObstacleEncoder(nn.Module):
    """
    CNN encoder for obstacle maps.
    
    Takes a 64x64 binary obstacle map and produces a latent vector.
    """
    
    def __init__(self, output_dim: int = 256):
        super().__init__()
        
        self.conv_layers = nn.Sequential(
            # 64x64 -> 32x32
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            
            # 32x32 -> 16x16
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            
            # 16x16 -> 8x8
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            
            # 8x8 -> 4x4
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
        )
        
        # 256 * 4 * 4 = 4096 -> output_dim
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, output_dim),
            nn.ReLU(),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Obstacle map of shape (B, 64, 64) or (B, 1, 64, 64)
        
        Returns:
            Latent vector of shape (B, output_dim)
        """
        if x.dim() == 3:
            x = x.unsqueeze(1)  # Add channel dim
        
        x = self.conv_layers(x)
        x = self.fc(x)
        return x


class ResidualBlock(nn.Module):
    """Residual MLP block with time conditioning."""
    
    def __init__(self, dim: int, time_dim: int, dropout: float = 0.1):
        super().__init__()
        
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
        )
        
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, dim),
        )
        
        self.norm = nn.LayerNorm(dim)
    
    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        h = self.mlp(h) + self.time_mlp(t_emb)
        return x + h


# =============================================================================
# MAIN DIFFUSION MODEL
# =============================================================================

class TrajectoryDiffusionModel(nn.Module):
    """
    Conditional Diffusion Model for Trajectory Generation.
    
    Architecture:
        1. Encode obstacle map with CNN
        2. Concatenate with start/target positions
        3. Process noisy trajectory through residual MLP blocks
        4. Predict noise to denoise
    
    Input:
        - x_t: Noisy trajectory (B, n_waypoints, 2)
        - t: Diffusion timestep (B,)
        - obstacle_map: Binary grid (B, 64, 64)
        - start_pos: (B, 2)
        - target_pos: (B, 2)
    
    Output:
        - eps_pred: Predicted noise (B, n_waypoints, 2)
    """
    
    def __init__(self, config: Optional[DiffusionConfig] = None):
        super().__init__()
        
        self.config = config or DiffusionConfig()
        
        # Trajectory dimension
        self.traj_dim = self.config.n_waypoints * self.config.waypoint_dim
        
        # Time embedding
        self.time_embed = nn.Sequential(
            SinusoidalPositionEmbedding(self.config.hidden_dim),
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim),
        )
        
        # Obstacle encoder
        self.obstacle_encoder = ObstacleEncoder(self.config.condition_dim)
        
        # Start/target encoder
        self.endpoint_encoder = nn.Sequential(
            nn.Linear(4, 64),  # start(2) + target(2)
            nn.SiLU(),
            nn.Linear(64, 64),
        )
        
        # Input projection: trajectory + condition
        total_condition_dim = self.config.condition_dim + 64  # obstacle + endpoints
        self.input_proj = nn.Linear(self.traj_dim + total_condition_dim, self.config.hidden_dim)
        
        # Residual blocks
        self.blocks = nn.ModuleList([
            ResidualBlock(self.config.hidden_dim, self.config.hidden_dim, self.config.dropout)
            for _ in range(self.config.n_layers)
        ])
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.LayerNorm(self.config.hidden_dim),
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.config.hidden_dim, self.traj_dim),
        )
        
        # Initialize diffusion schedule
        self._init_diffusion_schedule()
    
    def _init_diffusion_schedule(self):
        """Initialize beta schedule and derived quantities."""
        # Linear beta schedule
        betas = torch.linspace(
            self.config.beta_start,
            self.config.beta_end,
            self.config.n_timesteps
        )
        
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        
        # Register as buffers (not parameters)
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas', torch.sqrt(1.0 / alphas))
        
        # For posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.register_buffer('posterior_variance', posterior_variance)
    
    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        obstacle_map: torch.Tensor,
        start_pos: torch.Tensor,
        target_pos: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict noise from noisy trajectory.
        
        Args:
            x_t: Noisy trajectory (B, n_waypoints, 2)
            t: Timesteps (B,)
            obstacle_map: Binary grid (B, 64, 64)
            start_pos: Start position (B, 2)
            target_pos: Target position (B, 2)
        
        Returns:
            Predicted noise (B, n_waypoints, 2)
        """
        B = x_t.shape[0]
        
        # Flatten trajectory
        x_flat = x_t.view(B, -1)  # (B, n_waypoints * 2)
        
        # Time embedding
        t_emb = self.time_embed(t)  # (B, hidden_dim)
        
        # Condition encoding
        obs_emb = self.obstacle_encoder(obstacle_map)  # (B, condition_dim)
        endpoints = torch.cat([start_pos, target_pos], dim=-1)  # (B, 4)
        endpoint_emb = self.endpoint_encoder(endpoints)  # (B, 64)
        condition = torch.cat([obs_emb, endpoint_emb], dim=-1)  # (B, condition_dim + 64)
        
        # Concatenate trajectory with condition
        h = torch.cat([x_flat, condition], dim=-1)  # (B, traj_dim + cond_dim)
        h = self.input_proj(h)  # (B, hidden_dim)
        
        # Process through residual blocks
        for block in self.blocks:
            h = block(h, t_emb)
        
        # Output projection
        eps_pred = self.output_proj(h)  # (B, traj_dim)
        eps_pred = eps_pred.view(B, self.config.n_waypoints, 2)
        
        return eps_pred
    
    # =========================================================================
    # DIFFUSION PROCESS METHODS
    # =========================================================================
    
    def q_sample(
        self,
        x_0: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward diffusion: add noise to trajectory.
        
        q(x_t | x_0) = N(x_t; sqrt(α̅_t) * x_0, (1 - α̅_t) * I)
        
        Args:
            x_0: Clean trajectory (B, n_waypoints, 2)
            t: Timesteps (B,)
            noise: Optional pre-generated noise
        
        Returns:
            x_t: Noisy trajectory
            noise: The noise that was added
        """
        if noise is None:
            noise = torch.randn_like(x_0)
        
        sqrt_alpha_cumprod_t = self.sqrt_alphas_cumprod[t][:, None, None]
        sqrt_one_minus_alpha_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t][:, None, None]
        
        x_t = sqrt_alpha_cumprod_t * x_0 + sqrt_one_minus_alpha_cumprod_t * noise
        
        return x_t, noise
    
    @torch.no_grad()
    def p_sample(
        self,
        x_t: torch.Tensor,
        t: int,
        obstacle_map: torch.Tensor,
        start_pos: torch.Tensor,
        target_pos: torch.Tensor,
    ) -> torch.Tensor:
        """
        Reverse diffusion: denoise one step.
        
        p(x_{t-1} | x_t) using predicted noise.
        """
        B = x_t.shape[0]
        t_tensor = torch.full((B,), t, device=x_t.device, dtype=torch.long)
        
        # Predict noise
        eps_pred = self.forward(x_t, t_tensor, obstacle_map, start_pos, target_pos)
        
        # Compute x_{t-1}
        beta_t = self.betas[t]
        sqrt_recip_alpha_t = self.sqrt_recip_alphas[t]
        sqrt_one_minus_alpha_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t]
        
        # Mean of p(x_{t-1} | x_t)
        mean = sqrt_recip_alpha_t * (x_t - beta_t / sqrt_one_minus_alpha_cumprod_t * eps_pred)
        
        if t > 0:
            noise = torch.randn_like(x_t)
            variance = torch.sqrt(self.posterior_variance[t])
            x_t_minus_1 = mean + variance * noise
        else:
            x_t_minus_1 = mean
        
        return x_t_minus_1
    
    @torch.no_grad()
    def generate(
        self,
        obstacle_map: torch.Tensor,
        start_pos: torch.Tensor,
        target_pos: torch.Tensor,
        n_samples: int = 1,
    ) -> torch.Tensor:
        """
        Generate trajectories via full reverse diffusion.
        
        Args:
            obstacle_map: (64, 64) or (B, 64, 64)
            start_pos: (2,) or (B, 2)
            target_pos: (2,) or (B, 2)
            n_samples: Number of trajectories to generate
        
        Returns:
            Generated trajectories (n_samples, n_waypoints, 2)
        """
        device = next(self.parameters()).device
        
        # Handle batch dimensions
        if obstacle_map.dim() == 2:
            obstacle_map = obstacle_map.unsqueeze(0).expand(n_samples, -1, -1)
        if start_pos.dim() == 1:
            start_pos = start_pos.unsqueeze(0).expand(n_samples, -1)
        if target_pos.dim() == 1:
            target_pos = target_pos.unsqueeze(0).expand(n_samples, -1)
        
        obstacle_map = obstacle_map.to(device)
        start_pos = start_pos.to(device)
        target_pos = target_pos.to(device)
        
        # Start from pure noise
        x_t = torch.randn(
            n_samples, self.config.n_waypoints, 2,
            device=device
        )
        
        # Reverse diffusion
        for t in reversed(range(self.config.n_timesteps)):
            x_t = self.p_sample(x_t, t, obstacle_map, start_pos, target_pos)
        
        # Anchor endpoints
        x_t[:, 0, :] = start_pos
        x_t[:, -1, :] = target_pos
        
        return x_t


# =============================================================================
# TRAINING UTILITIES
# =============================================================================

class TrajectoryDataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset for expert trajectories.
    
    Loads data from .npz file generated by DataGenerator.
    """
    
    def __init__(self, npz_path: str, normalize: bool = True):
        """
        Args:
            npz_path: Path to .npz file with expert trajectories
            normalize: Whether to normalize trajectories to [0, 1]
        """
        data = np.load(npz_path, allow_pickle=True)
        
        self.trajectories = data['trajectories'].astype(np.float32)
        self.obstacle_maps = data['obstacle_maps'].astype(np.float32)
        self.start_positions = data['start_positions'].astype(np.float32)
        self.target_positions = data['target_positions'].astype(np.float32)
        
        self.normalize = normalize
        if normalize:
            # Normalize to [0, 1] assuming 100m environment
            self.trajectories = self.trajectories / 100.0
            self.start_positions = self.start_positions / 100.0
            self.target_positions = self.target_positions / 100.0
    
    def __len__(self) -> int:
        return len(self.trajectories)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            'trajectory': torch.from_numpy(self.trajectories[idx]),
            'obstacle_map': torch.from_numpy(self.obstacle_maps[idx]),
            'start_pos': torch.from_numpy(self.start_positions[idx]),
            'target_pos': torch.from_numpy(self.target_positions[idx]),
        }


def compute_loss(
    model: TrajectoryDiffusionModel,
    batch: Dict[str, torch.Tensor],
    device: torch.device,
) -> torch.Tensor:
    """
    Compute diffusion training loss.
    
    L = E[||ε - ε_θ(x_t, t)||²]
    """
    trajectory = batch['trajectory'].to(device)
    obstacle_map = batch['obstacle_map'].to(device)
    start_pos = batch['start_pos'].to(device)
    target_pos = batch['target_pos'].to(device)
    
    B = trajectory.shape[0]
    
    # Sample random timesteps
    t = torch.randint(0, model.config.n_timesteps, (B,), device=device)
    
    # Add noise
    noise = torch.randn_like(trajectory)
    x_t, _ = model.q_sample(trajectory, t, noise)
    
    # Predict noise
    eps_pred = model(x_t, t, obstacle_map, start_pos, target_pos)
    
    # MSE loss
    loss = F.mse_loss(eps_pred, noise)
    
    return loss


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    'TrajectoryDiffusionModel',
    'DiffusionConfig',
    'TrajectoryDataset',
    'ObstacleEncoder',
    'compute_loss',
]
