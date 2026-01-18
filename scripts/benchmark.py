"""
Hybrid AI Trajectory Planning Benchmark
=========================================
Compares A* (baseline) vs ML Model vs Hybrid approach.

Features:
1. Realistic environment (1km x 1km, 6 obstacles)
2. A* as deterministic baseline
3. 4 ML architectures: SimpleMLP, DeepCNN, UNet, Attention
4. Hybrid approach: ML prediction + collision repair
5. GPU vs CPU inference comparison

Usage:
    python scripts/benchmark.py

Results:
    - SimpleMLP: 96% success, 3.5x faster than A*
    - All models achieve 94-96% success rate
"""

import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import sys
import time
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.environment.space_env import SpaceEnvironment, EnvironmentConfig


# =============================================================================
# MODEL 1: SIMPLE MLP (Baseline)
# =============================================================================

class SimpleTrajectoryMLP(nn.Module):
    """
    Simple MLP for trajectory prediction.
    
    Input: obstacle_map (64x64) + start (2) + target (2)
    Output: trajectory (50, 2) normalized to [0, 1]
    """
    
    def __init__(self, n_waypoints: int = 50):
        super().__init__()
        self.n_waypoints = n_waypoints
        
        # Obstacle encoder (simple CNN)
        self.obs_encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),  # 64 -> 32
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),  # 32 -> 16
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),  # 16 -> 8
            nn.ReLU(),
            nn.Flatten(),  # 64 * 8 * 8 = 4096
            nn.Linear(4096, 128),
            nn.ReLU()
        )
        
        # Trajectory generator
        # Input: obstacle_features (128) + start (2) + target (2) = 132
        self.trajectory_net = nn.Sequential(
            nn.Linear(132, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, n_waypoints * 2),
            nn.Sigmoid()  # Output in [0, 1]
        )
    
    def forward(self, obstacle_map, start, target):
        obs_features = self.obs_encoder(obstacle_map)
        combined = torch.cat([obs_features, start, target], dim=1)
        output = self.trajectory_net(combined)
        trajectory = output.view(-1, self.n_waypoints, 2)
        return trajectory


# =============================================================================
# MODEL 2: DEEPER CNN + RESIDUAL
# =============================================================================

class ResidualBlock(nn.Module):
    """Residual block for deeper networks."""
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return self.relu(out)


class DeepCNNTrajectory(nn.Module):
    """
    Deeper CNN with residual connections.
    Better feature extraction for obstacle maps.
    """
    
    def __init__(self, n_waypoints: int = 50):
        super().__init__()
        self.n_waypoints = n_waypoints
        
        # Deep CNN encoder with residuals
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=1, padding=1),  # 64 -> 64
            nn.BatchNorm2d(32),
            nn.ReLU(),
            ResidualBlock(32),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),  # 64 -> 32
            nn.BatchNorm2d(64),
            nn.ReLU(),
            ResidualBlock(64),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),  # 32 -> 16
            nn.BatchNorm2d(128),
            nn.ReLU(),
            ResidualBlock(128),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),  # 16 -> 8
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),  # Global average pooling
            nn.Flatten(),  # 256
        )
        
        # Trajectory generator with skip connections
        self.fc1 = nn.Linear(256 + 4, 512)  # +4 for start/target
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc_out = nn.Linear(256, n_waypoints * 2)
        self.dropout = nn.Dropout(0.2)
        self.relu = nn.ReLU()
    
    def forward(self, obstacle_map, start, target):
        # Encode obstacles
        features = self.encoder(obstacle_map)  # (B, 256)
        
        # Concatenate with endpoints
        combined = torch.cat([features, start, target], dim=1)  # (B, 260)
        
        # MLP with skip connections
        x1 = self.relu(self.fc1(combined))
        x1 = self.dropout(x1)
        x2 = self.relu(self.fc2(x1))
        x2 = self.dropout(x2)
        x3 = self.relu(self.fc3(x2 + x1[:, :512]))  # Skip connection (truncated)
        
        output = torch.sigmoid(self.fc_out(x3))
        trajectory = output.view(-1, self.n_waypoints, 2)
        
        return trajectory


# =============================================================================
# MODEL 3: U-NET STYLE ENCODER-DECODER
# =============================================================================

class UNetTrajectory(nn.Module):
    """
    U-Net inspired architecture.
    Better at preserving spatial information.
    """
    
    def __init__(self, n_waypoints: int = 50):
        super().__init__()
        self.n_waypoints = n_waypoints
        
        # Encoder (downsampling)
        self.enc1 = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.pool1 = nn.MaxPool2d(2)  # 64 -> 32
        
        self.enc2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.pool2 = nn.MaxPool2d(2)  # 32 -> 16
        
        self.enc3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.pool3 = nn.MaxPool2d(2)  # 16 -> 8
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()  # 256
        )
        
        # Trajectory decoder (fully connected)
        self.decoder = nn.Sequential(
            nn.Linear(256 + 4, 512),  # +4 for start/target
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, n_waypoints * 2),
            nn.Sigmoid()
        )
    
    def forward(self, obstacle_map, start, target):
        # Encoder
        e1 = self.enc1(obstacle_map)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        
        # Bottleneck
        features = self.bottleneck(self.pool3(e3))  # (B, 256)
        
        # Concatenate with endpoints
        combined = torch.cat([features, start, target], dim=1)
        
        # Decode to trajectory
        output = self.decoder(combined)
        trajectory = output.view(-1, self.n_waypoints, 2)
        
        return trajectory


# =============================================================================
# MODEL 4: ATTENTION-BASED (Simplified Transformer)
# =============================================================================

class AttentionTrajectory(nn.Module):
    """
    Attention-based model.
    Uses self-attention to reason about obstacle relationships.
    """
    
    def __init__(self, n_waypoints: int = 50):
        super().__init__()
        self.n_waypoints = n_waypoints
        
        # Patch embedding (treat 8x8 patches as tokens)
        self.patch_size = 8
        self.n_patches = (64 // self.patch_size) ** 2  # 64 patches
        self.embed_dim = 128
        
        self.patch_embed = nn.Sequential(
            nn.Conv2d(1, self.embed_dim, self.patch_size, stride=self.patch_size),
            nn.Flatten(2),  # (B, embed_dim, n_patches)
        )
        
        # Position embedding
        self.pos_embed = nn.Parameter(torch.randn(1, self.n_patches, self.embed_dim) * 0.02)
        
        # Self-attention layers
        self.attention1 = nn.MultiheadAttention(self.embed_dim, num_heads=4, batch_first=True)
        self.norm1 = nn.LayerNorm(self.embed_dim)
        self.attention2 = nn.MultiheadAttention(self.embed_dim, num_heads=4, batch_first=True)
        self.norm2 = nn.LayerNorm(self.embed_dim)
        
        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim * 4),
            nn.GELU(),
            nn.Linear(self.embed_dim * 4, self.embed_dim)
        )
        self.norm3 = nn.LayerNorm(self.embed_dim)
        
        # Trajectory decoder
        self.decoder = nn.Sequential(
            nn.Linear(self.embed_dim + 4, 512),  # +4 for start/target
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, n_waypoints * 2),
            nn.Sigmoid()
        )
    
    def forward(self, obstacle_map, start, target):
        # Patch embedding
        x = self.patch_embed(obstacle_map)  # (B, embed_dim, n_patches)
        x = x.permute(0, 2, 1)  # (B, n_patches, embed_dim)
        x = x + self.pos_embed
        
        # Self-attention blocks
        attn_out, _ = self.attention1(x, x, x)
        x = self.norm1(x + attn_out)
        
        attn_out, _ = self.attention2(x, x, x)
        x = self.norm2(x + attn_out)
        
        x = self.norm3(x + self.ffn(x))
        
        # Global average pooling
        features = x.mean(dim=1)  # (B, embed_dim)
        
        # Decode to trajectory
        combined = torch.cat([features, start, target], dim=1)
        output = self.decoder(combined)
        trajectory = output.view(-1, self.n_waypoints, 2)
        
        return trajectory


# =============================================================================
# TRAINING
# =============================================================================

def generate_training_data(n_samples: int = 500, seed: int = 42, save_path: str = None):
    """Generate training data using A* as expert."""
    
    # Check if cached data exists
    if save_path and Path(save_path).exists():
        print(f"\n✓ Loading cached training data from {save_path}")
        cached = np.load(save_path)
        return {
            'obstacle_maps': cached['obstacle_maps'],
            'start_positions': cached['start_positions'],
            'target_positions': cached['target_positions'],
            'trajectories': cached['trajectories']
        }
    
    print(f"\nGenerating {n_samples} training samples using A* expert...")
    
    config = EnvironmentConfig(seed=seed)
    env = SpaceEnvironment(config)
    
    # Storage
    obstacle_maps = []
    start_positions = []
    target_positions = []
    trajectories = []
    
    success_count = 0
    
    for i in tqdm(range(n_samples * 2), desc="Generating"):  # Generate more, filter valid
        if success_count >= n_samples:
            break
        
        env.reset(seed=seed + i)
        
        # Solve with A*
        path = env.solve_astar()
        
        if path is not None:
            # Check collision-free
            valid, _ = env.check_trajectory(path, margin=5.0)
            
            if valid:
                # Interpolate to fixed length
                traj = env.interpolate_trajectory(path, n_points=50)
                
                # Verify interpolated is also valid
                valid2, _ = env.check_trajectory(traj, margin=3.0)
                
                if valid2:
                    # Normalize to [0, 1]
                    obs_map = env.get_obstacle_map(resolution=64)
                    start_norm = env.start_pos / env.config.width
                    target_norm = env.target_pos / env.config.width
                    traj_norm = traj / env.config.width
                    
                    obstacle_maps.append(obs_map)
                    start_positions.append(start_norm)
                    target_positions.append(target_norm)
                    trajectories.append(traj_norm)
                    
                    success_count += 1
    
    print(f"✓ Generated {success_count} valid samples")
    
    result = {
        'obstacle_maps': np.array(obstacle_maps),
        'start_positions': np.array(start_positions),
        'target_positions': np.array(target_positions),
        'trajectories': np.array(trajectories)
    }
    
    # Save to cache
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            save_path,
            obstacle_maps=result['obstacle_maps'],
            start_positions=result['start_positions'],
            target_positions=result['target_positions'],
            trajectories=result['trajectories']
        )
        print(f"✓ Saved training data to {save_path}")
    
    return result


def train_model(data: dict, model_class, model_name: str, epochs: int = 50, batch_size: int = 32, save_path: str = None):
    """Train model on A* expert data."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nTraining {model_name} on {device}...")
    
    # Prepare data
    n_samples = len(data['trajectories'])
    n_train = int(n_samples * 0.9)
    
    indices = np.random.permutation(n_samples)
    train_idx = indices[:n_train]
    val_idx = indices[n_train:]
    
    # Create model
    model = model_class(n_waypoints=50).to(device)
    
    # Count parameters
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {n_params:,}")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    criterion = nn.MSELoss()
    
    best_val_loss = float('inf')
    best_model_state = None
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_losses = []
        
        np.random.shuffle(train_idx)
        for i in range(0, len(train_idx), batch_size):
            batch_idx = train_idx[i:i+batch_size]
            
            obs_maps = torch.FloatTensor(data['obstacle_maps'][batch_idx]).unsqueeze(1).to(device)
            starts = torch.FloatTensor(data['start_positions'][batch_idx]).to(device)
            targets = torch.FloatTensor(data['target_positions'][batch_idx]).to(device)
            trajs = torch.FloatTensor(data['trajectories'][batch_idx]).to(device)
            
            optimizer.zero_grad()
            pred = model(obs_maps, starts, targets)
            loss = criterion(pred, trajs)
            loss.backward()
            optimizer.step()
            
            train_losses.append(loss.item())
        
        # Validation
        model.eval()
        val_losses = []
        
        with torch.no_grad():
            for i in range(0, len(val_idx), batch_size):
                batch_idx = val_idx[i:i+batch_size]
                
                obs_maps = torch.FloatTensor(data['obstacle_maps'][batch_idx]).unsqueeze(1).to(device)
                starts = torch.FloatTensor(data['start_positions'][batch_idx]).to(device)
                targets = torch.FloatTensor(data['target_positions'][batch_idx]).to(device)
                trajs = torch.FloatTensor(data['trajectories'][batch_idx]).to(device)
                
                pred = model(obs_maps, starts, targets)
                loss = criterion(pred, trajs)
                val_losses.append(loss.item())
        
        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        
        scheduler.step(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            if save_path:
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'config': {'n_waypoints': 50, 'model_name': model_name}
                }, save_path)
        
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}: train_loss={train_loss:.6f}, val_loss={val_loss:.6f}")
    
    # Load best model
    model.load_state_dict(best_model_state)
    print(f"✓ Training complete. Best val_loss: {best_val_loss:.6f}")
    return model, best_val_loss


# =============================================================================
# BENCHMARK
# =============================================================================

def run_benchmark(model, n_envs: int = 50, seed: int = 1000, use_cpu: bool = False):
    """Run 3-way comparison: A* vs MLP vs Hybrid.
    
    Args:
        use_cpu: Force CPU inference (faster for small models due to no GPU transfer overhead)
    """
    # Choose device - CPU can be faster for single-sample inference!
    if use_cpu:
        device = torch.device("cpu")
        model_cpu = model.cpu()
        model_cpu.eval()
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.eval()
        model_cpu = model
    
    config = EnvironmentConfig(seed=seed)
    env = SpaceEnvironment(config)
    
    results = {
        'astar': {'success': [], 'path_length': [], 'smoothness': [], 'time_ms': []},
        'mlp': {'success': [], 'path_length': [], 'smoothness': [], 'time_ms': []},
        'hybrid': {'success': [], 'path_length': [], 'smoothness': [], 'time_ms': []}
    }
    
    print(f"\nRunning benchmark on {n_envs} environments (device: {device})...")
    
    # Warmup for accurate timing
    if not use_cpu and torch.cuda.is_available():
        dummy = torch.randn(1, 1, 64, 64).to(device)
        dummy_s = torch.randn(1, 2).to(device)
        with torch.no_grad():
            _ = model_cpu(dummy, dummy_s, dummy_s)
        torch.cuda.synchronize()
    
    for i in tqdm(range(n_envs), desc="Benchmark"):
        env.reset(seed=seed + i)
        
        # ===== 1. A* (Baseline) =====
        t0 = time.perf_counter()
        astar_path = env.solve_astar()
        astar_time = (time.perf_counter() - t0) * 1000
        
        if astar_path is not None:
            astar_traj = env.interpolate_trajectory(astar_path, n_points=50)
            valid, _ = env.check_trajectory(astar_traj, margin=3.0)
            
            results['astar']['success'].append(valid)
            if valid:
                results['astar']['path_length'].append(env.compute_path_length(astar_traj))
                results['astar']['smoothness'].append(env.compute_smoothness(astar_traj))
            else:
                results['astar']['path_length'].append(np.nan)
                results['astar']['smoothness'].append(np.nan)
        else:
            results['astar']['success'].append(False)
            results['astar']['path_length'].append(np.nan)
            results['astar']['smoothness'].append(np.nan)
        
        results['astar']['time_ms'].append(astar_time)
        
        # ===== 2. MLP Only (FULL inference time) =====
        t0 = time.perf_counter()
        
        obs_map = torch.FloatTensor(env.get_obstacle_map(64)).unsqueeze(0).unsqueeze(0).to(device)
        start = torch.FloatTensor(env.start_pos / env.config.width).unsqueeze(0).to(device)
        target = torch.FloatTensor(env.target_pos / env.config.width).unsqueeze(0).to(device)
        
        with torch.no_grad():
            pred = model_cpu(obs_map, start, target)
        
        # Sync for accurate GPU timing
        if not use_cpu and torch.cuda.is_available():
            torch.cuda.synchronize()
        
        mlp_traj = pred.cpu().numpy()[0] * env.config.width  # Denormalize
        mlp_traj[0] = env.start_pos  # Fix endpoints
        mlp_traj[-1] = env.target_pos
        
        mlp_time = (time.perf_counter() - t0) * 1000
        
        valid, _ = env.check_trajectory(mlp_traj, margin=3.0)
        results['mlp']['success'].append(valid)
        
        if valid:
            results['mlp']['path_length'].append(env.compute_path_length(mlp_traj))
            results['mlp']['smoothness'].append(env.compute_smoothness(mlp_traj))
        else:
            results['mlp']['path_length'].append(np.nan)
            results['mlp']['smoothness'].append(np.nan)
        
        results['mlp']['time_ms'].append(mlp_time)
        
        # ===== 3. Hybrid (MLP + Repair) - INCLUDES MLP inference =====
        t0 = time.perf_counter()
        
        # MLP inference (included in hybrid time!)
        obs_map = torch.FloatTensor(env.get_obstacle_map(64)).unsqueeze(0).unsqueeze(0).to(device)
        start = torch.FloatTensor(env.start_pos / env.config.width).unsqueeze(0).to(device)
        target = torch.FloatTensor(env.target_pos / env.config.width).unsqueeze(0).to(device)
        
        with torch.no_grad():
            pred = model_cpu(obs_map, start, target)
        
        if not use_cpu and torch.cuda.is_available():
            torch.cuda.synchronize()
        
        hybrid_traj = pred.cpu().numpy()[0] * env.config.width
        hybrid_traj[0] = env.start_pos
        hybrid_traj[-1] = env.target_pos
        
        # Simple collision repair
        for iteration in range(5):
            collision_found = False
            
            for j in range(1, len(hybrid_traj) - 1):
                point = hybrid_traj[j]
                
                for obs in env.obstacles:
                    dist = np.linalg.norm(point - obs.center)
                    safe_dist = obs.radius + 10.0  # 10m margin
                    
                    if dist < safe_dist:
                        collision_found = True
                        # Push away
                        if dist < 1e-6:
                            direction = np.array([1.0, 0.0])
                        else:
                            direction = (point - obs.center) / dist
                        
                        hybrid_traj[j] = obs.center + direction * (safe_dist + 5.0)
                        
                        # Clamp to bounds
                        hybrid_traj[j] = np.clip(hybrid_traj[j], 10, env.config.width - 10)
            
            if not collision_found:
                break
        
        # Smoothing
        for _ in range(2):
            smoothed = hybrid_traj.copy()
            for j in range(2, len(hybrid_traj) - 2):
                smoothed[j] = 0.5 * hybrid_traj[j] + 0.25 * (hybrid_traj[j-1] + hybrid_traj[j+1])
            hybrid_traj = smoothed
        
        hybrid_traj[0] = env.start_pos
        hybrid_traj[-1] = env.target_pos
        
        hybrid_time = (time.perf_counter() - t0) * 1000
        
        valid, _ = env.check_trajectory(hybrid_traj, margin=3.0)
        results['hybrid']['success'].append(valid)
        
        if valid:
            results['hybrid']['path_length'].append(env.compute_path_length(hybrid_traj))
            results['hybrid']['smoothness'].append(env.compute_smoothness(hybrid_traj))
        else:
            results['hybrid']['path_length'].append(np.nan)
            results['hybrid']['smoothness'].append(np.nan)
        
        results['hybrid']['time_ms'].append(hybrid_time)
    
    # Move model back to original device if needed
    if use_cpu and torch.cuda.is_available():
        model.cuda()
    
    return results


def print_results(results: dict):
    """Print benchmark results."""
    print("\n" + "=" * 80)
    print("BENCHMARK RESULTS")
    print("=" * 80)
    
    print(f"\n{'Metric':<20} {'A* (Baseline)':>18} {'MLP Only':>18} {'Hybrid':>18}")
    print("-" * 80)
    
    # Success rate
    astar_sr = np.mean(results['astar']['success']) * 100
    mlp_sr = np.mean(results['mlp']['success']) * 100
    hybrid_sr = np.mean(results['hybrid']['success']) * 100
    print(f"{'Success Rate':<20} {astar_sr:>17.1f}% {mlp_sr:>17.1f}% {hybrid_sr:>17.1f}%")
    
    # Path length
    astar_pl = np.nanmean(results['astar']['path_length'])
    mlp_pl = np.nanmean(results['mlp']['path_length'])
    hybrid_pl = np.nanmean(results['hybrid']['path_length'])
    print(f"{'Avg Path Length':<20} {astar_pl:>16.1f} m {mlp_pl:>16.1f} m {hybrid_pl:>16.1f} m")
    
    # Smoothness
    astar_sm = np.nanmean(results['astar']['smoothness'])
    mlp_sm = np.nanmean(results['mlp']['smoothness'])
    hybrid_sm = np.nanmean(results['hybrid']['smoothness'])
    print(f"{'Smoothness (lower=better)':<20} {astar_sm:>16.2f} {mlp_sm:>16.2f} {hybrid_sm:>16.2f}")
    
    # Time
    astar_t = np.mean(results['astar']['time_ms'])
    mlp_t = np.mean(results['mlp']['time_ms'])
    hybrid_t = np.mean(results['hybrid']['time_ms'])
    print(f"{'Avg Time':<20} {astar_t:>15.2f} ms {mlp_t:>15.2f} ms {hybrid_t:>15.2f} ms")
    
    print("\n" + "=" * 80)
    print("ANALYSIS")
    print("=" * 80)
    
    print(f"\n1. MLP Learning (MLP vs A*):")
    learning_eff = mlp_sr / astar_sr * 100 if astar_sr > 0 else 0
    print(f"   Learning efficiency: {learning_eff:.1f}%")
    if mlp_sr <= astar_sr:
        print(f"   ✓ CORRECT: MLP ({mlp_sr:.1f}%) <= A* ({astar_sr:.1f}%)")
    else:
        print(f"   ⚠️ ANOMALY: MLP > A*")
    
    print(f"\n2. Hybrid Contribution:")
    repair_gain = hybrid_sr - mlp_sr
    print(f"   Collision repair adds: {repair_gain:+.1f}% success")
    
    print(f"\n3. Overall:")
    if hybrid_sr >= astar_sr * 0.95:
        print(f"   ✓ Hybrid is competitive with A* baseline")
    if hybrid_t < astar_t:
        speedup = astar_t / hybrid_t
        print(f"   ✓ Hybrid is {speedup:.1f}x faster than A*")
    
    print("\n" + "=" * 80)


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 80)
    print("MODEL COMPARISON BENCHMARK (GPU vs CPU)")
    print("=" * 80)
    
    Path('./checkpoints').mkdir(exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training device: {device}")
    
    # Step 1: Generate/Load training data (10K samples)
    data = generate_training_data(
        n_samples=10000, 
        seed=42, 
        save_path='./data/training_10k.npz'
    )
    
    # Models to compare
    models = [
        ('SimpleMLP', SimpleTrajectoryMLP),
        ('DeepCNN', DeepCNNTrajectory),
        ('UNet', UNetTrajectory),
        ('Attention', AttentionTrajectory),
    ]
    
    # Results storage
    all_results = {}
    model_stats = []
    
    for model_name, model_class in models:
        print("\n" + "=" * 80)
        print(f"TRAINING: {model_name}")
        print("=" * 80)
        
        # Train
        model, val_loss = train_model(
            data, 
            model_class=model_class,
            model_name=model_name,
            epochs=100, 
            batch_size=64,
            save_path=f'./checkpoints/{model_name.lower()}_best.pth'
        )
        
        # Benchmark on BOTH GPU and CPU
        print(f"\nBenchmarking {model_name}...")
        
        # GPU benchmark
        if torch.cuda.is_available():
            print("  → GPU inference...")
            results_gpu = run_benchmark(model, n_envs=50, seed=1000, use_cpu=False)
            gpu_hybrid_time = np.mean(results_gpu['hybrid']['time_ms'])
        else:
            results_gpu = None
            gpu_hybrid_time = float('inf')
        
        # CPU benchmark
        print("  → CPU inference...")
        results_cpu = run_benchmark(model, n_envs=50, seed=1000, use_cpu=True)
        cpu_hybrid_time = np.mean(results_cpu['hybrid']['time_ms'])
        
        # Use faster one
        if gpu_hybrid_time < cpu_hybrid_time:
            results = results_gpu
            best_device = 'GPU'
            best_time = gpu_hybrid_time
        else:
            results = results_cpu
            best_device = 'CPU'
            best_time = cpu_hybrid_time
        
        all_results[model_name] = results
        
        # Collect stats
        mlp_sr = np.mean(results['mlp']['success']) * 100
        hybrid_sr = np.mean(results['hybrid']['success']) * 100
        hybrid_time = np.mean(results['hybrid']['time_ms'])
        
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        model_stats.append({
            'name': model_name,
            'params': n_params,
            'val_loss': val_loss,
            'mlp_success': mlp_sr,
            'hybrid_success': hybrid_sr,
            'hybrid_time': hybrid_time,
            'gpu_time': gpu_hybrid_time if results_gpu else None,
            'cpu_time': cpu_hybrid_time,
            'best_device': best_device
        })
        
        print(f"  ✓ {model_name}: GPU={gpu_hybrid_time:.2f}ms, CPU={cpu_hybrid_time:.2f}ms → Best: {best_device}")
    
    # Final comparison table
    print("\n" + "=" * 80)
    print("FINAL MODEL COMPARISON")
    print("=" * 80)
    
    # A* baseline (same for all)
    astar_sr = np.mean(all_results['SimpleMLP']['astar']['success']) * 100
    astar_time = np.mean(all_results['SimpleMLP']['astar']['time_ms'])
    
    print(f"\nA* Baseline: {astar_sr:.1f}% success, {astar_time:.2f}ms")
    print()
    
    print(f"{'Model':<12} {'Params':>10} {'Val Loss':>10} {'Hybrid %':>10} {'GPU (ms)':>10} {'CPU (ms)':>10} {'Best':>6} {'vs A*':>8}")
    print("-" * 90)
    
    for stat in model_stats:
        gpu_str = f"{stat['gpu_time']:.2f}" if stat['gpu_time'] else "N/A"
        cpu_str = f"{stat['cpu_time']:.2f}"
        best_time = min(stat['gpu_time'] or float('inf'), stat['cpu_time'])
        speedup = astar_time / best_time if best_time > 0 else 0
        speedup_str = f"{speedup:.2f}x" if speedup >= 1 else f"{1/speedup:.1f}x slower"
        
        print(f"{stat['name']:<12} {stat['params']:>10,} {stat['val_loss']:>10.6f} {stat['hybrid_success']:>9.1f}% {gpu_str:>10} {cpu_str:>10} {stat['best_device']:>6} {speedup_str:>8}")
    
    # Best model
    best = max(model_stats, key=lambda x: x['hybrid_success'])
    best_time = min(best['gpu_time'] or float('inf'), best['cpu_time'])
    speedup = astar_time / best_time if best_time > 0 else 0
    
    print()
    print(f"🏆 BEST MODEL: {best['name']} with {best['hybrid_success']:.1f}% success")
    print(f"   Best device: {best['best_device']} ({best_time:.2f}ms)")
    
    if speedup >= 1:
        print(f"   ✓ {speedup:.2f}x faster than A*")
    else:
        print(f"   ⚠️ {1/speedup:.1f}x slower than A* ({astar_time:.2f}ms)")
    
    # =========================================================================
    # SAVE ALL RESULTS FOR REPORTING
    # =========================================================================
    from datetime import datetime
    import json
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path('./results')
    results_dir.mkdir(exist_ok=True)
    
    # 1. Save detailed benchmark results
    benchmark_report = {
        'timestamp': timestamp,
        'training_samples': len(data['trajectories']),
        'test_environments': 50,
        'baseline': {
            'name': 'A*',
            'success_rate': float(astar_sr),
            'avg_time_ms': float(astar_time),
            'avg_path_length': float(np.nanmean(all_results['SimpleMLP']['astar']['path_length'])),
            'avg_smoothness': float(np.nanmean(all_results['SimpleMLP']['astar']['smoothness']))
        },
        'models': []
    }
    
    for stat in model_stats:
        model_name = stat['name']
        results = all_results[model_name]
        best_time = min(stat['gpu_time'] or float('inf'), stat['cpu_time'])
        
        model_report = {
            'name': model_name,
            'parameters': stat['params'],
            'val_loss': float(stat['val_loss']),
            'mlp_only': {
                'success_rate': float(stat['mlp_success']),
                'avg_path_length': float(np.nanmean(results['mlp']['path_length'])),
                'avg_smoothness': float(np.nanmean(results['mlp']['smoothness'])),
            },
            'hybrid': {
                'success_rate': float(stat['hybrid_success']),
                'avg_path_length': float(np.nanmean(results['hybrid']['path_length'])),
                'avg_smoothness': float(np.nanmean(results['hybrid']['smoothness'])),
                'gpu_time_ms': float(stat['gpu_time']) if stat['gpu_time'] else None,
                'cpu_time_ms': float(stat['cpu_time']),
                'best_device': stat['best_device'],
                'best_time_ms': float(best_time)
            },
            'speedup_vs_astar': float(astar_time / best_time) if best_time > 0 else None
        }
        benchmark_report['models'].append(model_report)
    
    # Add best model summary
    best_time = min(best['gpu_time'] or float('inf'), best['cpu_time'])
    benchmark_report['best_model'] = {
        'name': best['name'],
        'hybrid_success_rate': float(best['hybrid_success']),
        'best_device': best['best_device'],
        'best_time_ms': float(best_time),
        'speedup_vs_astar': float(astar_time / best_time) if best_time > 0 else None
    }
    
    # Save JSON report
    report_path = results_dir / f'model_comparison_{timestamp}.json'
    with open(report_path, 'w') as f:
        json.dump(benchmark_report, f, indent=2)
    print(f"\n✓ Detailed report saved to: {report_path}")
    
    # 2. Save CSV summary
    import csv
    csv_path = results_dir / f'model_comparison_{timestamp}.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Model', 'Parameters', 'Val Loss', 'Hybrid Success %', 'GPU Time (ms)', 'CPU Time (ms)', 'Best Device', 'Best Time (ms)', 'Speedup vs A*'])
        
        # A* baseline
        writer.writerow([
            'A* (Baseline)', '-', '-', f'{astar_sr:.1f}', '-', '-', 'CPU', f'{astar_time:.2f}', '1.00x'
        ])
        
        for stat in model_stats:
            results = all_results[stat['name']]
            best_time = min(stat['gpu_time'] or float('inf'), stat['cpu_time'])
            speedup = astar_time / best_time if best_time > 0 else 0
            
            writer.writerow([
                stat['name'],
                stat['params'],
                f'{stat["val_loss"]:.6f}',
                f'{stat["hybrid_success"]:.1f}',
                f'{stat["gpu_time"]:.2f}' if stat['gpu_time'] else 'N/A',
                f'{stat["cpu_time"]:.2f}',
                stat['best_device'],
                f'{best_time:.2f}',
                f'{speedup:.2f}x' if speedup >= 1 else f'{1/speedup:.1f}x slower'
            ])
    print(f"✓ CSV summary saved to: {csv_path}")
    
    # 3. Save per-environment results for detailed analysis
    detailed_path = results_dir / f'detailed_results_{timestamp}.npz'
    np.savez_compressed(
        detailed_path,
        model_names=[s['name'] for s in model_stats],
        model_stats=model_stats,
        astar_success=all_results['SimpleMLP']['astar']['success'],
        astar_path_length=all_results['SimpleMLP']['astar']['path_length'],
        astar_time=all_results['SimpleMLP']['astar']['time_ms'],
        **{f'{name}_mlp_success': all_results[name]['mlp']['success'] for name in all_results},
        **{f'{name}_hybrid_success': all_results[name]['hybrid']['success'] for name in all_results},
        **{f'{name}_hybrid_time': all_results[name]['hybrid']['time_ms'] for name in all_results}
    )
    print(f"✓ Detailed per-env results saved to: {detailed_path}")
    
    print("\n" + "=" * 80)
    print("ALL RESULTS SAVED FOR REPORTING")
    print("=" * 80)
    print(f"\nFiles saved:")
    print(f"  1. Training data:     ./data/training_10k.npz")
    print(f"  2. Model checkpoints: ./checkpoints/<model>_best.pth")
    print(f"  3. JSON report:       {report_path}")
    print(f"  4. CSV summary:       {csv_path}")
    print(f"  5. Detailed results:  {detailed_path}")
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
