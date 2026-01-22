
import torch
import torch.nn as nn
import torch.nn.functional as F

class PointNetEncoder(nn.Module):
    """
    Encodes a set of N obstacles (Point Cloud) into a global feature vector.
    Input: (Batch, N, 4) -> [dx, dy, dz, radius]
    Output: (Batch, feature_dim)
    """
    def __init__(self, input_dim=4, feature_dim=128):
        super().__init__()
        
        # Shared MLP for per-point feature extraction
        # (B, N, 4) -> (B, N, 64) -> (B, N, 128)
        self.mlp1 = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64), # Note: BN expects (B, C, N) usually, handle carefully
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128)
        )
        
        self.fc_global = nn.Linear(128, feature_dim)
        
    def forward(self, x):
        """
        Args:
            x: Tensor of shape (Batch, N_obs, 4)
        """
        # Transpose for BatchNorm1d: (B, N, C) -> (B, C, N)
        # But wait, BN1d applies to C dimension. 
        # If input is (B, N, C), PyTorch BN expects (B, C, N) for 1D convolution style or (B, C) for standard.
        # Let's map points independently using Linear layers.
        # Linear applies to last dim. (B, N, 4) -> Linear(4->64) -> (B, N, 64). Perfect.
        # BUT BatchNorm1d expects (B, C, L) or (B, C). 
        # If we have (B, N, C), we need to permute to (B, C, N) for BN.
        
        B, N, C = x.shape
        
        # 1. Per-Point Encoding
        # x: (B, N, 4)
        x = F.relu(self.mlp1[0](x)) # Linear 64
        
        # BN trick: Permute to (B, C, N)
        x = x.permute(0, 2, 1) 
        x = self.mlp1[2](x) # BN 64
        x = x.permute(0, 2, 1) # Back to (B, N, C)
        
        x = F.relu(self.mlp1[3](x)) # Linear 128
        
        x = x.permute(0, 2, 1)
        x = self.mlp1[5](x) # BN 128
        x = x.permute(0, 2, 1) # (B, N, 128)
        
        # 2. Global Max Pooling (Symmetric Function)
        # Take max over N obstacles -> (B, 128)
        global_feat, _ = torch.max(x, dim=1)
        
        # 3. Final Projection
        out = self.fc_global(global_feat)
        return out

class VectorTrajectoryGenerator(nn.Module):
    """
    V5 Model: PointNet + MLP Decoder.
    Input:
        - Obstacles: (B, N, 4)
        - Start: (B, 3)
        - Goal: (B, 3)
    Output:
        - Trajectory: (B, Steps, 3)
    """
    def __init__(self, n_obstacles=20, obs_dim=4, feature_dim=128, path_len=10):
        super().__init__()
        
        self.path_len = path_len
        
        # 1. Obstacle Encoder (PointNet)
        self.obs_encoder = PointNetEncoder(input_dim=obs_dim, feature_dim=feature_dim)
        
        # 2. State Encoder (Start + Goal)
        # Start(3) + Goal(3) = 6
        self.state_encoder = nn.Sequential(
            nn.Linear(6, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )
        
        # 3. Fusion & Decoder
        # Concatenate: Obs(128) + State(64) = 192
        self.decoder = nn.Sequential(
            nn.Linear(feature_dim + 64, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, path_len * 3) # Output: Flattened path
        )
        
    def forward(self, obstacles, start, goal):
        """
        obstacles: (B, N, 4)
        start: (B, 3)
        goal: (B, 3)
        """
        # Encode Obstacles (Egocentric View expected)
        obs_feat = self.obs_encoder(obstacles)
        
        # Encode State
        state = torch.cat([start, goal], dim=1)
        state_feat = self.state_encoder(state)
        
        # Fuse
        fusion = torch.cat([obs_feat, state_feat], dim=1)
        
        # Decode
        flat_path = self.decoder(fusion)
        
        # Reshape to (B, Steps, 3)
        path = flat_path.view(-1, self.path_len, 3)
        
        # Add start position to path? 
        # Typically the model predicts waypoints *between* start and goal, 
        # or inclusive. Let's assume it predicts the *deltas* or absolute positions.
        # Since input obstacles are relative (egocentric), maybe predicting relative trajectory is easier?
        # Current V3 model predicts A -> B.
        # Let's predict absolute positions for simplicity with loss function compatibility.
        
        return path
