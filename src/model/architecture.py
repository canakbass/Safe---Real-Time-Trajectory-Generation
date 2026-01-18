
import torch
import torch.nn as nn
import torch.nn.functional as F

class TrajectoryNet3D(nn.Module):
    """
    3D Trajectory Generation Model (IAC 2026).
    
    Structure:
    1. 3D CNN Encoder (Compress 100x100x100 grid)
    2. Feature Injection (Start + Goal)
    3. MLP Decoder (Output 20 waypoints)
    """
    
    def __init__(self, input_dim=100, output_points=20):
        super(TrajectoryNet3D, self).__init__()
        self.output_points = output_points
        
        # 1. 3D CNN Encoder
        # Input: (B, 1, 100, 100, 100)
        self.encoder = nn.Sequential(
            # Layer 1
            nn.Conv3d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.MaxPool3d(2), # -> 50x50x50
            
            # Layer 2
            nn.Conv3d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.MaxPool3d(2), # -> 25x25x25
            
            # Layer 3
            nn.Conv3d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d(2), # -> 12x12x12
            
            # Layer 4 (Bottleneck)
            nn.Conv3d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.MaxPool3d(2), # -> 6x6x6
        )
        
        # Flatten size: 128 * 6 * 6 * 6 = 27,648
        self.flatten_dim = 128 * 6 * 6 * 6
        
        # 2. MLP Head
        # Injection: Features + Start(3) + Goal(3)
        self.fc_input_dim = self.flatten_dim + 6
        
        self.decoder = nn.Sequential(
            nn.Linear(self.fc_input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(512, 256),
            nn.ReLU(),
            
            nn.Linear(256, output_points * 3) # Output: 60 values
        )

    def forward(self, grid, start, goal):
        """
        grid: (B, 1, 100, 100, 100)
        start: (B, 3)
        goal: (B, 3)
        """
        # Encode Grid
        x = self.encoder(grid)
        x = x.view(x.size(0), -1) # Flatten
        
        # Inject Start/Goal
        combined = torch.cat([x, start, goal], dim=1)
        
        # Decode
        out = self.decoder(combined) 
        
        # Reshape to (B, N, 3)
        return out.view(-1, self.output_points, 3)

if __name__ == "__main__":
    # Test Shape
    model = TrajectoryNet3D()
    dummy_grid = torch.randn(2, 1, 100, 100, 100)
    dummy_start = torch.randn(2, 3)
    dummy_goal = torch.randn(2, 3)
    
    out = model(dummy_grid, dummy_start, dummy_goal)
    print(f"Output Shape: {out.shape}") # Should be (2, 20, 3)
    
    # Parameter count
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Parameters: {params:,}")
