
import torch
import torch.nn as nn

class TrajectoryLoss(nn.Module):
    """
    Hybrid Loss Function:
    1. MSE Loss: Accuracy (Euclidean distance to partial ground truth).
    2. Smoothness Loss: Penalize high variance in segment lengths.
    """
    
    def __init__(self, smoothness_weight=0.1):
        super(TrajectoryLoss, self).__init__()
        self.mse = nn.MSELoss()
        self.alpha = smoothness_weight
        
    def forward(self, pred, target):
        """
        pred: (B, N, 3)
        target: (B, N, 3)
        """
        # 1. Accuracy Term
        loss_mse = self.mse(pred, target)
        
        # 2. Smoothness Term
        # Calculate element-wise differences (vectors between points)
        diffs = pred[:, 1:, :] - pred[:, :-1, :] # (B, N-1, 3)
        
        # Calculate lengths of these segments
        dists = torch.norm(diffs, dim=2) # (B, N-1)
        
        # Minimize variance of distances (aim for uniform spacing)
        # Var = Mean((x - Mean)^2)
        dist_mean = torch.mean(dists, dim=1, keepdim=True)
        dist_var = torch.mean((dists - dist_mean) ** 2)
        
        # Total Loss
        total_loss = loss_mse + self.alpha * dist_var
        
        return total_loss, loss_mse, dist_var

