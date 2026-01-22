
"""
Training Script for V5 Vector Model (PointNet)
==============================================
Trains the Egocentric PointNet model using vector data.
"""

import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model.vector_net import VectorTrajectoryGenerator
from src.model.loss import TrajectoryLoss

class VectorDataset(Dataset):
    def __init__(self, data_path):
        data = np.load(data_path)
        self.obstacles = torch.FloatTensor(data['obstacles']) # (N, 20, 4)
        self.starts = torch.FloatTensor(data['starts'])       # (N, 3)
        self.goals = torch.FloatTensor(data['goals'])         # (N, 3)
        self.paths = torch.FloatTensor(data['paths'])         # (N, 20, 3)
        
    def __len__(self):
        return len(self.obstacles)
    
    def __getitem__(self, idx):
        return {
            'obstacles': self.obstacles[idx],
            'start': self.starts[idx],
            'goal': self.goals[idx],
            'path': self.paths[idx]
        }

def train():
    # Config
    BATCH_SIZE = 64
    EPOCHS = 100 # Fast training
    LR = 1e-3
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {DEVICE}")
    
    # 1. Load Data
    try:
        dataset = VectorDataset("data/train_data_vector.npz")
    except FileNotFoundError:
        print("Data not found. Please run scripts/generate_vector_data.py first.")
        return

    # Split
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE)
    
    print(f"Data Loaded: {len(train_set)} train, {len(val_set)} val")
    
    # 2. Model
    model = VectorTrajectoryGenerator(
        n_obstacles=20, 
        obs_dim=4, 
        feature_dim=128, 
        path_len=20
    ).to(DEVICE)
    
    # 3. Loss & Optimizer
    criterion = TrajectoryLoss(smoothness_weight=0.1)
    
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
    
    # 4. Loop
    best_loss = float('inf')
    Path("checkpoints").mkdir(exist_ok=True)
    
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        
        for batch in train_loader:
            obs = batch['obstacles'].to(DEVICE)
            start = batch['start'].to(DEVICE)
            goal = batch['goal'].to(DEVICE)
            gt_path = batch['path'].to(DEVICE)
            
            optimizer.zero_grad()
            
            pred_path = model(obs, start, goal)
            
            # Loss: MSE + Smoothness
            loss, _, _ = criterion(pred_path, gt_path) 
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_train_loss = total_loss / len(train_loader)
        
        # Val
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                obs = batch['obstacles'].to(DEVICE)
                start = batch['start'].to(DEVICE)
                goal = batch['goal'].to(DEVICE)
                gt_path = batch['path'].to(DEVICE)
                
                pred_path = model(obs, start, goal)
                loss, _, _ = criterion(pred_path, gt_path)
                val_loss += loss.item()
                
        avg_val_loss = val_loss / len(val_loader)
        scheduler.step(avg_val_loss)
        
        print(f"Epoch {epoch+1}/{EPOCHS} | Train: {avg_train_loss:.4f} | Val: {avg_val_loss:.4f} | LR: {optimizer.param_groups[0]['lr']:.1e}")
        
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            torch.save(model.state_dict(), "checkpoints/best_model_vector_3d.pth")
            print("  --> Saved Best Model")
            
    print("Training Complete.")

if __name__ == "__main__":
    train()
