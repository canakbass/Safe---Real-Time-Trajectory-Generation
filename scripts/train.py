
"""
Training Script
===============
Trains the 3D Trajectory Model using generated data.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import sys
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt

# Add src
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model.architecture import TrajectoryNet3D
from src.model.loss import TrajectoryLoss

class TrajectoryDataset(Dataset):
    def __init__(self, data_path):
        print(f"Loading data from {data_path}...")
        data = np.load(data_path)
        
        # Keep as compressed/smaller types in RAM
        # grids from npz are boolean or packed. unpack if needed but keep as uint8/bool
        self.grids = data['grids'] # (N, 100, 100, 100) bool or uint8
        self.starts = torch.FloatTensor(data['starts'])
        self.goals = torch.FloatTensor(data['goals'])
        self.targets = torch.FloatTensor(data['paths'])
        
        print(f"Dataset Size: {len(self.grids)}")

    def __len__(self):
        return len(self.grids)

    def __getitem__(self, idx):
        # Convert to float32 on the fly (Lightweight RAM usage)
        grid = torch.from_numpy(self.grids[idx]).float().unsqueeze(0) # Add channel dim -> (1, 100, 100, 100)
        return grid, self.starts[idx], self.goals[idx], self.targets[idx]

def train(data_path="data/train_data.npz", epochs=100, batch_size=16, lr=1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. Prepare Data
    if not Path(data_path).exists():
        print(f"Error: {data_path} not found. Run generate_data.py first.")
        return
        
    full_dataset = TrajectoryDataset(data_path)
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_set, val_set = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    
    # 2. Model & Loss
    model = TrajectoryNet3D().to(device)
    criterion = TrajectoryLoss(smoothness_weight=0.5)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)
    
    # 3. Training Loop
    history = {'train_loss': [], 'val_loss': []}
    best_val_loss = float('inf')
    
    Path("checkpoints").mkdir(exist_ok=True)
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for grid, start, goal, target in pbar:
            grid, start, goal, target = grid.to(device), start.to(device), goal.to(device), target.to(device)
            
            optimizer.zero_grad()
            pred = model(grid, start, goal)
            
            loss, mse, smooth = criterion(pred, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            pbar.set_postfix({'loss': loss.item(), 'mse': mse.item()})
            
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for grid, start, goal, target in val_loader:
                grid, start, goal, target = grid.to(device), start.to(device), goal.to(device), target.to(device)
                pred = model(grid, start, goal)
                loss, _, _ = criterion(pred, target)
                val_loss += loss.item()
                
        avg_val_loss = val_loss / len(val_loader)
        scheduler.step(avg_val_loss)
        
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        
        print(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        
        # Save Best
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "checkpoints/best_model_3d.pth")
            print("  -> Saved best model.")
            
    # Plot history
    plt.figure()
    plt.plot(history['train_loss'], label='Train')
    plt.plot(history['val_loss'], label='Val')
    plt.title("Training History")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("results/training_curve.png")
    print("Training complete. History saved.")

if __name__ == "__main__":
    # Use verify batch if train data doesn't exist, just for demo
    if Path("data/train_data.npz").exists():
        path = "data/train_data.npz"
    elif Path("data/verify_batch.npz").exists():
        print("Using verification batch for training test...")
        path = "data/verify_batch.npz"
    else:
        path = "data/train_data.npz"
        
    train(data_path=path, epochs=100) # 100 epochs for proper convergence
