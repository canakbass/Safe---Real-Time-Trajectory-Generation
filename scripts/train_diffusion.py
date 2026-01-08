"""
Diffusion Model Training Script
===============================
Trains the trajectory diffusion model on expert demonstrations.

Usage:
    python scripts/train_diffusion.py --dataset data/expert_trajectories_latest.npz --epochs 100

Requirements:
    - PyTorch
    - Expert trajectory dataset (.npz)
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import json
from datetime import datetime
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.training.diffusion_model import (
    TrajectoryDiffusionModel,
    DiffusionConfig,
    TrajectoryDataset,
    compute_loss,
)


def train_diffusion_model(
    dataset_path: str,
    output_dir: str = "./checkpoints",
    n_epochs: int = 100,
    batch_size: int = 32,
    learning_rate: float = 1e-4,
    val_split: float = 0.1,
    device: str = "auto",
    save_every: int = 20,
) -> str:
    """
    Train the diffusion model on expert trajectories.
    
    Args:
        dataset_path: Path to .npz dataset
        output_dir: Directory to save checkpoints
        n_epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Adam learning rate
        val_split: Validation set fraction
        device: 'cuda', 'cpu', or 'auto'
        save_every: Save checkpoint every N epochs
    
    Returns:
        Path to best model checkpoint
    """
    print("=" * 70)
    print("DIFFUSION MODEL TRAINING")
    print("Layer 0: Learning from Expert Trajectories")
    print("=" * 70)
    
    # Device setup
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    print(f"\nDevice: {device}")
    
    # Load dataset
    print(f"\nLoading dataset: {dataset_path}")
    dataset = TrajectoryDataset(dataset_path, normalize=True)
    print(f"  Total samples: {len(dataset)}")
    
    # Train/val split
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    print(f"  Train samples: {train_size}")
    print(f"  Val samples: {val_size}")
    
    # Data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # Windows compatibility
        pin_memory=True if device.type == "cuda" else False,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    
    # Model
    config = DiffusionConfig(
        n_waypoints=50,
        hidden_dim=256,
        n_layers=4,
        n_timesteps=100,
        learning_rate=learning_rate,
        batch_size=batch_size,
    )
    
    model = TrajectoryDiffusionModel(config).to(device)
    
    # Count parameters
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel parameters: {n_params:,}")
    
    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
    
    # Output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'best_val_loss': float('inf'),
        'best_epoch': 0,
    }
    
    best_model_path = None
    
    print(f"\nTraining for {n_epochs} epochs...")
    print("-" * 70)
    
    for epoch in range(1, n_epochs + 1):
        # =====================================================================
        # TRAINING
        # =====================================================================
        model.train()
        train_losses = []
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{n_epochs} [Train]", leave=False)
        for batch in pbar:
            optimizer.zero_grad()
            
            loss = compute_loss(model, batch, device)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_losses.append(loss.item())
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_train_loss = np.mean(train_losses)
        history['train_loss'].append(avg_train_loss)
        
        # =====================================================================
        # VALIDATION
        # =====================================================================
        model.eval()
        val_losses = []
        
        with torch.no_grad():
            for batch in val_loader:
                loss = compute_loss(model, batch, device)
                val_losses.append(loss.item())
        
        avg_val_loss = np.mean(val_losses)
        history['val_loss'].append(avg_val_loss)
        
        # Update scheduler
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        
        # Print progress
        print(f"Epoch {epoch:3d}/{n_epochs} | "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {avg_val_loss:.4f} | "
              f"LR: {current_lr:.6f}")
        
        # =====================================================================
        # CHECKPOINTING
        # =====================================================================
        
        # Save best model
        if avg_val_loss < history['best_val_loss']:
            history['best_val_loss'] = avg_val_loss
            history['best_epoch'] = epoch
            
            best_model_path = output_path / "diffusion_best.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': avg_val_loss,
                'config': config.__dict__,
            }, best_model_path)
            print(f"  ✓ New best model saved! (val_loss: {avg_val_loss:.4f})")
        
        # Periodic checkpoint
        if epoch % save_every == 0:
            checkpoint_path = output_path / f"diffusion_epoch_{epoch}.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': avg_val_loss,
                'config': config.__dict__,
            }, checkpoint_path)
    
    print("-" * 70)
    print(f"\n✓ Training complete!")
    print(f"  Best epoch: {history['best_epoch']}")
    print(f"  Best val loss: {history['best_val_loss']:.4f}")
    print(f"  Best model: {best_model_path}")
    
    # Save training history
    history_path = output_path / "training_history.json"
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"  History: {history_path}")
    
    # Copy best model as "latest"
    import shutil
    latest_path = output_path / "diffusion_latest.pth"
    if best_model_path:
        shutil.copy(best_model_path, latest_path)
        print(f"  Latest: {latest_path}")
    
    return str(best_model_path)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Train Diffusion Model for Trajectory Generation"
    )
    
    parser.add_argument(
        '--dataset', '-d',
        type=str,
        default='./data/expert_trajectories_latest.npz',
        help='Path to expert trajectory dataset'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='./checkpoints',
        help='Output directory for checkpoints'
    )
    
    parser.add_argument(
        '--epochs', '-e',
        type=int,
        default=100,
        help='Number of training epochs (default: 100)'
    )
    
    parser.add_argument(
        '--batch-size', '-b',
        type=int,
        default=32,
        help='Batch size (default: 32)'
    )
    
    parser.add_argument(
        '--lr',
        type=float,
        default=1e-4,
        help='Learning rate (default: 1e-4)'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        choices=['auto', 'cuda', 'cpu'],
        default='auto',
        help='Device (default: auto)'
    )
    
    args = parser.parse_args()
    
    # Check dataset exists
    if not Path(args.dataset).exists():
        print(f"ERROR: Dataset not found: {args.dataset}")
        print("Run 'python scripts/generate_dataset.py' first!")
        sys.exit(1)
    
    train_diffusion_model(
        dataset_path=args.dataset,
        output_dir=args.output,
        n_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        device=args.device,
    )


if __name__ == "__main__":
    main()
