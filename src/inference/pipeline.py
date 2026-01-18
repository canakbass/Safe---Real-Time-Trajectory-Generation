
import torch
import numpy as np
import time
from pathlib import Path
import sys

# Paths
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.model.architecture import TrajectoryNet3D
from src.inference.postprocess import PostProcessor

class InferencePipeline:
    """
    Full Inference Pipeline:
    Input (Grid, Start, Goal) -> Model -> Repair -> Smooth -> Trajectory
    """
    
    def __init__(self, model_path: str = None, device: str = "cpu"):
        self.device = torch.device(device)
        self.model = TrajectoryNet3D().to(self.device)
        self.post_proc = PostProcessor()
        
        if model_path:
            self.load_model(model_path)
        else:
            print("Warning: No model path provided. Using random weights.")
            self.model.eval()

    def load_model(self, path: str):
        if Path(path).exists():
            checkpoint = torch.load(path, map_location=self.device)
            # Support both directly saved state_dict or wrapped in dict
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
            self.model.eval()
            print(f"Model loaded from {path}")
        else:
            print(f"Error: Model file {path} not found!")

    def predict(self, grid: np.ndarray, start: np.ndarray, goal: np.ndarray, obstacles: list = []) -> dict:
        """
        Run full pipeline.
        
        Args:
            grid: (100, 100, 100) binary occupancy grid
            start: (3,) start coordinates
            goal: (3,) goal coordinates
            obstacles: List[Obstacle] for repair
            
        Returns:
            dict with 'raw', 'repaired', 'smoothed', 'time'
        """
        t_start = time.perf_counter()
        
        # 1. Preprocess Input
        # Add Batch Dim: (1, 1, 100, 100, 100)
        grid_tensor = torch.FloatTensor(grid).unsqueeze(0).unsqueeze(0).to(self.device)
        start_tensor = torch.FloatTensor(start).unsqueeze(0).to(self.device)
        goal_tensor = torch.FloatTensor(goal).unsqueeze(0).to(self.device)
        
        # 2. Model Inference
        with torch.no_grad():
            raw_path = self.model(grid_tensor, start_tensor, goal_tensor)
            
        raw_path = raw_path.cpu().numpy()[0] # (20, 3)
        
        # Force Start/Goal exactness on raw output (Model can be noisy)
        raw_path[0] = start
        raw_path[-1] = goal
        
        t_model = time.perf_counter()
        
        # 3. Physics-Based Repair
        repaired_path = self.post_proc.repair_trajectory(raw_path, obstacles)
        
        t_repair = time.perf_counter()
        
        # 4. Smoothing
        final_path = self.post_proc.smooth_trajectory(repaired_path)
        
        t_end = time.perf_counter()
        
        return {
            "path": final_path,
            "raw_path": raw_path,
            "repaired_path": repaired_path,
            "metrics": {
                "total_time_ms": (t_end - t_start) * 1000,
                "model_time_ms": (t_model - t_start) * 1000,
                "repair_time_ms": (t_repair - t_model) * 1000,
                "smooth_time_ms": (t_end - t_repair) * 1000
            }
        }
