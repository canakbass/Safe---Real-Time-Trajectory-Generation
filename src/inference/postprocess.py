
import numpy as np
from scipy.interpolate import make_interp_spline

class PotentialFieldRepair:
    """
    Physics-Based Repair using Artificial Potential Fields (APF).
    Moves waypoints out of obstacles by simulating repulsive forces.
    """
    def __init__(self, bounds=None):
        self.bounds = bounds if bounds is not None else np.array([1000, 1000, 1000])

    def repair(self, path: np.ndarray, obstacles: list, iterations: int = 10, margin: float = 20.0) -> np.ndarray:
        repaired = path.copy()
        
        for k in range(iterations):
            is_collision = False
            for i in range(1, len(repaired) - 1): # Fix start/end
                point = repaired[i]
                move_vec = np.zeros(3)
                count = 0
                
                for obs in obstacles:
                    diff = point - obs.center
                    dist = np.linalg.norm(diff)
                    min_dist = obs.radius + margin
                    
                    if dist < min_dist:
                        is_collision = True
                        if dist < 1e-6:
                            direction = np.random.randn(3)
                            direction /= np.linalg.norm(direction)
                        else:
                            direction = diff / dist
                        
                        push = (min_dist - dist) * 1.1 # Push out
                        move_vec += direction * push
                        count += 1
                
                if count > 0:
                    repaired[i] += move_vec
                    # Clamp
                    repaired[i] = np.maximum([0,0,0], np.minimum(self.bounds, repaired[i]))
            
            if not is_collision:
                break
        return repaired

class BezierSmoother:
    """
    Kinodynamic Smoothing using B-Splines (generalization of Bezier).
    """
    def smooth(self, path: np.ndarray, num_points: int = 50, degree: int = 3) -> np.ndarray:
        if len(path) < degree + 1:
            return path
            
        # Parameterize by chord length
        dists = np.linalg.norm(path[1:] - path[:-1], axis=1)
        t = np.insert(np.cumsum(dists), 0, 0)
        if t[-1] == 0: return path
        t /= t[-1]
        
        try:
            bspline = make_interp_spline(t, path, k=degree)
            t_new = np.linspace(0, 1, num_points)
            smoothed = bspline(t_new)
            
            # Pin endpoints
            smoothed[0] = path[0]
            smoothed[-1] = path[-1]
            return smoothed
        except:
            return path

class PostProcessor:
    """Wrapper for backward compatibility."""
    def __init__(self, x_range=1000, y_range=1000, z_range=1000):
        self.repair_module = PotentialFieldRepair(bounds=np.array([x_range, y_range, z_range]))
        self.smooth_module = BezierSmoother()
        
    def repair_trajectory(self, path, obstacles, **kwargs):
        return self.repair_module.repair(path, obstacles, **kwargs)
        
    def smooth_trajectory(self, path, **kwargs):
        return self.smooth_module.smooth(path, **kwargs)
