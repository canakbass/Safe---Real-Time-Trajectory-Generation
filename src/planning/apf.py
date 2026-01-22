
import numpy as np
from typing import Optional

class PotentialFieldPlanner:
    """
    Artificial Potential Field (APF) Planner.
    Fast, reactive, but prone to local minima.
    """
    def __init__(self, env, step_size=5.0, max_iter=2000):
        self.env = env
        self.step_size = step_size
        self.max_iter = max_iter
        
        # Tuning Parameters
        self.k_att = 1.0  # Attraction gain
        self.k_rep = 1000000.0 # Repulsion gain (needs to be high for safety)
        self.d0 = 60.0    # Influence distance of obstacles

    def solve(self, start, goal, timeout_sec=2.0) -> Optional[np.ndarray]:
        """
        Gradient Descent on Potential Surface.
        """
        path = [start]
        curr = np.array(start, dtype=np.float32)
        goal = np.array(goal, dtype=np.float32)
        
        for i in range(self.max_iter):
            # 1. Attraction Force
            f_att = self._get_att_force(curr, goal)
            
            # 2. Repulsion Force
            f_rep = self._get_rep_force(curr)
            
            # 3. Total Force
            f_total = f_att + f_rep
            
            # 4. Step
            # Normalize force to step size (Gradient Descent with fixed step)
            norm = np.linalg.norm(f_total)
            if norm > 1e-3:
                step = (f_total / norm) * self.step_size
            else:
                step = np.zeros(3)
                
            curr = curr + step
            
            # Check Collision (If step puts us inside, we failed/crashed)
            # APF should theoretically avoid, but discrete steps might jump in
            if self.env.check_collision(curr):
                # Stuck in obstacle or oscillating
                # For APF, we usually consider this a failure of the 'Navigator'
                # But let's let it run, maybe it slides out?
                # Actually, if collision, let's stop.
                return None 
            
            path.append(curr.copy())
            
            # Check Goal
            if np.linalg.norm(curr - goal) < self.step_size * 2:
                path.append(goal)
                return np.array(path)
                
            # Local Minima Check (Oscillation)
            if len(path) > 30:
                # If we haven't moved far in last 20 steps
                past = path[-20]
                if np.linalg.norm(curr - past) < self.step_size:
                    # Stuck in Local Minima
                    return None # Fail
                    
        return None # Timeout

    def _get_att_force(self, curr, goal):
        """Attraction: Linear proportional to distance (Parabolic potential)."""
        return self.k_att * (goal - curr)

    def _get_rep_force(self, curr):
        """Repulsion: Inverse square law inside influence radius."""
        f_rep = np.zeros(3)
        for obs in self.env.obstacles:
            dist = np.linalg.norm(curr - obs.center) - obs.radius
            if dist <= self.d0:
                if dist <= 0.1: dist = 0.1 # Avoid singularity
                # Pot = 0.5 * k * (1/d - 1/d0)^2
                # Force = -Grad(Pot)
                # F = k * (1/d - 1/d0) * (1/d^2) * unit_vec_from_obs
                grad_mag = self.k_rep * (1.0/dist - 1.0/self.d0) * (1.0/(dist**2))
                unit_vec = (curr - obs.center) / np.linalg.norm(curr - obs.center)
                f_rep += grad_mag * unit_vec
        return f_rep
