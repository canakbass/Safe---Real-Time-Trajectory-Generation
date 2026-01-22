
import numpy as np
import math
from typing import List, Tuple, Optional

class Node:
    def __init__(self, position):
        self.position = np.array(position, dtype=np.float32)
        self.cost = 0.0
        self.parent: Optional[Node] = None

class RRTStarPlanner:
    """
    Rapidly-exploring Random Tree Star (RRT*) for 3D Space.
    Optimized for continuous 3D environments with spherical obstacles.
    """
    def __init__(self, env, step_size=20.0, max_iter=2000, search_radius=50.0):
        self.env = env
        self.step_size = step_size
        self.max_iter = max_iter
        self.search_radius = search_radius
        
        self.start = None
        self.goal = None
        self.nodes = []

    def solve(self, start, goal, timeout_sec=5.0) -> Optional[np.ndarray]:
        """
        Attempts to find a path from start to goal.
        Returns: np.ndarray of shape (N, 3) or None if failed.
        """
        self.start = Node(start)
        self.goal = Node(goal)
        self.nodes = [self.start]
        
        # Simple timeout logic would go here, but for benchmark we use iterations
        for i in range(self.max_iter):
            # 1. Sample Random Point
            if np.random.rand() < 0.1: # Goal Bias
                rnd_point = self.goal.position
            else:
                rnd_point = np.random.uniform(
                    low=[0, 0, 0], 
                    high=[self.env.config.x_range, self.env.config.y_range, self.env.config.z_range]
                )
            
            # 2. Find Nearest
            nearest_node = self._get_nearest_node(rnd_point)
            
            # 3. Steer
            new_node = self._steer(nearest_node, rnd_point)
            
            # 4. Collision Check
            if self._check_collision_segment(nearest_node.position, new_node.position):
                continue
                
            # 5. Near Neighbors Search
            near_nodes = self._find_near_nodes(new_node)
            
            # 6. Choose Best Parent
            min_cost = nearest_node.cost + np.linalg.norm(new_node.position - nearest_node.position)
            new_node.parent = nearest_node
            new_node.cost = min_cost
            
            for near_node in near_nodes:
                cost = near_node.cost + np.linalg.norm(new_node.position - near_node.position)
                if cost < min_cost:
                    if not self._check_collision_segment(near_node.position, new_node.position):
                        min_cost = cost
                        new_node.parent = near_node
                        new_node.cost = cost
            
            self.nodes.append(new_node)
            
            # 7. Rewire
            for near_node in near_nodes:
                cost = new_node.cost + np.linalg.norm(new_node.position - near_node.position)
                if cost < near_node.cost:
                    if not self._check_collision_segment(new_node.position, near_node.position):
                        near_node.parent = new_node
                        near_node.cost = cost
                        
            # Check Goal Reached
            if np.linalg.norm(new_node.position - self.goal.position) < self.step_size:
                final_node = Node(self.goal.position)
                if not self._check_collision_segment(new_node.position, final_node.position):
                    final_node.parent = new_node
                    final_node.cost = new_node.cost + np.linalg.norm(new_node.position - final_node.position)
                    return self._generate_path(final_node)

        return None # Failed

    def _get_nearest_node(self, point):
        dists = [np.linalg.norm(node.position - point) for node in self.nodes]
        min_idx = np.argmin(dists)
        return self.nodes[min_idx]

    def _steer(self, from_node, to_point):
        direction = to_point - from_node.position
        dist = np.linalg.norm(direction)
        
        if dist > self.step_size:
            direction = direction / dist * self.step_size
            new_point = from_node.position + direction
        else:
            new_point = to_point
            
        return Node(new_point)

    def _check_collision_segment(self, start, end):
        """Discretized collision check along line."""
        dist = np.linalg.norm(end - start)
        steps = int(dist / (self.env.config.min_radius / 2.0)) + 1 # Conservative steps
        for i in range(steps + 1):
            t = i / steps
            pt = start + (end - start) * t
            if self.env.check_collision(pt, margin=5.0): # Margin for safety
                return True
        return False

    def _find_near_nodes(self, new_node):
        n_node = len(self.nodes) + 1
        r = 50.0 * math.sqrt((math.log(n_node) / n_node))
        r = min(r, self.search_radius)
        
        dists = [np.linalg.norm(node.position - new_node.position) for node in self.nodes]
        near_nodes = [self.nodes[i] for i in range(len(self.nodes)) if dists[i] < r]
        return near_nodes

    def _generate_path(self, end_node):
        path = []
        curr = end_node
        while curr is not None:
            path.append(curr.position)
            curr = curr.parent
        return np.array(path[::-1])
