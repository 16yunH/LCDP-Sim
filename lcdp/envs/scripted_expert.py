"""
Scripted expert for data collection in simulation.
Provides baseline demonstrations for imitation learning.
"""

import numpy as np
from typing import Dict, Tuple, Optional


class ScriptedExpert:
    """
    Simple scripted expert for robot manipulation tasks.
    Uses privileged information (object positions) to compute actions.
    """
    
    def __init__(
        self,
        task: str = "pick",
        noise_scale: float = 0.01,
        move_speed: float = 0.05
    ):
        """
        Args:
            task: Task type ("pick", "push", "stack")
            noise_scale: Scale of noise to add to actions
            move_speed: Movement speed (meters per step)
        """
        self.task = task
        self.noise_scale = noise_scale
        self.move_speed = move_speed
        self.state = "approach"  # State machine: approach, grasp, lift, move, release
        self.target_pos = None
    
    def compute_action(
        self,
        ee_pos: np.ndarray,
        ee_quat: np.ndarray,
        obj_pos: np.ndarray,
        gripper_state: float,
        target_pos: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Compute action based on current state.
        
        Args:
            ee_pos: End-effector position [x, y, z]
            ee_quat: End-effector quaternion [w, x, y, z]
            obj_pos: Object position [x, y, z]
            gripper_state: Current gripper state (0=open, 1=closed)
            target_pos: Target position for object (optional)
        Returns:
            Action [dx, dy, dz, droll, dpitch, dyaw, gripper_action]
        """
        action = np.zeros(7)
        
        if self.task == "pick":
            action = self._pick_policy(ee_pos, obj_pos, gripper_state)
        elif self.task == "push":
            action = self._push_policy(ee_pos, obj_pos, target_pos)
        elif self.task == "stack":
            action = self._stack_policy(ee_pos, obj_pos, gripper_state, target_pos)
        
        # Add small noise for robustness
        if self.noise_scale > 0:
            action[:6] += np.random.randn(6) * self.noise_scale
        
        return action
    
    def _pick_policy(
        self,
        ee_pos: np.ndarray,
        obj_pos: np.ndarray,
        gripper_state: float
    ) -> np.ndarray:
        """
        Simple pick policy: approach -> grasp -> lift
        """
        action = np.zeros(7)
        
        # Compute distance to object
        dist = np.linalg.norm(ee_pos[:2] - obj_pos[:2])
        height_diff = ee_pos[2] - obj_pos[2]
        
        if self.state == "approach":
            # Move towards object (xy plane)
            if dist > 0.02:
                direction = (obj_pos[:2] - ee_pos[:2]) / (dist + 1e-6)
                action[0] = direction[0] * self.move_speed
                action[1] = direction[1] * self.move_speed
            
            # Lower to grasp height
            if height_diff > 0.05:
                action[2] = -self.move_speed
            
            # Keep gripper open
            action[6] = -1.0
            
            # Transition to grasp when close enough
            if dist < 0.02 and abs(height_diff) < 0.05:
                self.state = "grasp"
        
        elif self.state == "grasp":
            # Close gripper
            action[6] = 1.0
            
            # Wait for gripper to close
            if gripper_state > 0.8:
                self.state = "lift"
        
        elif self.state == "lift":
            # Lift object
            action[2] = self.move_speed
            action[6] = 1.0  # Keep gripper closed
        
        return action
    
    def _push_policy(
        self,
        ee_pos: np.ndarray,
        obj_pos: np.ndarray,
        target_pos: Optional[np.ndarray]
    ) -> np.ndarray:
        """
        Simple push policy: approach from behind -> push towards target
        """
        action = np.zeros(7)
        
        if target_pos is None:
            target_pos = obj_pos + np.array([0.1, 0.0, 0.0])
        
        # Compute push direction
        push_dir = (target_pos[:2] - obj_pos[:2])
        push_dir = push_dir / (np.linalg.norm(push_dir) + 1e-6)
        
        # Approach from opposite side
        pre_push_pos = obj_pos[:2] - push_dir * 0.05
        
        if self.state == "approach":
            dist = np.linalg.norm(ee_pos[:2] - pre_push_pos)
            
            if dist > 0.02:
                direction = (pre_push_pos - ee_pos[:2]) / (dist + 1e-6)
                action[0] = direction[0] * self.move_speed
                action[1] = direction[1] * self.move_speed
            else:
                self.state = "push"
            
            # Keep at table height
            action[2] = -self.move_speed if ee_pos[2] > obj_pos[2] + 0.02 else 0
            action[6] = -1.0  # Open gripper
        
        elif self.state == "push":
            # Push towards target
            action[0] = push_dir[0] * self.move_speed
            action[1] = push_dir[1] * self.move_speed
            action[6] = -1.0
        
        return action
    
    def _stack_policy(
        self,
        ee_pos: np.ndarray,
        obj_pos: np.ndarray,
        gripper_state: float,
        target_pos: Optional[np.ndarray]
    ) -> np.ndarray:
        """
        Stacking policy: pick first object -> move -> place on second
        """
        # Similar to pick, but with target position
        if target_pos is None:
            return self._pick_policy(ee_pos, obj_pos, gripper_state)
        
        action = np.zeros(7)
        
        if self.state == "approach":
            # First pick the object
            return self._pick_policy(ee_pos, obj_pos, gripper_state)
        
        elif self.state == "lift":
            # After lifting, move to target
            if ee_pos[2] > obj_pos[2] + 0.1:
                self.state = "move"
        
        elif self.state == "move":
            # Move towards target position
            dist = np.linalg.norm(ee_pos[:2] - target_pos[:2])
            
            if dist > 0.02:
                direction = (target_pos[:2] - ee_pos[:2]) / (dist + 1e-6)
                action[0] = direction[0] * self.move_speed
                action[1] = direction[1] * self.move_speed
            else:
                self.state = "release"
            
            action[6] = 1.0  # Keep gripper closed
        
        elif self.state == "release":
            # Open gripper to release
            action[6] = -1.0
        
        return action
    
    def reset(self):
        """Reset expert state."""
        self.state = "approach"
        self.target_pos = None


if __name__ == "__main__":
    # Test scripted expert
    expert = ScriptedExpert(task="pick")
    
    # Simulate pick task
    ee_pos = np.array([0.0, 0.0, 0.5])
    obj_pos = np.array([0.2, 0.1, 0.0])
    gripper_state = 0.0
    
    print("Testing pick policy...")
    for step in range(20):
        action = expert.compute_action(ee_pos, None, obj_pos, gripper_state)
        print(f"Step {step}, State: {expert.state}, Action: {action}")
        
        # Simulate action execution
        ee_pos += action[:3]
        if action[6] > 0:
            gripper_state = min(1.0, gripper_state + 0.1)
        else:
            gripper_state = max(0.0, gripper_state - 0.1)
        
        if expert.state == "lift" and ee_pos[2] > 0.3:
            print("Pick completed!")
            break
