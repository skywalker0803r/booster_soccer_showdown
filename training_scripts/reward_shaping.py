"""
Reward Shaping for Soccer Robot - Dense Reward Engineering
Based on the sparse reward solutions from the documentation
"""
import numpy as np


class SoccerRewardShaper:
    """
    Add dense intermediate rewards to help with sparse reward problem
    """
    
    def __init__(self):
        self.prev_ball_dist = None
        self.prev_goal_dist = None
        self.prev_robot_stability = None
    
    def shape_reward(self, obs, info, original_reward, terminated):
        """
        Add shaped rewards based on soccer fundamentals:
        1. Approaching the ball
        2. Ball moving toward goal  
        3. Robot stability (not falling)
        4. Exploration bonus
        """
        shaped_reward = 0.0
        
        # Extract useful information
        ball_pos = info.get("ball_xpos_rel_robot", np.zeros(3))
        ball_dist = np.linalg.norm(ball_pos)
        
        goal_pos = info.get("goal_team_1_rel_robot", np.zeros(3))  # Opponent goal
        goal_dist = np.linalg.norm(goal_pos)
        
        robot_quat = info.get("robot_quat", np.array([0, 0, 0, 1]))
        robot_stability = abs(robot_quat[2])  # Z-axis orientation, closer to 0 is more upright
        
        # 1. Reward for approaching the ball (fundamental soccer skill)
        if self.prev_ball_dist is not None:
            ball_approach_reward = (self.prev_ball_dist - ball_dist) * 0.1
            shaped_reward += ball_approach_reward
        self.prev_ball_dist = ball_dist
        
        # 2. Reward for staying upright (basic robot control)
        stability_reward = (1.0 - robot_stability) * 0.05  # Reward upright position
        shaped_reward += stability_reward
        
        # 3. Small penalty for being too far from action (encourage engagement)
        if ball_dist > 5.0:  # If very far from ball
            shaped_reward -= 0.02
            
        # 4. Reward for ball being close (proximity bonus)
        if ball_dist < 1.0:  # Very close to ball
            shaped_reward += 0.1
        elif ball_dist < 2.0:  # Close to ball
            shaped_reward += 0.05
            
        # 5. Goal-oriented behavior (if ball is moving toward goal)
        ball_vel = info.get("ball_velp_rel_robot", np.zeros(3))
        if np.linalg.norm(ball_vel) > 0.1:  # Ball is moving
            goal_direction = goal_pos / (np.linalg.norm(goal_pos) + 1e-8)
            ball_vel_normalized = ball_vel / (np.linalg.norm(ball_vel) + 1e-8)
            goal_alignment = np.dot(goal_direction, ball_vel_normalized)
            if goal_alignment > 0:  # Ball moving toward goal
                shaped_reward += goal_alignment * 0.2
        
        # 6. Exploration bonus for new positions
        robot_pos = info.get("robot_velocimeter", np.zeros(3))
        movement_bonus = min(np.linalg.norm(robot_pos) * 0.01, 0.05)  # Reward movement, cap at 0.05
        shaped_reward += movement_bonus
        
        # 7. Success amplification - if original reward is positive, amplify the good behavior
        if original_reward > 0:
            shaped_reward += original_reward * 0.1  # Small bonus for any positive reward
        
        # 8. Prevent reward hacking - cap the shaped reward
        shaped_reward = np.clip(shaped_reward, -0.5, 0.5)
        
        return original_reward + shaped_reward
    
    def reset(self):
        """Reset for new episode"""
        self.prev_ball_dist = None
        self.prev_goal_dist = None
        self.prev_robot_stability = None


def enhanced_preprocessor_with_shaping(original_preprocessor_class):
    """
    Wrapper to add reward shaping to existing preprocessor
    """
    class EnhancedPreprocessor(original_preprocessor_class):
        def __init__(self):
            super().__init__()
            self.reward_shaper = SoccerRewardShaper()
        
        def modify_state(self, obs, info):
            # Use original preprocessing
            return super().modify_state(obs, info)
        
        def shape_reward(self, obs, info, reward, terminated):
            # Add reward shaping
            return self.reward_shaper.shape_reward(obs, info, reward, terminated)
        
        def reset_episode(self):
            # Reset reward shaper for new episode
            self.reward_shaper.reset()
    
    return EnhancedPreprocessor