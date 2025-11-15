"""
Aligned Reward Shaping for Soccer Robot
Based on official evaluation criteria to ensure positive correlation
"""
import numpy as np


class AlignedSoccerRewardShaper:
    """
    Conservative reward shaping aligned with official evaluation metrics
    """
    
    def __init__(self):
        self.prev_ball_dist = None
        self.step_count = 0
    
    def shape_reward(self, obs, info, original_reward, terminated):
        """
        Conservative shaping aligned with official rewards:
        
        Official positive rewards:
        - goal_scored: +2.5
        - ball_vel_twd_goal: +1.5  
        - robot_distance_ball: +0.25
        - success: +2.0
        - distance: +0.5
        
        Official penalties:
        - steps: -1.0/-0.3
        - robot_fallen: -1.5
        - offside: -3.0
        - ball_hits: -0.2
        """
        shaped_reward = 0.0
        
        # Extract key information
        ball_pos = info.get("ball_xpos_rel_robot", np.zeros((1, 3)))
        if len(ball_pos.shape) > 1:
            ball_pos = ball_pos[0]
        ball_dist = np.linalg.norm(ball_pos)
        
        goal_pos = info.get("goal_team_1_rel_robot", np.zeros((1, 3)))
        if len(goal_pos.shape) > 1:
            goal_pos = goal_pos[0]
        
        robot_quat = info.get("robot_quat", np.array([[0, 0, 0, 1]]))
        robot_upright = 1.0 - abs(robot_quat[0][2])  # 1.0 = perfectly upright
        
        ball_vel = info.get("ball_velp_rel_robot", np.zeros((1, 3)))
        if len(ball_vel.shape) > 1:
            ball_vel = ball_vel[0]
        
        # 1. STABILITY: Prevent robot_fallen penalty (-1.5)
        if robot_upright > 0.9:  # Very stable
            shaped_reward += 0.005
        elif robot_upright < 0.5:  # Unstable/falling
            shaped_reward -= 0.02  # Mild penalty to prevent falling
            
        # 2. BALL PROXIMITY: Support robot_distance_ball (+0.25)
        # Only reward if robot is stable
        if robot_upright > 0.8:
            if ball_dist < 1.0:
                shaped_reward += 0.01  # Close to ball
            elif ball_dist > 5.0:
                shaped_reward -= 0.005  # Too far from action
                
        # 3. BALL TOWARD GOAL: Support ball_vel_twd_goal (+1.5)
        ball_speed = np.linalg.norm(ball_vel)
        if ball_speed > 0.1:  # Ball is moving
            goal_direction = goal_pos / (np.linalg.norm(goal_pos) + 1e-8)
            ball_direction = ball_vel / (ball_speed + 1e-8)
            goal_alignment = np.dot(goal_direction, ball_direction)
            
            if goal_alignment > 0.7:  # Strong alignment toward goal
                shaped_reward += 0.015
            elif goal_alignment > 0.3:  # Moderate alignment
                shaped_reward += 0.005
                
        # 4. PROGRESS TRACKING: Reward getting closer to ball
        if self.prev_ball_dist is not None and robot_upright > 0.8:
            ball_progress = self.prev_ball_dist - ball_dist
            if ball_progress > 0.1:  # Significant approach
                shaped_reward += 0.003
            elif ball_progress < -0.2:  # Moving away
                shaped_reward -= 0.002
        self.prev_ball_dist = ball_dist
        
        # 5. MINIMIZE STEP PENALTY: Reward efficient movement
        robot_vel = info.get("robot_velocimeter", np.zeros((1, 3)))
        if len(robot_vel.shape) > 1:
            robot_vel = robot_vel[0]
        movement_speed = np.linalg.norm(robot_vel)
        
        # Reward controlled movement, penalize excessive motion
        if 0.1 < movement_speed < 1.5:  # Good controlled movement
            shaped_reward += 0.002
        elif movement_speed > 3.0:  # Too frantic
            shaped_reward -= 0.005
            
        # 6. SUCCESS AMPLIFICATION: Boost any positive original reward
        if original_reward > 0:
            shaped_reward += 0.01
            
        # 7. CONSERVATIVE CLIPPING: Keep shaping minimal
        shaped_reward = np.clip(shaped_reward, -0.05, 0.03)
        
        self.step_count += 1
        return original_reward + shaped_reward
    
    def reset(self):
        """Reset for new episode"""
        self.prev_ball_dist = None
        self.step_count = 0


def aligned_enhanced_preprocessor(original_preprocessor_class):
    """
    Create enhanced preprocessor with aligned reward shaping
    """
    class AlignedEnhancedPreprocessor(original_preprocessor_class):
        def __init__(self):
            super().__init__()
            self.reward_shaper = AlignedSoccerRewardShaper()
        
        def modify_state(self, obs, info):
            return super().modify_state(obs, info)
        
        def shape_reward(self, obs, info, reward, terminated):
            return self.reward_shaper.shape_reward(obs, info, reward, terminated)
        
        def reset_episode(self):
            self.reward_shaper.reset()
    
    return AlignedEnhancedPreprocessor