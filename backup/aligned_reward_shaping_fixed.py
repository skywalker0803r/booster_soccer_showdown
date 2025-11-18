"""
Aligned Soccer Reward Shaper
Based on official evaluation criteria to guide robot learning
"""

import numpy as np


class AlignedSoccerRewardShaper:
    """
    Reward shaper aligned with official evaluation criteria
    """
    
    def __init__(self):
        self.prev_ball_dist = None
        self.step_count = 0
    
    def shape_reward(self, obs, info, original_reward, terminated):
        """
        Apply conservative reward shaping aligned with official criteria
        """
        shaped_reward = 0.0
        self.step_count += 1
        
        # Extract key information
        robot_quat = info.get("robot_quat", np.array([[0, 0, 0, 1]]))
        robot_vel = info.get("robot_velocimeter", np.array([[0, 0, 0]]))
        ball_pos = info.get("ball_xpos_rel_robot", np.array([[0, 0, 0]]))
        ball_vel = info.get("ball_velp_rel_robot", np.array([[0, 0, 0]]))
        goal_pos = info.get("goal_team_1_rel_robot", np.array([[0, 0, 0]]))
        
        # Handle shape consistency
        if len(robot_quat.shape) > 1:
            robot_quat = robot_quat[0]
        if len(robot_vel.shape) > 1:
            robot_vel = robot_vel[0]
        if len(ball_pos.shape) > 1:
            ball_pos = ball_pos[0]
        if len(ball_vel.shape) > 1:
            ball_vel = ball_vel[0]
        if len(goal_pos.shape) > 1:
            goal_pos = goal_pos[0]
        
        # 1. STABILITY REWARD: Support robot_fallen penalty avoidance
        robot_upright = 1.0 - abs(robot_quat[2])  # 1.0 = perfectly upright
        
        if robot_upright > 0.9:
            shaped_reward += 0.005  # Small reward for stability
        elif robot_upright < 0.5:
            shaped_reward -= 0.02   # Penalty for instability
        
        # 2. BALL APPROACH REWARD: Support robot_distance_ball (+0.25)
        ball_dist = np.linalg.norm(ball_pos)
        
        if robot_upright > 0.8:  # Only reward approach when stable
            if ball_dist < 1.0:
                shaped_reward += 0.01
            elif ball_dist > 5.0:
                shaped_reward -= 0.005  # Too far from action
        
        # 3. BALL TOWARD GOAL REWARD: Support ball_vel_twd_goal (+1.5)
        ball_speed = np.linalg.norm(ball_vel)
        if ball_speed > 0.1:  # Ball is moving
            goal_direction = goal_pos / (np.linalg.norm(goal_pos) + 1e-8)
            ball_direction = ball_vel / ball_speed
            goal_alignment = np.dot(goal_direction, ball_direction)
            
            if goal_alignment > 0.7:
                shaped_reward += 0.015
            elif goal_alignment > 0.3:
                shaped_reward += 0.005
        
        # 4. BALL APPROACH PROGRESS: Track progress toward ball
        if self.prev_ball_dist is not None and robot_upright > 0.8:
            ball_progress = self.prev_ball_dist - ball_dist
            if ball_progress > 0.1:
                shaped_reward += 0.003
            elif ball_progress < -0.2:
                shaped_reward -= 0.002
        
        self.prev_ball_dist = ball_dist
        
        # 5. MOVEMENT EFFICIENCY: Minimize steps penalty
        movement_speed = np.linalg.norm(robot_vel)
        
        if 0.1 < movement_speed < 1.5:
            shaped_reward += 0.002  # Controlled movement
        elif movement_speed > 3.0:
            shaped_reward -= 0.005  # Too frantic
        
        # 6. BALL VELOCITY REWARD: Task-specific ball speed rewards
        shaped_reward = self._add_ball_velocity_reward(info, shaped_reward)
        
        # 7. SUCCESS AMPLIFICATION: Boost any positive original reward
        if original_reward > 0:
            shaped_reward += 0.01
            
        # 8. CONSERVATIVE CLIPPING: Keep shaping minimal
        shaped_reward = np.clip(shaped_reward, -0.05, 0.05)
        
        return original_reward + shaped_reward
    
    def _add_ball_velocity_reward(self, info, shaped_reward):
        """
        Task-specific ball velocity rewards based on direction and speed
        """
        ball_vel = info.get("ball_velp_rel_robot", np.zeros((1, 3)))
        if len(ball_vel.shape) > 1:
            ball_vel = ball_vel[0]
        
        ball_speed = np.linalg.norm(ball_vel)
        
        # Skip if ball is not moving
        if ball_speed < 0.1:
            return shaped_reward
        
        # Get task-relevant positions
        goal_pos = info.get("goal_team_1_rel_robot", np.zeros((1, 3)))
        if len(goal_pos.shape) > 1:
            goal_pos = goal_pos[0]
            
        target_pos = info.get("target_xpos_rel_robot", np.zeros((1, 3)))
        if len(target_pos.shape) > 1:
            target_pos = target_pos[0]
        
        ball_direction = ball_vel / ball_speed
        
        # Determine task type and apply appropriate velocity rewards
        if np.linalg.norm(target_pos) > 0.1:  # Task 3: Precision Pass
            target_direction = target_pos / (np.linalg.norm(target_pos) + 1e-8)
            target_alignment = np.dot(target_direction, ball_direction)
            
            if target_alignment > 0.8 and 1.0 < ball_speed < 5.0:
                # Perfect direction, optimal speed for passing
                shaped_reward += 0.02 * min(ball_speed, 4.0)
            elif target_alignment > 0.6 and ball_speed < 3.0:
                # Good direction, acceptable speed
                shaped_reward += 0.01 * ball_speed
            elif target_alignment < -0.3 and ball_speed > 2.0:
                # Wrong direction, high speed - penalty
                shaped_reward -= 0.01
                
        else:  # Task 1 & 2: Penalty Kicks
            goal_direction = goal_pos / (np.linalg.norm(goal_pos) + 1e-8)
            goal_alignment = np.dot(goal_direction, ball_direction)
            
            if goal_alignment > 0.8:  # Strong alignment toward goal
                if ball_speed > 3.0:
                    # High speed toward goal - excellent
                    shaped_reward += 0.025 * min(ball_speed, 8.0)
                elif ball_speed > 1.0:
                    # Medium speed toward goal - good
                    shaped_reward += 0.015 * ball_speed
                else:
                    # Low speed toward goal - okay
                    shaped_reward += 0.01 * ball_speed
                    
            elif goal_alignment > 0.5:  # Moderate alignment
                shaped_reward += 0.008 * min(ball_speed, 5.0)
                
            elif goal_alignment < -0.3 and ball_speed > 2.0:
                # High speed away from goal - penalty
                shaped_reward -= 0.015
        
        return shaped_reward

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