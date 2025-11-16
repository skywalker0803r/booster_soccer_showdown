"""
Reward Shaping Wrapper to Fix the Step Penalty Problem
This addresses the core issue where step penalty (-1.0) makes early termination more attractive
"""

import numpy as np


class SmartRewardShaper:
    """
    Intelligent reward shaping to fix the step penalty issue
    """
    
    def __init__(self, step_penalty_scale=0.01):
        self.step_penalty_scale = step_penalty_scale
        self.prev_ball_dist = None
        self.prev_robot_height = None
        self.consecutive_fall_steps = 0
        self.episode_length = 0
        
    def shape_reward(self, obs, info, original_reward, terminated, truncated):
        """
        Smart reward shaping that addresses the step penalty problem
        
        Strategy:
        1. Drastically reduce step penalty
        2. Add heavy penalty for early termination due to falling
        3. Add progressive rewards for soccer fundamentals
        4. Ensure total episode reward can be positive
        """
        self.episode_length += 1
        shaped_reward = original_reward
        
        # Extract key information
        ball_pos = info.get("ball_xpos_rel_robot", np.zeros(3))
        if len(ball_pos.shape) > 1:
            ball_pos = ball_pos[0]
        ball_dist = np.linalg.norm(ball_pos)
        
        robot_quat = info.get("robot_quat", np.array([0, 0, 0, 1]))
        if len(robot_quat.shape) > 1:
            robot_quat = robot_quat[0]
        robot_height = robot_quat[2]  # Z component indicates tilt
        robot_upright = 1.0 - abs(robot_height)  # 1.0 = perfectly upright
        
        goal_pos = info.get("goal_team_1_rel_robot", np.zeros(3))
        if len(goal_pos.shape) > 1:
            goal_pos = goal_pos[0]
        
        ball_vel = info.get("ball_velp_rel_robot", np.zeros(3))
        if len(ball_vel.shape) > 1:
            ball_vel = ball_vel[0]
        
        robot_vel = info.get("robot_velocimeter", np.zeros(3))
        if len(robot_vel.shape) > 1:
            robot_vel = robot_vel[0]
        
        # ===== FIX THE CORE PROBLEM: STEP PENALTY =====
        
        # 1. DRAMATICALLY REDUCE STEP PENALTY
        # Instead of -1.0 per step, use much smaller penalty
        step_reward = -self.step_penalty_scale  # Default: -0.01 instead of -1.0
        shaped_reward += step_reward
        
        # 2. HEAVY PENALTY FOR EARLY TERMINATION DUE TO FALLING
        if terminated and self.episode_length < 50:  # Very early termination
            if robot_upright < 0.5:  # Likely fell down
                early_termination_penalty = -10.0  # Make early falling very expensive
                shaped_reward += early_termination_penalty
                print(f"WARNING: Early termination penalty applied: {early_termination_penalty}")
        
        # ===== POSITIVE REWARD SHAPING FOR SOCCER FUNDAMENTALS =====
        
        # 3. STABILITY REWARD (Fundamental: Stay upright)
        if robot_upright > 0.9:  # Very stable
            stability_reward = 0.1
            shaped_reward += stability_reward
        elif robot_upright > 0.7:  # Moderately stable
            stability_reward = 0.05
            shaped_reward += stability_reward
        elif robot_upright < 0.5:  # Falling/unstable
            # Track consecutive unstable steps
            self.consecutive_fall_steps += 1
            if self.consecutive_fall_steps > 5:  # Sustained falling
                fall_penalty = -0.2
                shaped_reward += fall_penalty
        else:
            self.consecutive_fall_steps = 0
        
        # 4. BALL APPROACH REWARD (Fundamental: Get close to ball)
        if self.prev_ball_dist is not None and robot_upright > 0.7:
            ball_progress = self.prev_ball_dist - ball_dist
            if ball_progress > 0.05:  # Significant approach
                approach_reward = min(ball_progress * 2.0, 0.5)  # Cap at 0.5
                shaped_reward += approach_reward
            elif ball_progress < -0.1:  # Moving away from ball
                retreat_penalty = -0.1
                shaped_reward += retreat_penalty
        self.prev_ball_dist = ball_dist
        
        # 5. BALL PROXIMITY BONUS (Reward being near the ball)
        if robot_upright > 0.8:  # Only if stable
            if ball_dist < 0.5:  # Very close
                proximity_reward = 0.3
                shaped_reward += proximity_reward
            elif ball_dist < 1.0:  # Close
                proximity_reward = 0.15
                shaped_reward += proximity_reward
            elif ball_dist < 2.0:  # Moderately close
                proximity_reward = 0.05
                shaped_reward += proximity_reward
        
        # 6. BALL MOVEMENT TOWARD GOAL REWARD
        ball_speed = np.linalg.norm(ball_vel)
        if ball_speed > 0.1:  # Ball is moving
            goal_direction = goal_pos / (np.linalg.norm(goal_pos) + 1e-8)
            ball_direction = ball_vel / (ball_speed + 1e-8)
            goal_alignment = np.dot(goal_direction, ball_direction)
            
            if goal_alignment > 0.7:  # Strong alignment toward goal
                goal_reward = 0.5 * ball_speed  # Reward proportional to speed
                shaped_reward += goal_reward
            elif goal_alignment > 0.3:  # Moderate alignment
                goal_reward = 0.2 * ball_speed
                shaped_reward += goal_reward
        
        # 7. CONTROLLED MOVEMENT REWARD
        movement_speed = np.linalg.norm(robot_vel)
        if robot_upright > 0.8:  # Only if stable
            if 0.1 < movement_speed < 2.0:  # Good controlled movement
                movement_reward = 0.02
                shaped_reward += movement_reward
            elif movement_speed > 4.0:  # Too frantic movement
                movement_penalty = -0.05
                shaped_reward += movement_penalty
        
        # 8. SUCCESS AMPLIFICATION
        if original_reward > 0:  # Any positive original reward
            success_bonus = min(original_reward * 0.5, 2.0)  # Amplify but cap
            shaped_reward += success_bonus
        
        # 9. EXPLORATION BONUS (Encourage trying new things)
        if self.episode_length < 100:  # Early in episode
            exploration_bonus = 0.01
            shaped_reward += exploration_bonus
        
        # 10. PREVENT REWARD HACKING
        # Ensure shaped rewards don't overwhelm original rewards
        shaped_portion = shaped_reward - original_reward
        shaped_portion = np.clip(shaped_portion, -5.0, 5.0)
        final_reward = original_reward + shaped_portion
        
        return final_reward
    
    def reset(self):
        """Reset for new episode"""
        self.prev_ball_dist = None
        self.prev_robot_height = None
        self.consecutive_fall_steps = 0
        self.episode_length = 0


class FixedRewardPreprocessor:
    """
    Enhanced Preprocessor that fixes the reward function while maintaining state processing
    """
    
    def __init__(self, original_preprocessor_class):
        self.original_preprocessor = original_preprocessor_class()
        self.reward_shaper = SmartRewardShaper(step_penalty_scale=0.01)  # Much smaller step penalty
    
    def modify_state(self, obs, info):
        """Use original state processing"""
        return self.original_preprocessor.modify_state(obs, info)
    
    def shape_reward(self, obs, info, reward, terminated=False, truncated=False):
        """Apply intelligent reward shaping"""
        return self.reward_shaper.shape_reward(obs, info, reward, terminated, truncated)
    
    def reset_episode(self):
        """Reset for new episode"""
        self.reward_shaper.reset()
    
    def quat_rotate_inverse(self, q: np.ndarray, v: np.ndarray):
        """Delegate to original preprocessor if it exists"""
        if hasattr(self.original_preprocessor, 'quat_rotate_inverse'):
            return self.original_preprocessor.quat_rotate_inverse(q, v)
        else:
            # Fallback implementation
            q_w = q[:,[-1]]
            q_vec = q[:,:3]
            a = v * (2.0 * q_w**2 - 1.0)
            b = np.cross(q_vec, v) * (q_w * 2.0)
            c = q_vec * (np.dot(q_vec, v).reshape(-1,1) * 2.0)    
            return a - b + c 

    def get_task_onehot(self, info):
        """Delegate to original preprocessor if it exists"""
        if hasattr(self.original_preprocessor, 'get_task_onehot'):
            return self.original_preprocessor.get_task_onehot(info)
        else:
            # Fallback
            if 'task_index' in info:
                return info['task_index']
            else:
                return np.array([])


def create_fixed_reward_environment_wrapper(env, preprocessor_class):
    """
    Create a wrapper that fixes the reward function during training
    """
    class FixedRewardEnvironment:
        def __init__(self, env, preprocessor_class):
            self.env = env
            self.preprocessor = FixedRewardPreprocessor(preprocessor_class)
            self.episode_step = 0
            
        def reset(self):
            self.episode_step = 0
            self.preprocessor.reset_episode()
            return self.env.reset()
        
        def step(self, action):
            obs, reward, terminated, truncated, info = self.env.step(action)
            
            # Apply reward shaping to fix the step penalty problem
            shaped_reward = self.preprocessor.shape_reward(
                obs, info, reward, terminated, truncated
            )
            
            # Debug logging for first few steps
            if self.episode_step < 10:
                print(f"Step {self.episode_step}: Original reward: {reward:.3f}, Shaped reward: {shaped_reward:.3f}")
            
            self.episode_step += 1
            
            return obs, shaped_reward, terminated, truncated, info
        
        def __getattr__(self, name):
            # Delegate other attributes to the original environment
            return getattr(self.env, name)
    
    return FixedRewardEnvironment(env, preprocessor_class)


# Quick test function
def test_reward_shaping():
    """Test the reward shaping logic"""
    print("Testing Smart Reward Shaper...")
    
    shaper = SmartRewardShaper(step_penalty_scale=0.01)
    
    # Simulate different scenarios
    scenarios = [
        {
            'name': 'Early fall scenario',
            'obs': np.random.randn(89),
            'info': {
                'ball_xpos_rel_robot': np.array([2.0, 0, 0]),
                'robot_quat': np.array([0, 0, 0.8, 0.6]),  # Fallen
                'goal_team_1_rel_robot': np.array([5.0, 0, 0]),
                'ball_velp_rel_robot': np.array([0, 0, 0]),
                'robot_velocimeter': np.array([0, 0, 0])
            },
            'original_reward': -1.5,  # Robot fallen
            'terminated': True
        },
        {
            'name': 'Stable approach scenario',
            'obs': np.random.randn(89),
            'info': {
                'ball_xpos_rel_robot': np.array([1.0, 0, 0]),
                'robot_quat': np.array([0, 0, 0.1, 0.995]),  # Upright
                'goal_team_1_rel_robot': np.array([5.0, 0, 0]),
                'ball_velp_rel_robot': np.array([0, 0, 0]),
                'robot_velocimeter': np.array([0.5, 0, 0])
            },
            'original_reward': -1.0,  # Just step penalty
            'terminated': False
        }
    ]
    
    for scenario in scenarios:
        shaped_reward = shaper.shape_reward(
            scenario['obs'],
            scenario['info'], 
            scenario['original_reward'],
            scenario['terminated'],
            False
        )
        
        print(f"\n{scenario['name']}:")
        print(f"  Original reward: {scenario['original_reward']:.3f}")
        print(f"  Shaped reward: {shaped_reward:.3f}")
        print(f"  Change: {shaped_reward - scenario['original_reward']:+.3f}")
        
        shaper.reset()  # Reset for next test


if __name__ == "__main__":
    test_reward_shaping()