"""
Episode-Level Reward Wrapper
Fixes the step penalty at episode end, not at each step
"""

import numpy as np


class EpisodeLevelRewardFixer:
    """
    Fixes rewards at episode level where SAI actually applies them
    """
    
    def __init__(self, step_penalty_replacement=-0.01):
        self.step_penalty_replacement = step_penalty_replacement
        self.episode_step_count = 0
        self.accumulated_shaped_rewards = 0
        
    def reset_episode(self):
        """Reset for new episode"""
        self.episode_step_count = 0
        self.accumulated_shaped_rewards = 0
        
    def add_step_reward(self, obs, info):
        """Add positive step rewards during episode"""
        self.episode_step_count += 1
        step_reward = 0
        
        # Extract key information
        ball_pos = info.get("ball_xpos_rel_robot", np.zeros(3))
        if len(ball_pos.shape) > 1:
            ball_pos = ball_pos[0]
        ball_dist = np.linalg.norm(ball_pos)
        
        robot_quat = info.get("robot_quat", np.array([0, 0, 0, 1]))
        if len(robot_quat.shape) > 1:
            robot_quat = robot_quat[0]
        robot_height = robot_quat[2]
        robot_upright = 1.0 - abs(robot_height)
        
        robot_vel = info.get("robot_velocimeter", np.zeros(3))
        if len(robot_vel.shape) > 1:
            robot_vel = robot_vel[0]
        
        # Stability reward
        if robot_upright > 0.9:
            step_reward += 0.05  # Small reward for staying upright
        elif robot_upright > 0.7:
            step_reward += 0.02
        
        # Ball proximity reward
        if robot_upright > 0.8:
            if ball_dist < 0.5:
                step_reward += 0.1
            elif ball_dist < 1.0:
                step_reward += 0.05
            elif ball_dist < 2.0:
                step_reward += 0.02
        
        # Controlled movement reward
        movement_speed = np.linalg.norm(robot_vel)
        if robot_upright > 0.8 and 0.1 < movement_speed < 2.0:
            step_reward += 0.01
        
        self.accumulated_shaped_rewards += step_reward
        return step_reward
    
    def fix_episode_reward(self, episode_reward, terminated, truncated):
        """Fix the final episode reward"""
        if not (terminated or truncated):
            return episode_reward  # Not end of episode
        
        print(f"🔍 Episode End Analysis:")
        print(f"   Original episode reward: {episode_reward:.3f}")
        print(f"   Episode steps: {self.episode_step_count}")
        print(f"   Accumulated step rewards: {self.accumulated_shaped_rewards:.3f}")
        
        # Calculate what the step penalty should be
        # Original: step_penalty = -1.0 * steps = -190 for 190 steps
        # Fixed: step_penalty = -0.01 * steps = -1.9 for 190 steps
        original_step_penalty = -1.0 * self.episode_step_count
        fixed_step_penalty = self.step_penalty_replacement * self.episode_step_count
        step_penalty_improvement = original_step_penalty - fixed_step_penalty
        
        # The episode reward contains the original step penalty
        # Remove original step penalty and add our fixed version
        other_rewards = episode_reward - original_step_penalty  # Extract non-step rewards
        fixed_reward = other_rewards + fixed_step_penalty + self.accumulated_shaped_rewards
        
        print(f"   Calculated original step penalty: {original_step_penalty:.3f}")
        print(f"   Fixed step penalty: {fixed_step_penalty:.3f}")
        print(f"   Other rewards (ball, goal, etc.): {other_rewards:.3f}")
        print(f"   Final fixed reward: {fixed_reward:.3f}")
        print(f"   Improvement: {fixed_reward - episode_reward:+.3f}")
        
        return fixed_reward


class EpisodeRewardEnvironmentWrapper:
    """
    Environment wrapper that fixes rewards at episode level
    """
    
    def __init__(self, env):
        self.env = env
        self.reward_fixer = EpisodeLevelRewardFixer(step_penalty_replacement=-0.01)
        
    def reset(self, **kwargs):
        self.reward_fixer.reset_episode()
        return self.env.reset(**kwargs)
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Add step-level shaped rewards (these accumulate)
        step_shaped_reward = self.reward_fixer.add_step_reward(obs, info)
        
        # Fix final episode reward if episode ended
        if terminated or truncated:
            fixed_reward = self.reward_fixer.fix_episode_reward(reward, terminated, truncated)
            print(f"📊 Episode Complete: Original={reward:.3f} → Fixed={fixed_reward:.3f}")
        else:
            # During episode, return small step rewards instead of 0
            fixed_reward = step_shaped_reward  # Give immediate feedback
        
        return obs, fixed_reward, terminated, truncated, info
    
    def __getattr__(self, name):
        # Delegate other attributes to the original environment
        return getattr(self.env, name)


def test_episode_reward_fixing():
    """Test the episode reward fixing logic"""
    print("🧪 Testing Episode Reward Fixing")
    print("="*50)
    
    fixer = EpisodeLevelRewardFixer(step_penalty_replacement=-0.01)
    fixer.reset_episode()
    
    # Simulate episode
    for step in range(190):  # Typical episode length
        obs = np.random.randn(89)
        info = {
            'ball_xpos_rel_robot': np.array([2.0, 0, 0]),
            'robot_quat': np.array([0, 0, 0.1, 0.995]),  # Upright
            'goal_team_1_rel_robot': np.array([5.0, 0, 0]),
            'ball_velp_rel_robot': np.array([0, 0, 0]),
            'robot_velocimeter': np.array([0.5, 0, 0])
        }
        
        step_reward = fixer.add_step_reward(obs, info)
    
    # Test different episode ending scenarios
    scenarios = [
        (-30.828, "Typical episode like in log"),
        (-190.0, "Pure step penalty episode"),
        (-187.5, "Step penalty + robot fallen"),
        (-185.5, "Step penalty + some positive rewards")
    ]
    
    print(f"\n📊 Episode Ending Scenarios:")
    for episode_reward, description in scenarios:
        print(f"\n{description}:")
        fixed_reward = fixer.fix_episode_reward(episode_reward, True, False)
        improvement = fixed_reward - episode_reward
        print(f"   Improvement: {improvement:+.3f}")
        
        if improvement > 150:  # Expect ~189 improvement for 190 steps
            print("   ✅ Major improvement achieved!")
        elif improvement > 50:
            print("   ⚠️ Some improvement, but check logic")
        else:
            print("   ❌ Little improvement, something wrong")


if __name__ == "__main__":
    test_episode_reward_fixing()