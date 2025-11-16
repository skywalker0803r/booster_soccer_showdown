"""
Analyze Original Training Data
Based on new understanding of SAI reward system
"""

def analyze_original_training():
    """Analyze the original training to understand the real problem"""
    
    print("🔍 Analysis of Original Training Data")
    print("="*60)
    
    # From log.txt, we know:
    original_data = {
        'episode_783': {'reward': -26.987, 'steps': 154},
        'episode_784': {'reward': -50.692, 'steps': 239}, 
        'episode_785': {'reward': -16.439, 'steps': 182},
        'episode_786': {'reward': -19.950, 'steps': 191},
        'final_avg': -30.828,
        'max_reward': 25.418,  # From TensorBoard
        'min_reward': -123.774  # From TensorBoard
    }
    
    print("📊 Original Training Analysis:")
    print("-" * 30)
    
    for ep_name, data in original_data.items():
        if 'episode_' in ep_name:
            reward = data['reward']
            steps = data['steps']
            reward_per_step = reward / steps
            
            print(f"{ep_name}:")
            print(f"  Total reward: {reward:8.3f}")
            print(f"  Steps: {steps:3d}")
            print(f"  Reward/step: {reward_per_step:8.5f}")
            
            # Analyze what this could be
            if abs(reward_per_step + 1.0) < 0.1:
                print(f"  ✅ Looks like pure -1.0 step penalty")
            elif abs(reward_per_step + 0.3) < 0.1:
                print(f"  ✅ Looks like pure -0.3 step penalty (Task 3)")
            elif -0.5 < reward_per_step < -0.1:
                print(f"  🤔 Mixed penalties (step + other)")
            elif reward_per_step > 0:
                print(f"  🎉 Net positive reward!")
            else:
                print(f"  ⚠️ Unexpected pattern")
            
            print()
    
    print("📈 Key Insights:")
    print(f"  Best episode reward: +{original_data['max_reward']:.3f} (POSITIVE!)")
    print(f"  Worst episode reward: {original_data['min_reward']:.3f}")
    print(f"  Final average: {original_data['final_avg']:.3f}")
    
    # Calculate what step penalty reduction would do
    print(f"\n🔧 Step Penalty Reduction Impact:")
    print("-" * 40)
    
    avg_episode_length = 190  # From TensorBoard
    
    scenarios = [
        ("Current (mixed)", -30.828, "Actual training result"),
        ("Pure -1.0 step", -1.0 * avg_episode_length, "If pure step penalty"),
        ("Pure -0.3 step", -0.3 * avg_episode_length, "If Task 3 step penalty"),
        ("Reduced -0.01 step", -0.01 * avg_episode_length + 5, "Our fix + some positive")
    ]
    
    for name, reward, description in scenarios:
        print(f"{name:20s}: {reward:8.3f} ({description})")
    
    print(f"\n💡 Real Problem Analysis:")
    print("1. ✅ Agent IS learning (max reward +25.418!)")
    print("2. ✅ Some episodes are successful")
    print("3. ❌ Too many failed episodes bringing average down")
    print("4. ❌ Need to improve success rate, not just reduce penalties")
    
    print(f"\n🎯 New Strategy:")
    print("Instead of fixing step penalty, we should:")
    print("1. 🎯 Improve action quality to avoid failures")
    print("2. 📈 Add dense rewards to guide learning") 
    print("3. 🚀 Focus on increasing success rate")
    print("4. ⚖️ Balance exploration vs exploitation")

def create_success_focused_wrapper():
    """Create wrapper focused on improving success rate"""
    
    wrapper_code = '''"""
Success-Focused Environment Wrapper
Focus on improving success rate rather than just reducing penalties
"""

import numpy as np

class SuccessFocusedWrapper:
    """
    Wrapper that provides dense rewards to improve success rate
    """
    
    def __init__(self, env):
        self.env = env
        self.episode_step = 0
        self.best_ball_dist = float('inf')
        self.stability_streak = 0
        self.progress_rewards = 0
        
    def reset(self, **kwargs):
        self.episode_step = 0
        self.best_ball_dist = float('inf')
        self.stability_streak = 0
        self.progress_rewards = 0
        return self.env.reset(**kwargs)
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.episode_step += 1
        
        # Add dense shaping rewards during training
        shaping_reward = 0
        
        # Extract state info
        ball_pos = info.get("ball_xpos_rel_robot", np.zeros(3))
        if len(ball_pos.shape) > 1:
            ball_pos = ball_pos[0]
        ball_dist = np.linalg.norm(ball_pos)
        
        robot_quat = info.get("robot_quat", np.array([0, 0, 0, 1]))
        if len(robot_quat.shape) > 1:
            robot_quat = robot_quat[0]
        robot_upright = 1.0 - abs(robot_quat[2])
        
        # 1. Stability rewards (prevent falling)
        if robot_upright > 0.9:
            self.stability_streak += 1
            shaping_reward += 0.01 + min(self.stability_streak * 0.001, 0.05)
        else:
            self.stability_streak = 0
            if robot_upright < 0.7:
                shaping_reward -= 0.02  # Instability penalty
        
        # 2. Ball approach rewards (main objective)
        if ball_dist < self.best_ball_dist:
            progress = self.best_ball_dist - ball_dist
            shaping_reward += min(progress * 5.0, 0.1)  # Reward progress
            self.best_ball_dist = ball_dist
        
        # 3. Distance-based rewards
        if robot_upright > 0.8:
            if ball_dist < 0.5:
                shaping_reward += 0.05  # Very close
            elif ball_dist < 1.0:
                shaping_reward += 0.02  # Close
        
        # 4. Episode completion bonus
        if terminated or truncated:
            if self.episode_step > 150:  # Survived reasonable time
                shaping_reward += 2.0
            self.progress_rewards += shaping_reward
            
            # Print episode summary
            print(f"Episode: {self.episode_step} steps, progress rewards: {self.progress_rewards:.2f}")
            
            # Return original + shaping for final reward
            return obs, reward + self.progress_rewards, terminated, truncated, info
        else:
            # During episode, return just shaping rewards
            self.progress_rewards += shaping_reward
            return obs, shaping_reward, terminated, truncated, info
    
    def __getattr__(self, name):
        return getattr(self.env, name)
'''
    
    return wrapper_code

if __name__ == "__main__":
    analyze_original_training()
    
    print("\n" + "="*60)
    print("🚀 RECOMMENDED NEXT STEPS:")
    print("="*60)
    print("1. 📊 Your agent IS learning (max +25.418 proves it works!)")
    print("2. 🎯 Focus on improving success rate")
    print("3. 📈 Use dense rewards to guide learning")
    print("4. ⚖️ Reduce action noise/improve stability")
    print("5. 🚀 Train longer with better guidance")
    print("\nThe step penalty is NOT the main problem!")
    print("The real issue is inconsistent performance.")