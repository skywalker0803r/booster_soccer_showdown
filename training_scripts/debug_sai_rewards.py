"""
Debug SAI Reward System
Deep dive into understanding when and how SAI gives rewards
"""

import numpy as np
from sai_rl import SAIClient

def debug_sai_rewards_detailed():
    """Detailed analysis of SAI reward system"""
    
    print("🔍 Deep SAI Reward System Analysis")
    print("="*60)
    
    # Initialize SAI
    sai = SAIClient(comp_id="booster-soccer-showdown", api_key="sai_LFcuaCZiqEkUbNVolQ3wbk5yU7H11jfv")
    env = sai.make_env()
    
    print(f"📊 Environment Info:")
    print(f"   Action space: {env.action_space}")
    print(f"   Observation space: {env.observation_space}")
    
    # Test different episode lengths
    for max_steps in [50, 100, 200, 400]:
        print(f"\n🎯 Testing {max_steps} step episode:")
        print("-" * 40)
        
        obs, info = env.reset()
        episode_rewards = []
        total_reward = 0
        step_count = 0
        
        for step in range(max_steps):
            # Small random actions to avoid immediate failure
            action = np.random.uniform(-0.2, 0.2, size=env.action_space.shape[0])
            
            next_obs, reward, terminated, truncated, next_info = env.step(action)
            
            episode_rewards.append(reward)
            total_reward += reward
            step_count += 1
            
            # Print info for first few and last few steps
            if step < 5 or step > max_steps - 6 or abs(reward) > 0.001:
                print(f"   Step {step:3d}: reward={reward:8.3f}, term={terminated}, trunc={truncated}")
            
            if terminated or truncated:
                print(f"   → Episode ended early at step {step}")
                print(f"     Terminated: {terminated}, Truncated: {truncated}")
                break
            
            obs = next_obs
            info = next_info
        
        print(f"   📈 Results:")
        print(f"      Total steps: {step_count}")
        print(f"      Total reward: {total_reward:.3f}")
        print(f"      Average reward/step: {total_reward/step_count:.6f}")
        print(f"      Non-zero rewards: {len([r for r in episode_rewards if abs(r) > 0.001])}")
        print(f"      Reward range: {min(episode_rewards):.3f} to {max(episode_rewards):.3f}")
        
        # Check if this looks like step penalty
        if abs(total_reward + step_count) < 1.0:  # Close to -1.0 * steps
            print(f"      ⚠️ Looks like step penalty: {total_reward:.3f} ≈ -{step_count}")
        elif abs(total_reward) < 0.1:
            print(f"      ✅ No significant penalties detected")
        else:
            print(f"      🤔 Unexpected reward pattern")
    
    # Test with forced termination
    print(f"\n🎯 Testing Environment Termination Conditions:")
    print("-" * 50)
    
    obs, info = env.reset()
    
    # Try to make robot fall (extreme actions)
    print("Trying extreme actions to trigger termination...")
    for step in range(100):
        # Extreme action that should make robot fall
        action = np.ones(env.action_space.shape[0]) * 5.0  # Very large action
        
        next_obs, reward, terminated, truncated, next_info = env.step(action)
        
        if terminated or truncated or abs(reward) > 0.001:
            print(f"   Step {step}: reward={reward:.3f}, terminated={terminated}, truncated={truncated}")
            
            # Check info for clues
            interesting_keys = ['robot_fallen', 'goal_scored', 'ball_out_of_bounds', 'offside']
            for key in interesting_keys:
                if key in next_info:
                    print(f"      {key}: {next_info[key]}")
        
        if terminated or truncated:
            print(f"   🎉 Episode terminated at step {step}!")
            break
        
        obs = next_obs
        info = next_info
    else:
        print("   😞 No termination triggered even with extreme actions")
    
    print(f"\n💡 Key Findings:")
    print("1. Check if rewards are only given at episode end")
    print("2. Look for reward accumulation patterns")
    print("3. Understand termination conditions")
    print("4. Compare training vs evaluation modes")

if __name__ == "__main__":
    debug_sai_rewards_detailed()