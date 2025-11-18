"""
Debug Environment Rewards
Let's understand what the actual reward mechanism is
"""

import numpy as np
from sai_rl import SAIClient

def debug_environment_rewards():
    """Debug the actual reward structure"""
    
    print("🔍 Debugging SAI Environment Rewards")
    print("="*50)
    
    # Initialize SAI
    sai = SAIClient(comp_id="booster-soccer-showdown", api_key="sai_LFcuaCZiqEkUbNVolQ3wbk5yU7H11jfv")
    env = sai.make_env()
    
    # Run a short episode to understand rewards
    obs, info = env.reset()
    print(f"Initial observation shape: {obs.shape}")
    
    total_reward = 0
    rewards = []
    
    print("\n📊 Step-by-step reward analysis:")
    print("Step | Reward  | Cumulative | Terminated | Info")
    print("-" * 50)
    
    for step in range(50):  # Short test episode
        # Take random action
        action = env.action_space.sample()
        
        # Step environment
        next_obs, reward, terminated, truncated, next_info = env.step(action)
        
        total_reward += reward
        rewards.append(reward)
        
        # Print detailed info
        print(f"{step:4d} | {reward:7.3f} | {total_reward:10.3f} | {terminated:10} | ", end="")
        
        # Check for specific events
        events = []
        if 'robot_fallen' in next_info and next_info.get('robot_fallen', False):
            events.append("FALLEN")
        if 'goal_scored' in next_info and next_info.get('goal_scored', False):
            events.append("GOAL")
        if 'offside' in next_info and next_info.get('offside', False):
            events.append("OFFSIDE")
        if 'ball_out_of_bounds' in next_info and next_info.get('ball_out_of_bounds', False):
            events.append("OUT_OF_BOUNDS")
        
        if events:
            print(" | ".join(events))
        else:
            print("Normal step")
        
        if terminated or truncated:
            print(f"\nEpisode ended at step {step}")
            break
        
        obs = next_obs
        info = next_info
    
    # Analysis
    print(f"\n📈 Reward Analysis:")
    print(f"   Total reward: {total_reward:.3f}")
    print(f"   Average reward per step: {np.mean(rewards):.3f}")
    print(f"   Reward std deviation: {np.std(rewards):.3f}")
    print(f"   Non-zero rewards: {len([r for r in rewards if abs(r) > 0.001])}")
    print(f"   Zero rewards: {len([r for r in rewards if abs(r) < 0.001])}")
    
    # Check for patterns
    zero_rewards = [r for r in rewards if abs(r) < 0.001]
    negative_rewards = [r for r in rewards if r < -0.001]
    positive_rewards = [r for r in rewards if r > 0.001]
    
    print(f"\n📊 Reward Breakdown:")
    print(f"   Zero rewards: {len(zero_rewards)} ({len(zero_rewards)/len(rewards)*100:.1f}%)")
    print(f"   Negative rewards: {len(negative_rewards)} ({len(negative_rewards)/len(rewards)*100:.1f}%)")
    print(f"   Positive rewards: {len(positive_rewards)} ({len(positive_rewards)/len(rewards)*100:.1f}%)")
    
    if negative_rewards:
        print(f"   Average negative: {np.mean(negative_rewards):.3f}")
        print(f"   Most negative: {min(negative_rewards):.3f}")
    
    if positive_rewards:
        print(f"   Average positive: {np.mean(positive_rewards):.3f}")
        print(f"   Most positive: {max(positive_rewards):.3f}")
    
    # Conclusions
    print(f"\n💡 Conclusions:")
    if len(zero_rewards) / len(rewards) > 0.8:
        print("✅ Environment gives mostly zero rewards (sparse reward)")
        print("   → Step penalty is NOT -1.0 per step!")
        print("   → Need different reward shaping strategy")
    else:
        print("⚠️ Environment has dense rewards")
        print("   → Step penalty might be embedded in the reward calculation")
    
    # Check reward configuration from docs
    print(f"\n📖 From docs/Evaluation.MD:")
    print("   steps: -1 (penalty per time step)")
    print("   BUT environment might aggregate these differently!")
    
    return total_reward, rewards

if __name__ == "__main__":
    debug_environment_rewards()