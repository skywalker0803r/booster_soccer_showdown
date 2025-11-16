"""
Quick Test of Episode-Level Reward Fixing
Test with actual SAI environment for 5 episodes
"""

import numpy as np
from sai_rl import SAIClient
from episode_reward_wrapper import EpisodeRewardEnvironmentWrapper

def quick_test_episode_fixing():
    """Quick test with real environment"""
    
    print("🚀 Quick Test: Episode-Level Reward Fixing")
    print("="*60)
    
    # Initialize SAI
    sai = SAIClient(comp_id="booster-soccer-showdown", api_key="sai_LFcuaCZiqEkUbNVolQ3wbk5yU7H11jfv")
    original_env = sai.make_env()
    
    # Wrap with our episode fixer
    env = EpisodeRewardEnvironmentWrapper(original_env)
    
    total_original = 0
    total_fixed = 0
    episode_lengths = []
    
    for episode in range(5):  # Test 5 short episodes
        print(f"\n🎯 Episode {episode + 1}")
        print("-" * 30)
        
        obs, info = env.reset()
        episode_reward_original = 0
        episode_reward_fixed = 0
        steps = 0
        
        for step in range(200):  # Max 200 steps
            # Random action
            action = original_env.action_space.sample() * 0.1  # Small actions for stability
            
            # Step environment  
            next_obs, reward, terminated, truncated, next_info = env.step(action)
            
            episode_reward_fixed += reward
            steps += 1
            
            if terminated or truncated:
                # Calculate what original reward would have been
                episode_reward_original = -1.0 * steps  # Just step penalty for comparison
                break
                
            obs = next_obs
            info = next_info
        
        episode_lengths.append(steps)
        total_original += episode_reward_original
        total_fixed += episode_reward_fixed
        
        print(f"Episode {episode + 1} Results:")
        print(f"   Steps: {steps}")
        print(f"   Original (estimated): {episode_reward_original:.3f}")
        print(f"   Fixed: {episode_reward_fixed:.3f}")
        print(f"   Improvement: {episode_reward_fixed - episode_reward_original:+.3f}")
    
    print(f"\n" + "="*60)
    print(f"🏆 SUMMARY:")
    print(f"   Total original: {total_original:.3f}")
    print(f"   Total fixed: {total_fixed:.3f}")
    print(f"   Total improvement: {total_fixed - total_original:+.3f}")
    print(f"   Average episode length: {np.mean(episode_lengths):.1f}")
    
    if total_fixed > total_original + 50:
        print("   ✅ SUCCESS: Major improvement in episode rewards!")
        print("   → Ready to use this fix in training")
    elif total_fixed > total_original:
        print("   ⚠️ PARTIAL: Some improvement, may need tuning")
    else:
        print("   ❌ FAILED: No improvement, check logic")
    
    return total_fixed > total_original + 50

if __name__ == "__main__":
    success = quick_test_episode_fixing()
    
    if success:
        print(f"\n🎉 Episode reward fixing successful!")
        print(f"Next step: Update training script to use EpisodeRewardEnvironmentWrapper")
    else:
        print(f"\n😞 Need to debug the episode reward fixing logic")