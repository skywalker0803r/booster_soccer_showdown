"""
Simple Log Analyzer (No TensorBoard dependency)
Analyzes log.txt and provides insights
"""

import re
import numpy as np
from pathlib import Path

def analyze_log_txt():
    """Analyze the log.txt file for patterns and issues"""
    
    log_file = Path("log.txt")
    if not log_file.exists():
        print("❌ log.txt not found!")
        return
    
    print("🔍 Analyzing log.txt")
    print("="*50)
    
    # Read log content
    with open(log_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Extract episode data
    episode_pattern = r"Episode\s+(\d+):\s+Reward\s+=\s+([-\d.]+),\s+Steps\s+=\s+(\d+),\s+Avg100\s+=\s+([-\d.]+)"
    episode_matches = re.findall(episode_pattern, content)
    
    if episode_matches:
        episodes = [int(m[0]) for m in episode_matches]
        rewards = [float(m[1]) for m in episode_matches]
        steps = [int(m[2]) for m in episode_matches]
        avg_rewards = [float(m[3]) for m in episode_matches]
        
        print(f"📊 Found {len(episode_matches)} episode records")
        print(f"   Episode range: {min(episodes)} - {max(episodes)}")
        print(f"   Final episode reward: {rewards[-1]:.3f}")
        print(f"   Final 100-ep average: {avg_rewards[-1]:.3f}")
        print(f"   Final episode length: {steps[-1]} steps")
        
        # Statistics
        print(f"\n📈 Reward Statistics:")
        print(f"   Min reward: {min(rewards):.3f}")
        print(f"   Max reward: {max(rewards):.3f}")
        print(f"   Average reward: {np.mean(rewards):.3f}")
        print(f"   Std deviation: {np.std(rewards):.3f}")
        
        print(f"\n📏 Episode Length Statistics:")
        print(f"   Min length: {min(steps)} steps")
        print(f"   Max length: {max(steps)} steps")
        print(f"   Average length: {np.mean(steps):.1f} steps")
        
        # Check for improvement trends
        if len(rewards) > 20:
            early_rewards = rewards[:10]
            late_rewards = rewards[-10:]
            improvement = np.mean(late_rewards) - np.mean(early_rewards)
            print(f"\n📊 Learning Progress:")
            print(f"   Early episodes avg: {np.mean(early_rewards):.3f}")
            print(f"   Recent episodes avg: {np.mean(late_rewards):.3f}")
            print(f"   Improvement: {improvement:+.3f}")
            
            if improvement < 1.0:
                print("   ⚠️  Very little learning progress detected!")
    
    # Extract reward shaping data
    step_pattern = r"Step\s+(\d+):\s+Original reward:\s+([-\d.]+),\s+Shaped reward:\s+([-\d.]+)"
    step_matches = re.findall(step_pattern, content)
    
    if step_matches:
        step_nums = [int(m[0]) for m in step_matches]
        orig_rewards = [float(m[1]) for m in step_matches]
        shaped_rewards = [float(m[2]) for m in step_matches]
        
        print(f"\n🔧 Reward Shaping Analysis:")
        print(f"   Found {len(step_matches)} step-level records")
        
        # Check for -1.0 step penalties
        step_penalties = [r for r in orig_rewards if abs(r + 1.0) < 0.01]
        zero_rewards = [r for r in orig_rewards if abs(r) < 0.01]
        
        print(f"   Steps with -1.0 penalty: {len(step_penalties)}")
        print(f"   Steps with zero reward: {len(zero_rewards)}")
        
        if step_penalties:
            # Check if we successfully replaced -1.0 penalties
            corresponding_shaped = [shaped_rewards[i] for i, r in enumerate(orig_rewards) if abs(r + 1.0) < 0.01]
            if corresponding_shaped:
                avg_replacement = np.mean(corresponding_shaped)
                print(f"   Average shaped reward for -1.0 penalties: {avg_replacement:.3f}")
                
                if avg_replacement > -0.1:
                    print("   ✅ Step penalty replacement working!")
                else:
                    print("   ❌ Step penalty replacement failed!")
    
    # Extract loss data
    loss_pattern = r"(World Model Loss|Reconstruction|Reward|KL|Actor|Critic):\s+([-\d.]+)"
    loss_matches = re.findall(loss_pattern, content)
    
    if loss_matches:
        print(f"\n🧠 Model Training Analysis:")
        loss_dict = {}
        for loss_type, loss_value in loss_matches:
            if loss_type not in loss_dict:
                loss_dict[loss_type] = []
            loss_dict[loss_type].append(float(loss_value))
        
        for loss_type, values in loss_dict.items():
            if values:
                latest_value = values[-1]
                print(f"   {loss_type}: {latest_value:.4f}")
                
                # Specific diagnostics
                if loss_type == "World Model Loss" and latest_value > 2.0:
                    print(f"     ⚠️  High world model loss - convergence issues")
                elif loss_type == "Reconstruction" and latest_value < 0.5:
                    print(f"     ✅ Good reconstruction loss")
                elif loss_type == "KL" and latest_value > 5.0:
                    print(f"     ⚠️  High KL divergence - regularization issues")
    
    # Check for specific problems
    print(f"\n🩺 Problem Diagnosis:")
    
    # Problem 1: Step penalty not fixed
    if episode_matches and all(r < -10 for r in rewards[-5:]):
        print("❌ CRITICAL: Step penalty still causing very negative rewards")
        print("   → Reward replacement mechanism not working correctly")
    
    # Problem 2: Early termination
    if episode_matches:
        short_episodes = [s for s in steps[-10:] if s < 50]
        if len(short_episodes) > 5:
            print("❌ CRITICAL: Too many short episodes (early termination)")
            print("   → Robot is falling down immediately")
    
    # Problem 3: No learning
    if episode_matches and len(rewards) > 50:
        recent_variance = np.std(rewards[-20:])
        if recent_variance < 2.0:
            print("❌ PROBLEM: Very low reward variance")
            print("   → Agent not exploring or learning effectively")
    
    # Problem 4: Shaped rewards not improving
    if step_matches:
        negative_shaped = [r for r in shaped_rewards if r < -0.5]
        if len(negative_shaped) / len(shaped_rewards) > 0.8:
            print("❌ PROBLEM: Most shaped rewards still very negative")
            print("   → Reward shaping system needs adjustment")
    
    # Recommendations
    print(f"\n💡 Recommendations:")
    if episode_matches and np.mean(rewards[-5:]) < -20:
        print("1. 🔧 Fix reward replacement mechanism - step penalties still too high")
        print("2. 📉 Check environment wrapper is actually being used")
        print("3. 🎯 Consider even more aggressive reward reshaping")
    
    if episode_matches and np.mean(steps[-5:]) < 100:
        print("4. 🤖 Add stability rewards to encourage standing")
        print("5. ⚙️ Check action bounds and scaling")
        print("6. 🎮 Start with smaller/safer actions")
    
    print("\n" + "="*50)
    print("Analysis complete! Check above for specific issues.")

if __name__ == "__main__":
    analyze_log_txt()