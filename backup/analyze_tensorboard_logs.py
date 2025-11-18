"""
TensorBoard Log Analyzer
Analyzes training logs to understand what went wrong
"""

import os
import sys
try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    print("⚠️ TensorBoard not available. Install with: pip install tensorboard")

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def analyze_tensorboard_logs(log_dir):
    """Analyze TensorBoard logs and create visual reports"""
    
    if not TENSORBOARD_AVAILABLE:
        print("❌ Cannot analyze logs without TensorBoard package")
        return
    
    print(f"🔍 Analyzing logs in: {log_dir}")
    
    # Find event files
    event_files = list(Path(log_dir).glob("*.tfevents.*"))
    
    if not event_files:
        print("❌ No TensorBoard event files found!")
        return
    
    # Load event data
    ea = EventAccumulator(str(event_files[0]))
    ea.Reload()
    
    # Get available tags
    tags = ea.Tags()
    print("📊 Available metrics:")
    for category, tag_list in tags.items():
        if tag_list:
            print(f"  {category}: {tag_list}")
    
    # Create analysis plots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('DreamerV3 Training Analysis', fontsize=16)
    
    # 1. Episode Rewards
    if 'Episode/Reward' in tags['scalars']:
        reward_data = ea.Scalars('Episode/Reward')
        episodes = [x.step for x in reward_data]
        rewards = [x.value for x in reward_data]
        
        axes[0, 0].plot(episodes, rewards, 'b-', alpha=0.7)
        axes[0, 0].set_title('Episode Rewards')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Reward')
        axes[0, 0].grid(True)
        
        # Add statistics
        avg_reward = np.mean(rewards[-100:]) if len(rewards) >= 100 else np.mean(rewards)
        axes[0, 0].axhline(y=avg_reward, color='r', linestyle='--', label=f'Avg: {avg_reward:.2f}')
        axes[0, 0].legend()
        
        print(f"📈 Reward Stats:")
        print(f"   Final avg (last 100): {avg_reward:.3f}")
        print(f"   Min reward: {min(rewards):.3f}")
        print(f"   Max reward: {max(rewards):.3f}")
    
    # 2. Episode Length
    if 'Episode/Length' in tags['scalars']:
        length_data = ea.Scalars('Episode/Length')
        episodes = [x.step for x in length_data]
        lengths = [x.value for x in length_data]
        
        axes[0, 1].plot(episodes, lengths, 'g-', alpha=0.7)
        axes[0, 1].set_title('Episode Lengths')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Steps')
        axes[0, 1].grid(True)
        
        avg_length = np.mean(lengths[-100:]) if len(lengths) >= 100 else np.mean(lengths)
        print(f"📏 Episode Length Stats:")
        print(f"   Final avg: {avg_length:.1f} steps")
        print(f"   Min length: {min(lengths):.1f}")
        print(f"   Max length: {max(lengths):.1f}")
    
    # 3. World Model Loss
    loss_tags = [tag for tag in tags['scalars'] if 'loss' in tag.lower()]
    if loss_tags:
        ax = axes[0, 2]
        for i, tag in enumerate(loss_tags[:3]):  # Plot first 3 loss types
            loss_data = ea.Scalars(tag)
            steps = [x.step for x in loss_data]
            losses = [x.value for x in loss_data]
            ax.plot(steps, losses, label=tag.split('/')[-1], alpha=0.7)
        
        ax.set_title('Training Losses')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Loss')
        ax.set_yscale('log')
        ax.grid(True)
        ax.legend()
    
    # 4. Gradient Norms
    grad_tags = [tag for tag in tags['scalars'] if 'grad' in tag.lower()]
    if grad_tags:
        ax = axes[1, 0]
        for tag in grad_tags:
            grad_data = ea.Scalars(tag)
            steps = [x.step for x in grad_data]
            grads = [x.value for x in grad_data]
            ax.plot(steps, grads, label=tag.split('/')[-1], alpha=0.7)
        
        ax.set_title('Gradient Norms')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Gradient Norm')
        ax.set_yscale('log')
        ax.grid(True)
        ax.legend()
    
    # 5. Early Terminations
    if 'Episode/EarlyTerminations' in tags['scalars']:
        et_data = ea.Scalars('Episode/EarlyTerminations')
        episodes = [x.step for x in et_data]
        early_terms = [x.value for x in et_data]
        
        axes[1, 1].plot(episodes, early_terms, 'r-', alpha=0.7)
        axes[1, 1].set_title('Early Terminations (Cumulative)')
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('Count')
        axes[1, 1].grid(True)
        
        final_early_terms = early_terms[-1] if early_terms else 0
        total_episodes = episodes[-1] if episodes else 1
        et_rate = final_early_terms / total_episodes * 100
        print(f"⚠️ Early Termination Rate: {et_rate:.1f}% ({final_early_terms}/{total_episodes})")
    
    # 6. Average Rewards
    if 'Average/Reward_100ep' in tags['scalars']:
        avg_data = ea.Scalars('Average/Reward_100ep')
        episodes = [x.step for x in avg_data]
        avg_rewards = [x.value for x in avg_data]
        
        axes[1, 2].plot(episodes, avg_rewards, 'purple', alpha=0.7)
        axes[1, 2].set_title('100-Episode Average Reward')
        axes[1, 2].set_xlabel('Episode')
        axes[1, 2].set_ylabel('Average Reward')
        axes[1, 2].grid(True)
        
        # Check for improvement trend
        if len(avg_rewards) > 10:
            early_avg = np.mean(avg_rewards[:10])
            late_avg = np.mean(avg_rewards[-10:])
            improvement = late_avg - early_avg
            print(f"📊 Learning Progress:")
            print(f"   Early avg: {early_avg:.3f}")
            print(f"   Late avg: {late_avg:.3f}")
            print(f"   Improvement: {improvement:+.3f}")
    
    plt.tight_layout()
    
    # Save analysis
    output_path = os.path.join(log_dir, 'training_analysis.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"💾 Analysis saved to: {output_path}")
    
    # Diagnostic summary
    print("\n" + "="*60)
    print("🔍 DIAGNOSTIC SUMMARY")
    print("="*60)
    
    # Check for common problems
    if 'Episode/Reward' in tags['scalars']:
        reward_data = ea.Scalars('Episode/Reward')
        recent_rewards = [x.value for x in reward_data[-50:]]  # Last 50 episodes
        
        if all(r < -10 for r in recent_rewards):
            print("❌ PROBLEM: All recent rewards are very negative")
            print("   → Step penalty likely not fixed correctly")
        
        if np.std(recent_rewards) < 1:
            print("❌ PROBLEM: Very low reward variance")
            print("   → Agent not learning/exploring effectively")
        
        avg_reward = np.mean(recent_rewards)
        if avg_reward < -20:
            print(f"❌ PROBLEM: Average reward too low ({avg_reward:.2f})")
            print("   → Core reward mechanism still broken")
    
    # Check world model convergence
    if 'Loss/World Model Loss' in tags['scalars']:
        wm_data = ea.Scalars('Loss/World Model Loss')
        recent_wm_losses = [x.value for x in wm_data[-10:]]
        
        if all(loss > 2.0 for loss in recent_wm_losses):
            print("❌ PROBLEM: World model loss not converging")
            print("   → Architecture or training issues")
        elif all(loss < 1.0 for loss in recent_wm_losses):
            print("✅ GOOD: World model loss converged well")
    
    # Check for early termination issues
    if 'Episode/Length' in tags['scalars']:
        length_data = ea.Scalars('Episode/Length')
        recent_lengths = [x.value for x in length_data[-50:]]
        avg_length = np.mean(recent_lengths)
        
        if avg_length < 50:
            print("❌ PROBLEM: Episodes too short (early termination)")
            print("   → Robot likely falling down immediately")
        elif avg_length > 300:
            print("✅ GOOD: Episodes have reasonable length")
    
    return True

def find_latest_tensorboard_run():
    """Find the most recent TensorBoard run"""
    runs_dir = Path("runs")
    
    if not runs_dir.exists():
        print("❌ No 'runs' directory found!")
        return None
    
    # Find all run directories
    run_dirs = [d for d in runs_dir.iterdir() if d.is_dir()]
    
    if not run_dirs:
        print("❌ No run directories found!")
        return None
    
    # Get the most recent one
    latest_run = max(run_dirs, key=lambda x: x.stat().st_mtime)
    
    print(f"🎯 Found latest run: {latest_run}")
    return latest_run

if __name__ == "__main__":
    print("🔍 TensorBoard Log Analyzer")
    print("="*40)
    
    if len(sys.argv) > 1:
        log_dir = sys.argv[1]
    else:
        log_dir = find_latest_tensorboard_run()
    
    if log_dir and Path(log_dir).exists():
        analyze_tensorboard_logs(log_dir)
    else:
        print("❌ No valid log directory found!")
        print("Usage: python analyze_tensorboard_logs.py [log_directory]")
        
        # Try to find the specific run mentioned in log.txt
        specific_run = Path("runs/ImprovedDreamerV3_20251116_162447")
        if specific_run.exists():
            print(f"🎯 Found specific run directory: {specific_run}")
            analyze_tensorboard_logs(specific_run)
        else:
            print(f"❌ Specific run directory not found: {specific_run}")