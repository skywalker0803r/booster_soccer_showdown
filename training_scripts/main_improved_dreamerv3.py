"""
FIXED DreamerV3 Training Script
Addresses all major issues:
1. Step penalty problem (-1.0 -> -0.01)
2. World model loss convergence issues
3. Robot behavior problems (early termination, no movement)
4. Better training stability
"""

import torch
import torch.nn.functional as F
import numpy as np
from collections import deque
import random
from torch.utils.tensorboard import SummaryWriter
import os
from datetime import datetime

from sai_rl import SAIClient
from improved_dreamerv3 import ImprovedDreamerV3
from reward_shaping_wrapper import FixedRewardPreprocessor, create_fixed_reward_environment_wrapper
from sai_compatible_dreamerv3 import SAICompatibleDreamerV3

## Initialize the SAI client
sai = SAIClient(comp_id="booster-soccer-showdown", api_key="sai_LFcuaCZiqEkUbNVolQ3wbk5yU7H11jfv")

## Make the environment
env = sai.make_env()

class Preprocessor():
    """Original preprocessor from main_simple_dreamerv3.py"""
    
    def get_task_onehot(self, info):
        if 'task_index' in info:
            return info['task_index']
        else:
            return np.array([])

    def quat_rotate_inverse(self, q: np.ndarray, v: np.ndarray):
        q_w = q[:,[-1]]
        q_vec = q[:,:3]
        a = v * (2.0 * q_w**2 - 1.0)
        b = np.cross(q_vec, v) * (q_w * 2.0)
        c = q_vec * (np.dot(q_vec, v).reshape(-1,1) * 2.0)    
        return a - b + c 

    def modify_state(self, obs, info):
        if len(obs.shape) == 1:
            obs = np.expand_dims(obs, axis=0)

        task_onehot = self.get_task_onehot(info)
        if len(task_onehot.shape) == 1:
            task_onehot = np.expand_dims(task_onehot, axis=0)
        
        if len(info["robot_quat"].shape) == 1:
            info["robot_quat"] = np.expand_dims(info["robot_quat"], axis = 0)
            info["robot_gyro"] = np.expand_dims(info["robot_gyro"], axis = 0)
            info["robot_accelerometer"] = np.expand_dims(info["robot_accelerometer"], axis = 0)
            info["robot_velocimeter"] = np.expand_dims(info["robot_velocimeter"], axis = 0)
            info["goal_team_0_rel_robot"] = np.expand_dims(info["goal_team_0_rel_robot"], axis = 0)
            info["goal_team_1_rel_robot"] = np.expand_dims(info["goal_team_1_rel_robot"], axis = 0)
            info["goal_team_0_rel_ball"] = np.expand_dims(info["goal_team_0_rel_ball"], axis = 0)
            info["goal_team_1_rel_ball"] = np.expand_dims(info["goal_team_1_rel_ball"], axis = 0)
            info["ball_xpos_rel_robot"] = np.expand_dims(info["ball_xpos_rel_robot"], axis = 0) 
            info["ball_velp_rel_robot"] = np.expand_dims(info["ball_velp_rel_robot"], axis = 0) 
            info["ball_velr_rel_robot"] = np.expand_dims(info["ball_velr_rel_robot"], axis = 0) 
            info["player_team"] = np.expand_dims(info["player_team"], axis = 0)
            info["goalkeeper_team_0_xpos_rel_robot"] = np.expand_dims(info["goalkeeper_team_0_xpos_rel_robot"], axis = 0)
            info["goalkeeper_team_0_velp_rel_robot"] = np.expand_dims(info["goalkeeper_team_0_velp_rel_robot"], axis = 0)
            info["goalkeeper_team_1_xpos_rel_robot"] = np.expand_dims(info["goalkeeper_team_1_xpos_rel_robot"], axis = 0)
            info["goalkeeper_team_1_velp_rel_robot"] = np.expand_dims(info["goalkeeper_team_1_velp_rel_robot"], axis = 0)
            info["target_xpos_rel_robot"] = np.expand_dims(info["target_xpos_rel_robot"], axis = 0)
            info["target_velp_rel_robot"] = np.expand_dims(info["target_velp_rel_robot"], axis = 0)
            info["defender_xpos"] = np.expand_dims(info["defender_xpos"], axis = 0)
        
        robot_qpos = obs[:,:12]
        robot_qvel = obs[:,12:24]
        quat = info["robot_quat"]
        base_ang_vel = info["robot_gyro"]
        project_gravity = self.quat_rotate_inverse(quat, np.array([0.0, 0.0, -1.0]))
        
        obs = np.hstack((robot_qpos, 
                         robot_qvel,
                         project_gravity,
                         base_ang_vel,
                         info["robot_accelerometer"],
                         info["robot_velocimeter"],
                         info["goal_team_0_rel_robot"], 
                         info["goal_team_1_rel_robot"], 
                         info["goal_team_0_rel_ball"], 
                         info["goal_team_1_rel_ball"], 
                         info["ball_xpos_rel_robot"], 
                         info["ball_velp_rel_robot"], 
                         info["ball_velr_rel_robot"], 
                         info["player_team"], 
                         info["goalkeeper_team_0_xpos_rel_robot"], 
                         info["goalkeeper_team_0_velp_rel_robot"], 
                         info["goalkeeper_team_1_xpos_rel_robot"], 
                         info["goalkeeper_team_1_velp_rel_robot"], 
                         info["target_xpos_rel_robot"], 
                         info["target_velp_rel_robot"], 
                         info["defender_xpos"],
                         task_onehot))

        return obs


class ImprovedSequenceBuffer:
    """Improved buffer with better memory management"""
    
    def __init__(self, max_size=2000, sequence_length=50):
        self.max_size = max_size
        self.sequence_length = sequence_length
        self.buffer = deque(maxlen=max_size)
        self.total_steps = 0
        
    def add_episode(self, observations, actions, rewards):
        """Add episode with validation"""
        if len(observations) > 1 and len(actions) > 0 and len(rewards) > 0:
            episode = {
                'observations': np.array(observations),
                'actions': np.array(actions),
                'rewards': np.array(rewards)
            }
            self.buffer.append(episode)
            self.total_steps += len(rewards)
        
    def sample_sequences(self, batch_size):
        """Improved sequence sampling"""
        if len(self.buffer) == 0:
            return None, None, None
            
        batch_obs = []
        batch_actions = []
        batch_rewards = []
        
        for _ in range(batch_size):
            # Sample episode
            episode = random.choice(self.buffer)
            episode_length = len(episode['observations']) - 1  # -1 because obs has one more than actions
            
            if episode_length < self.sequence_length:
                # Handle short episodes with padding
                obs_seq = episode['observations'][:-1]  # Remove last obs to match action length
                act_seq = episode['actions']
                rew_seq = episode['rewards']
                
                # Pad sequences
                pad_length = self.sequence_length - episode_length
                if pad_length > 0:
                    obs_pad = np.repeat(obs_seq[-1:], pad_length, axis=0)
                    act_pad = np.repeat(act_seq[-1:], pad_length, axis=0)
                    rew_pad = np.zeros(pad_length)
                    
                    obs_seq = np.concatenate([obs_seq, obs_pad], axis=0)
                    act_seq = np.concatenate([act_seq, act_pad], axis=0)
                    rew_seq = np.concatenate([rew_seq, rew_pad], axis=0)
            else:
                # Sample random subsequence
                start_idx = random.randint(0, episode_length - self.sequence_length)
                end_idx = start_idx + self.sequence_length
                
                obs_seq = episode['observations'][start_idx:end_idx]
                act_seq = episode['actions'][start_idx:end_idx]
                rew_seq = episode['rewards'][start_idx:end_idx]
            
            batch_obs.append(obs_seq)
            batch_actions.append(act_seq)
            batch_rewards.append(rew_seq)
        
        return (torch.FloatTensor(np.array(batch_obs)),
                torch.FloatTensor(np.array(batch_actions)),
                torch.FloatTensor(np.array(batch_rewards)))


def improved_dreamerv3_training_loop():
    """Fixed training loop with all improvements"""
    
    print("=== Starting IMPROVED DreamerV3 Training ===")
    print("🔧 Fixes applied:")
    print("   - Step penalty: -1.0 → -0.01 (100x reduction!)")
    print("   - Early termination penalty: -10.0 for premature falls")
    print("   - Improved world model loss functions")
    print("   - Better exploration and stability rewards")
    print("   - Enhanced network architectures")
    
    # Check device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔥 Using device: {device}")
    if device.type == "cuda":
        print(f"   GPU Name: {torch.cuda.get_device_name(0)}")
        print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # TensorBoard
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"runs/ImprovedDreamerV3_{timestamp}"
    writer = SummaryWriter(log_dir)
    print(f"📊 TensorBoard logs: {log_dir}")
    print(f"🚀 To view: tensorboard --logdir=runs --port=6006")
    
    # Create FIXED environment wrapper
    fixed_env = create_fixed_reward_environment_wrapper(env, Preprocessor)
    print("✅ Environment wrapped with reward fixing")
    
    # Initialize improved model
    model = ImprovedDreamerV3(
        obs_dim=89,
        action_dim=12,
        hidden_dim=256,  # Slightly smaller for stability
        stoch_dim=32,
        discrete_dim=16
    ).to(device)
    
    preprocessor = FixedRewardPreprocessor(Preprocessor)
    sequence_buffer = ImprovedSequenceBuffer(max_size=2000, sequence_length=50)  # Improved buffer
    
    # Improved training parameters
    num_episodes = 1000      # Fewer episodes, better quality
    max_episode_length = 800  # Slightly shorter episodes
    batch_size = 16          # Larger batches for stability
    sequence_length = 50     # Longer sequences for better learning
    start_training_episodes = 20  # Start training sooner
    train_frequency = 3      # Train more frequently
    
    print(f"🎯 Training config:")
    print(f"   Episodes: {num_episodes}")
    print(f"   Max episode length: {max_episode_length}")
    print(f"   Batch size: {batch_size}")
    print(f"   Sequence length: {sequence_length}")
    
    total_steps = 0
    best_reward = float('-inf')
    
    # Tracking
    reward_window = deque(maxlen=100)
    episode_length_window = deque(maxlen=100)
    early_termination_count = 0
    
    for episode in range(num_episodes):
        # Reset environment and preprocessor
        obs, info = fixed_env.reset()
        obs = preprocessor.modify_state(obs, info).squeeze()
        
        episode_observations = [obs]
        episode_actions = []
        episode_rewards = []
        episode_reward = 0
        
        # Agent state
        agent_state = None
        
        # Episode loop
        for step in range(max_episode_length):
            # Action selection
            if len(sequence_buffer.buffer) < start_training_episodes:
                # Random exploration with safety constraints
                action = np.random.uniform(-0.5, 0.5, size=12)  # Smaller initial actions
            else:
                action, agent_state = model.select_action(obs, agent_state, deterministic=False)
                
                # Adaptive exploration
                if episode < num_episodes * 0.6:
                    exploration_rate = 0.1 * (1.0 - episode / (num_episodes * 0.6))
                    action += np.random.normal(0, exploration_rate, size=action.shape)
                    action = np.clip(action, -1, 1)
            
            # Convert to environment action
            env_action = env.action_space.low + (env.action_space.high - env.action_space.low) * (action + 1) / 2
            
            # Step environment (using fixed reward wrapper)
            next_obs, reward, terminated, truncated, next_info = fixed_env.step(env_action)
            next_obs = preprocessor.modify_state(next_obs, next_info).squeeze()
            
            # Store transition
            episode_observations.append(next_obs)
            episode_actions.append(action)
            episode_rewards.append(reward)
            episode_reward += reward
            total_steps += 1
            
            obs = next_obs
            
            if terminated or truncated:
                if terminated and step < 50:  # Early termination
                    early_termination_count += 1
                    print(f"⚠️  Early termination at step {step}, episode {episode}")
                break
        
        # Add to buffer
        if len(episode_actions) > 0:
            sequence_buffer.add_episode(episode_observations[:-1], episode_actions, episode_rewards)
        
        # Update tracking
        reward_window.append(episode_reward)
        episode_length_window.append(len(episode_actions))
        
        # Logging
        writer.add_scalar('Episode/Reward', episode_reward, episode)
        writer.add_scalar('Episode/Length', len(episode_actions), episode)
        writer.add_scalar('Episode/TotalSteps', total_steps, episode)
        writer.add_scalar('Episode/BufferSize', len(sequence_buffer.buffer), episode)
        writer.add_scalar('Episode/EarlyTerminations', early_termination_count, episode)
        
        if len(reward_window) > 0:
            writer.add_scalar('Average/Reward_100ep', np.mean(reward_window), episode)
            writer.add_scalar('Average/EpisodeLength_100ep', np.mean(episode_length_window), episode)
        
        # Console output
        avg_reward = np.mean(reward_window) if len(reward_window) > 0 else 0
        print(f"Episode {episode:4d}: Reward = {episode_reward:8.3f}, Steps = {len(episode_actions):3d}, "
              f"Avg100 = {avg_reward:8.3f}, Buffer = {len(sequence_buffer.buffer):4d}")
        
        # Track best reward
        if episode_reward > best_reward:
            best_reward = episode_reward
            writer.add_scalar('Episode/BestReward', best_reward, episode)
            print(f"  🎉 New best reward: {best_reward:.3f}")
        
        # Training
        if len(sequence_buffer.buffer) >= start_training_episodes and episode % train_frequency == 0:
            print(f"  🔄 Training model (buffer size: {len(sequence_buffer.buffer)})...")
            
            # Multiple training steps
            num_train_steps = 8
            total_losses = {
                'world_model_loss': 0,
                'reconstruction_loss': 0,
                'reward_loss': 0,
                'kl_loss': 0,
                'actor_loss': 0,
                'critic_loss': 0,
                'world_model_grad_norm': 0,
                'actor_grad_norm': 0,
                'critic_grad_norm': 0
            }
            
            for train_step in range(num_train_steps):
                obs_seq, action_seq, reward_seq = sequence_buffer.sample_sequences(batch_size)
                
                if obs_seq is not None:
                    # Move to device
                    obs_seq = obs_seq.to(device)
                    action_seq = action_seq.to(device)
                    reward_seq = reward_seq.to(device)
                    
                    # Training step
                    losses = model.train_step(obs_seq, action_seq, reward_seq)
                    
                    for key, value in losses.items():
                        total_losses[key] += value
            
            # Average losses
            for key in total_losses:
                total_losses[key] /= num_train_steps
            
            # Log training metrics
            for key, value in total_losses.items():
                writer.add_scalar(f'Loss/{key.replace("_", " ").title()}', value, episode)
            
            # Console training info
            print(f"    World Model Loss: {total_losses['world_model_loss']:.4f}")
            print(f"    Reconstruction:   {total_losses['reconstruction_loss']:.4f}")
            print(f"    Reward:          {total_losses['reward_loss']:.4f}")
            print(f"    KL:              {total_losses['kl_loss']:.4f}")
            print(f"    Actor:           {total_losses['actor_loss']:.4f}")
            print(f"    Critic:          {total_losses['critic_loss']:.4f}")
            
            # Check for improvement
            if total_losses['world_model_loss'] < 1.0:
                print(f"    ✅ World model loss converging!")
            if total_losses['reconstruction_loss'] < 0.5:
                print(f"    ✅ Reconstruction loss looking good!")
        
        # Save checkpoints
        if episode % 100 == 0 and episode > 0:
            checkpoint_path = f'saved_models/checkpoints/improved_dreamerv3_{episode}.pth'
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
            torch.save({
                'episode': episode,
                'model_state_dict': model.state_dict(),
                'best_reward': best_reward,
                'total_steps': total_steps
            }, checkpoint_path)
            print(f"  💾 Checkpoint saved: {checkpoint_path}")
    
    print("=== Training Complete ===")
    print(f"🎯 Best reward achieved: {best_reward:.3f}")
    print(f"📊 Early terminations: {early_termination_count}/{num_episodes} ({early_termination_count/num_episodes*100:.1f}%)")
    
    # Final save
    final_path = 'saved_models/best_models/improved_dreamerv3_final.pth'
    os.makedirs(os.path.dirname(final_path), exist_ok=True)
    torch.save(model.state_dict(), final_path)
    print(f"💾 Final model saved: {final_path}")
    
    writer.close()
    print(f"📊 TensorBoard logs: {log_dir}")
    
    return model, preprocessor


## Define action function (SAI-compatible name)
def action_function(policy):
    """Improved action function with better bounds handling"""
    expected_bounds = [-1, 1]
    action_percent = (policy - expected_bounds[0]) / (expected_bounds[1] - expected_bounds[0])
    bounded_percent = np.minimum(np.maximum(action_percent, 0), 1)
    return env.action_space.low + (env.action_space.high - env.action_space.low) * bounded_percent


if __name__ == "__main__":
    print("🚀 Starting Improved DreamerV3 Training...")
    print("=" * 60)
    
    # Test reward shaping first
    print("Testing reward shaping logic...")
    from reward_shaping_wrapper import test_reward_shaping
    test_reward_shaping()
    print("✅ Reward shaping test complete!\n")
    
    # Train the model
    model, preprocessor = improved_dreamerv3_training_loop()
    
    print("\n" + "=" * 60)
    print("🎯 Training complete! Starting evaluation...")
    
    # Create SAI-compatible wrapper
    print("Creating SAI-compatible model wrapper...")
    sai_model = SAICompatibleDreamerV3(model)
    print(f"✅ Model device: {next(sai_model.parameters()).device}")
    
    # Benchmark locally
    print("🔍 Running local benchmark...")
    sai.benchmark(sai_model, action_function, Preprocessor)
    
    # Submit to leaderboard
    print("🚀 Submitting to leaderboard...")
    sai.submit("Vedanta_ImprovedDreamerV3_Fixed", sai_model, action_function, Preprocessor)
    
    print("=" * 60)
    print("🎉 COMPLETE! Check the score improvement!")
    print("Expected improvements:")
    print("  - No more early termination (robot won't self-fall)")
    print("  - Positive episode rewards (step penalty fixed)")
    print("  - Better world model convergence")
    print("  - More natural robot movement")