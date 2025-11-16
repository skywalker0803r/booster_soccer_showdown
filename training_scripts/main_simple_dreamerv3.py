import torch
import torch.nn.functional as F
import numpy as np
from collections import deque
import random
from torch.utils.tensorboard import SummaryWriter
import os
from datetime import datetime

from sai_rl import SAIClient
from simple_dreamerv3 import SimpleDreamerV3
from sai_compatible_dreamerv3 import SAICompatibleDreamerV3

## Initialize the SAI client
sai = SAIClient(comp_id="booster-soccer-showdown", api_key="sai_LFcuaCZiqEkUbNVolQ3wbk5yU7H11jfv")

## Make the environment
env = sai.make_env()

class Preprocessor():

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


class SequenceBuffer:
    """Buffer to store sequences for DreamerV3 training"""
    
    def __init__(self, max_size=50000, sequence_length=50):
        self.max_size = max_size
        self.sequence_length = sequence_length
        self.buffer = deque(maxlen=max_size)
        
    def add_episode(self, observations, actions, rewards):
        """Add a complete episode to buffer"""
        episode = {
            'observations': np.array(observations),
            'actions': np.array(actions),
            'rewards': np.array(rewards)
        }
        self.buffer.append(episode)
        
    def sample_sequences(self, batch_size):
        """Sample random sequences for training"""
        if len(self.buffer) == 0:
            return None, None, None
            
        batch_obs = []
        batch_actions = []
        batch_rewards = []
        
        for _ in range(batch_size):
            # Sample random episode
            episode = random.choice(self.buffer)
            
            # Sample random sequence from episode
            episode_length = len(episode['observations'])
            if episode_length < self.sequence_length:
                # Pad short episodes
                start_idx = 0
                seq_len = episode_length
                
                obs_seq = episode['observations']
                act_seq = episode['actions']
                rew_seq = episode['rewards']
                
                # Pad with last observation/action
                if seq_len < self.sequence_length:
                    pad_len = self.sequence_length - seq_len
                    obs_pad = np.repeat(obs_seq[-1:], pad_len, axis=0)
                    act_pad = np.repeat(act_seq[-1:], pad_len, axis=0)
                    rew_pad = np.zeros(pad_len)
                    
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


def dreamerv3_training_loop():
    """Training loop for SimpleDreamerV3"""
    
    print("=== Starting DreamerV3 Training ===")
    
    # Check for GPU availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔥 Using device: {device}")
    if device.type == "cuda":
        print(f"   GPU Name: {torch.cuda.get_device_name(0)}")
        print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # Initialize TensorBoard writer
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"runs/SimpleDreamerV3_{timestamp}"
    writer = SummaryWriter(log_dir)
    print(f"📊 TensorBoard logs will be saved to: {log_dir}")
    print(f"🚀 To view logs, run: tensorboard --logdir=runs --port=6006")
    
    # Initialize model and move to device
    model = SimpleDreamerV3(
        obs_dim=89,  # preprocessed observation dimension
        action_dim=12,  # robot joint actions
        hidden_dim=256,  # Smaller for faster training
        stoch_dim=32,
        discrete_dim=16
    ).to(device)
    
    preprocessor = Preprocessor()
    sequence_buffer = SequenceBuffer(max_size=1000, sequence_length=25)
    
    # Training parameters
    num_episodes = 80 #正式再調大例如800
    max_episode_length = 1000
    batch_size = 8
    sequence_length = 25
    start_training_episodes = 10  # Start training after collecting some data
    train_frequency = 5  # Train every N episodes
    
    print(f"Training for {num_episodes} episodes...")
    
    total_steps = 0
    best_reward = float('-inf')
    
    # Running averages for monitoring
    reward_window = deque(maxlen=100)
    episode_length_window = deque(maxlen=100)
    exploration_rate = 0.0
    
    for episode in range(num_episodes):
        # Collect episode
        obs, info = env.reset()
        obs = preprocessor.modify_state(obs, info).squeeze()
        
        episode_observations = [obs]
        episode_actions = []
        episode_rewards = []
        episode_reward = 0
        
        # Agent state for consistent action selection
        agent_state = None
        
        for step in range(max_episode_length):
            # Select action
            if len(sequence_buffer.buffer) < start_training_episodes:
                # Random actions during initial data collection
                action = np.random.uniform(-1, 1, size=12)
            else:
                action, agent_state = model.select_action(obs, agent_state)
                
                # Add exploration noise
                if episode < num_episodes * 0.7:  # Reduce exploration over time
                    exploration_rate = 0.3 * (1.0 - episode / (num_episodes * 0.7))
                    action += np.random.normal(0, exploration_rate, size=action.shape)
                    action = np.clip(action, -1, 1)
                else:
                    exploration_rate = 0.0
            
            # Convert normalized action to environment action
            env_action = env.action_space.low + (env.action_space.high - env.action_space.low) * (action + 1) / 2
            
            # Step environment
            next_obs, reward, terminated, truncated, next_info = env.step(env_action)
            next_obs = preprocessor.modify_state(next_obs, next_info).squeeze()
            
            # Store transition
            episode_observations.append(next_obs)
            episode_actions.append(action)
            episode_rewards.append(reward)
            episode_reward += reward
            total_steps += 1
            
            obs = next_obs
            
            if terminated or truncated:
                break
        
        # Add episode to buffer
        if len(episode_actions) > 0:
            sequence_buffer.add_episode(episode_observations[:-1], episode_actions, episode_rewards)
        
        # Update running averages
        reward_window.append(episode_reward)
        episode_length_window.append(len(episode_actions))
        
        # Log to TensorBoard
        writer.add_scalar('Episode/Reward', episode_reward, episode)
        writer.add_scalar('Episode/Length', len(episode_actions), episode)
        writer.add_scalar('Episode/TotalSteps', total_steps, episode)
        writer.add_scalar('Episode/BufferSize', len(sequence_buffer.buffer), episode)
        writer.add_scalar('Episode/ExplorationRate', exploration_rate, episode)
        
        # Log running averages
        if len(reward_window) > 0:
            writer.add_scalar('Average/Reward_100ep', np.mean(reward_window), episode)
            writer.add_scalar('Average/EpisodeLength_100ep', np.mean(episode_length_window), episode)
        
        print(f"Episode {episode}: Reward = {episode_reward:.3f}, Steps = {len(episode_actions)}, Buffer = {len(sequence_buffer.buffer)}")
        
        # Update best reward
        if episode_reward > best_reward:
            best_reward = episode_reward
            writer.add_scalar('Episode/BestReward', best_reward, episode)
            print(f"  New best reward: {best_reward:.3f}")
        
        # Training step
        if len(sequence_buffer.buffer) >= start_training_episodes and episode % train_frequency == 0:
            print(f"  Training model...")
            
            # Multiple training steps per episode
            num_train_steps = 10
            total_losses = {
                'world_model_loss': 0,
                'reconstruction_loss': 0,
                'reward_loss': 0,
                'kl_loss': 0,
                'actor_loss': 0,
                'critic_loss': 0
            }
            
            for train_step in range(num_train_steps):
                # Sample sequences
                obs_seq, action_seq, reward_seq = sequence_buffer.sample_sequences(batch_size)
                
                if obs_seq is not None:
                    # Move tensors to device
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
            
            # Log training losses to TensorBoard
            writer.add_scalar('Loss/WorldModel', total_losses['world_model_loss'], episode)
            writer.add_scalar('Loss/Reconstruction', total_losses['reconstruction_loss'], episode)
            writer.add_scalar('Loss/Reward', total_losses['reward_loss'], episode)
            writer.add_scalar('Loss/KL_Divergence', total_losses['kl_loss'], episode)
            writer.add_scalar('Loss/Actor', total_losses['actor_loss'], episode)
            writer.add_scalar('Loss/Critic', total_losses['critic_loss'], episode)
            
            # Log total loss
            total_loss = sum(total_losses.values())
            writer.add_scalar('Loss/Total', total_loss, episode)
            
            print(f"  Training losses:")
            print(f"    World Model: {total_losses['world_model_loss']:.4f}")
            print(f"    Reconstruction: {total_losses['reconstruction_loss']:.4f}")
            print(f"    Reward: {total_losses['reward_loss']:.4f}")
            print(f"    KL: {total_losses['kl_loss']:.4f}")
            print(f"    Actor: {total_losses['actor_loss']:.4f}")
            print(f"    Critic: {total_losses['critic_loss']:.4f}")
            print(f"    Total: {total_loss:.4f}")
        
        # Save model periodically
        if episode % 100 == 0 and episode > 0:
            torch.save(model.state_dict(), f'dreamerv3_checkpoint_{episode}.pth')
            print(f"  Saved checkpoint at episode {episode}")
    
    print("=== Training Complete ===")
    print(f"Best reward achieved: {best_reward:.3f}")
    
    # Close TensorBoard writer
    writer.close()
    print(f"📊 TensorBoard logs saved to: {log_dir}")
    
    return model, preprocessor


## Create and train the model
print("Creating SimpleDreamerV3 model...")
model, preprocessor = dreamerv3_training_loop()

## Define an action function - exactly like your main.py
def action_function(policy):
    expected_bounds = [-1, 1]
    action_percent = (policy - expected_bounds[0]) / (
        expected_bounds[1] - expected_bounds[0]
    )
    bounded_percent = np.minimum(np.maximum(action_percent, 0), 1)
    return (
        env.action_space.low
        + (env.action_space.high - env.action_space.low) * bounded_percent
    )

print("=== Starting Evaluation ===")

# 創建SAI兼容的包裝器 - 確保完全CPU兼容
print("Creating SAI-compatible model wrapper...")
sai_model = SAICompatibleDreamerV3(model)

print("Model successfully wrapped for SAI compatibility.")
print(f"Model device: {next(sai_model.parameters()).device}")

## Benchmark the model locally - 使用SAI兼容包裝器
sai.benchmark(sai_model, action_function, Preprocessor)

## Submit to leaderboard
print("=== Submitting to Leaderboard ===")
sai.submit("Vedanta_SimpleDreamerV3", sai_model, action_function, Preprocessor)

print("=== Complete ===")