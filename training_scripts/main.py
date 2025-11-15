import torch
import torch.nn.functional as F
import numpy as np
from collections import deque
import random

from sai_rl import SAIClient
from simple_dreamerv3 import SimpleDreamerV3

## Initialize the SAI client
sai = SAIClient(comp_id="booster-soccer-showdown",api_key="sai_LFcuaCZiqEkUbNVolQ3wbk5yU7H11jfv")

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
    
    def __init__(self, max_size=1000, sequence_length=25):
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

## Create the SimpleDreamerV3 model
print("Creating SimpleDreamerV3 model...")
model = SimpleDreamerV3(
    obs_dim=89,  # preprocessed observation dimension
    action_dim=12,  # robot joint actions
    hidden_dim=256,  # Smaller for faster training
    stoch_dim=32,
    discrete_dim=16
)

## Define an action function
def action_function(policy):
    # DreamerV3 already outputs actions in [-1, 1] range
    # We need to map them to the environment's action space
    # Clip to ensure we're within [-1, 1]
    clipped_policy = np.clip(policy, -1, 1)
    
    # Map from [-1, 1] to [action_space.low, action_space.high]
    action_percent = (clipped_policy + 1) / 2  # Convert [-1, 1] to [0, 1]
    return (
        env.action_space.low
        + (env.action_space.high - env.action_space.low) * action_percent
    )


def dreamerv3_training_loop():
    """Training loop for SimpleDreamerV3"""
    
    print("=== Starting DreamerV3 Training ===")
    
    preprocessor = Preprocessor()
    sequence_buffer = SequenceBuffer(max_size=1000, sequence_length=25)
    
    # Training parameters
    num_episodes = 600
    max_episode_length = 1000
    batch_size = 8
    start_training_episodes = 10
    train_frequency = 5
    
    print(f"Training for {num_episodes} episodes...")
    
    best_reward = float('-inf')
    
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
                if episode < num_episodes * 0.7:
                    noise_scale = 0.3 * (1.0 - episode / (num_episodes * 0.7))
                    action += np.random.normal(0, noise_scale, size=action.shape)
                    action = np.clip(action, -1, 1)
            
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
            
            obs = next_obs
            
            if terminated or truncated:
                break
        
        # Add episode to buffer
        if len(episode_actions) > 0:
            sequence_buffer.add_episode(episode_observations[:-1], episode_actions, episode_rewards)
        
        if episode % 20 == 0:
            print(f"Episode {episode}: Reward = {episode_reward:.3f}, Steps = {len(episode_actions)}")
        
        # Update best reward
        if episode_reward > best_reward:
            best_reward = episode_reward
            if episode % 20 == 0:
                print(f"  New best reward: {best_reward:.3f}")
        
        # Training step
        if len(sequence_buffer.buffer) >= start_training_episodes and episode % train_frequency == 0:
            # Training steps
            num_train_steps = 8
            for train_step in range(num_train_steps):
                # Sample sequences
                obs_seq, action_seq, reward_seq = sequence_buffer.sample_sequences(batch_size)
                
                if obs_seq is not None:
                    # Training step
                    losses = model.train_step(obs_seq, action_seq, reward_seq)
            
            if episode % 50 == 0:
                print(f"  Latest training losses: WM={losses['world_model_loss']:.3f}, Actor={losses['actor_loss']:.3f}, Critic={losses['critic_loss']:.3f}")
        
        # Save model periodically
        if episode % 100 == 0 and episode > 0:
            torch.save(model.state_dict(), f'dreamerv3_checkpoint_{episode}.pth')
            print(f"  Saved checkpoint at episode {episode}")
    
    print("=== Training Complete ===")
    print(f"Best reward achieved: {best_reward:.3f}")
    
    return model

## Train the model
trained_model = dreamerv3_training_loop()

# Create SAI-compatible wrapper
class SAICompatibleWrapper:
    def __init__(self, dreamer_model):
        self.dreamer_model = dreamer_model
        self.state = None
        self.preprocessor = Preprocessor()
    
    def __call__(self, obs_tensor):
        # Convert to numpy
        if hasattr(obs_tensor, 'detach'):
            obs = obs_tensor.detach().cpu().numpy()
        else:
            obs = obs_tensor
        
        # Handle batch dimension
        if len(obs.shape) == 1:
            obs = obs[np.newaxis, :]
        
        # Reset state for each new sequence (SAI evaluation pattern)
        if self.state is None:
            self.state = None
        
        # Get action
        action, self.state = self.dreamer_model.select_action(obs[0], self.state)
        
        # Return as tensor
        return torch.FloatTensor(action).unsqueeze(0)
    
    def select_action(self, obs):
        """DDPG-style interface"""
        if hasattr(obs, 'detach'):
            obs = obs.detach().cpu().numpy()
        action, _ = self.dreamer_model.select_action(obs)
        return action

# Wrap the model
model = SAICompatibleWrapper(trained_model)

## Watch
#sai.watch(model, action_function, Preprocessor)

## Benchmark the model locally
sai.benchmark(model, action_function, Preprocessor)

sai.submit("Vedanta_DreamerV3", model, action_function, Preprocessor)
