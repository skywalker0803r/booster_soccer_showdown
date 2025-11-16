"""
Environment-Agnostic Training
Train on original environment but with better exploration and stability
No shaped rewards - focus on improving original environment performance
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
from sai_compatible_dreamerv3 import SAICompatibleDreamerV3
from main_improved_dreamerv3 import ImprovedSequenceBuffer

## Initialize the SAI client
sai = SAIClient(comp_id="booster-soccer-showdown", api_key="sai_LFcuaCZiqEkUbNVolQ3wbk5yU7H11jfv")
env = sai.make_env()

class Preprocessor():
    """Original preprocessor"""
    
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


class SmartExplorationWrapper:
    """
    Smart exploration without shaped rewards
    Focus on improving action quality and exploration strategy
    """
    
    def __init__(self, env, preprocessor):
        self.env = env
        self.preprocessor = preprocessor
        self.episode_step = 0
        self.episode_count = 0
        self.success_episodes = []
        
    def reset(self, **kwargs):
        self.episode_step = 0
        self.episode_count += 1
        return self.env.reset(**kwargs)
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.episode_step += 1
        
        # Log successful episodes for analysis
        if (terminated or truncated) and reward > -1.0:
            self.success_episodes.append({
                'episode': self.episode_count,
                'reward': reward,
                'steps': self.episode_step,
                'ball_dist': self._get_ball_distance(info)
            })
            print(f"🎉 SUCCESS Episode {self.episode_count}: R={reward:.3f}, Steps={self.episode_step}")
        
        return obs, reward, terminated, truncated, info
    
    def _get_ball_distance(self, info):
        """Get ball distance for analysis"""
        try:
            ball_pos = info.get("ball_xpos_rel_robot", np.zeros(3))
            if len(ball_pos.shape) > 1:
                ball_pos = ball_pos[0]
            return float(np.linalg.norm(ball_pos))
        except:
            return float('inf')
    
    def __getattr__(self, name):
        return getattr(self.env, name)


def environment_agnostic_training():
    """
    Training that works in both training and evaluation environments
    No shaped rewards - focus on original environment performance
    """
    
    print("🎯 ENVIRONMENT-AGNOSTIC DREAMERV3 TRAINING")
    print("="*60)
    print("🔍 Key insight: Training-evaluation environment mismatch!")
    print("🎯 New approach: Train on original environment")
    print("💡 Strategy: Better exploration + action quality")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Using device: {device}")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"runs/EnvironmentAgnosticDreamerV3_{timestamp}"
    writer = SummaryWriter(log_dir)
    print(f"📊 TensorBoard: {log_dir}")
    
    # Create exploration wrapper (no shaped rewards!)
    preprocessor = Preprocessor()
    smart_env = SmartExplorationWrapper(env, preprocessor)
    print("✅ Smart exploration wrapper (no shaped rewards)")
    
    # Initialize model with conservative settings
    model = ImprovedDreamerV3(
        obs_dim=89,
        action_dim=12,
        hidden_dim=512,  # Larger for better learning
        stoch_dim=32,
        discrete_dim=32
    ).to(device)
    
    sequence_buffer = ImprovedSequenceBuffer(max_size=3000, sequence_length=60)
    
    # Conservative training parameters
    num_episodes = 2000
    max_episode_length = 800
    batch_size = 12
    start_training_episodes = 30
    train_frequency = 4
    
    print(f"🎯 Training configuration:")
    print(f"   Episodes: {num_episodes}")
    print(f"   Original environment only")
    print(f"   Focus: Stable exploration")
    
    total_steps = 0
    best_reward = float('-inf')
    success_count = 0
    reward_window = deque(maxlen=100)
    
    # Action scaling schedule
    def get_action_scale(episode):
        """Conservative action scaling"""
        if episode < 200:
            return 0.3  # Very conservative start
        elif episode < 500:
            return 0.5  # Moderate
        elif episode < 1000:
            return 0.7  # More aggressive
        else:
            return 0.8  # Full scale
    
    for episode in range(num_episodes):
        obs, info = smart_env.reset()
        obs = preprocessor.modify_state(obs, info).squeeze()
        
        episode_observations = [obs]
        episode_actions = []
        episode_rewards = []
        episode_reward = 0
        
        agent_state = None
        action_scale = get_action_scale(episode)
        
        for step in range(max_episode_length):
            if len(sequence_buffer.buffer) < start_training_episodes:
                # Random exploration with curriculum
                if episode < 50:
                    # Very conservative random actions
                    action = np.random.uniform(-0.2, 0.2, size=12)
                else:
                    # Gradually increase exploration
                    max_action = min(0.5, 0.1 + episode * 0.01)
                    action = np.random.uniform(-max_action, max_action, size=12)
            else:
                # Model-based action
                action, agent_state = model.select_action(obs, agent_state, deterministic=False)
                
                # Smart exploration schedule
                if episode < num_episodes * 0.3:
                    noise_scale = 0.1 * (1.0 - episode / (num_episodes * 0.3))
                elif episode < num_episodes * 0.6:
                    noise_scale = 0.05
                else:
                    noise_scale = 0.02  # Very focused exploitation
                
                action += np.random.normal(0, noise_scale, size=action.shape)
                action = np.clip(action, -1, 1)
            
            # Conservative action scaling
            action = action * action_scale
            
            # Convert to environment action
            env_action = env.action_space.low + (env.action_space.high - env.action_space.low) * (action + 1) / 2
            
            # Step environment (no shaped rewards!)
            next_obs, reward, terminated, truncated, next_info = smart_env.step(env_action)
            next_obs = preprocessor.modify_state(next_obs, next_info).squeeze()
            
            episode_observations.append(next_obs)
            episode_actions.append(action)
            episode_rewards.append(reward)
            episode_reward += reward
            total_steps += 1
            
            obs = next_obs
            
            if terminated or truncated:
                break
        
        # Track successes (any positive reward)
        if episode_reward > -1.0:  # Better than pure step penalty
            success_count += 1
        
        # Add to buffer
        if len(episode_actions) > 0:
            sequence_buffer.add_episode(episode_observations[:-1], episode_actions, episode_rewards)
        
        reward_window.append(episode_reward)
        
        # Logging
        writer.add_scalar('Episode/Reward', episode_reward, episode)
        writer.add_scalar('Episode/Length', len(episode_actions), episode)
        writer.add_scalar('Episode/SuccessCount', success_count, episode)
        writer.add_scalar('Episode/ActionScale', action_scale, episode)
        
        avg_reward = np.mean(reward_window) if len(reward_window) > 0 else 0
        success_rate = success_count / (episode + 1) * 100
        
        print(f"Ep {episode:4d}: R={episode_reward:8.3f}, L={len(episode_actions):3d}, "
              f"Avg={avg_reward:8.3f}, Success={success_count:3d} ({success_rate:5.1f}%), "
              f"Scale={action_scale:.1f}")
        
        if episode_reward > best_reward:
            best_reward = episode_reward
            print(f"  🎉 New best reward: {best_reward:.3f}")
        
        # Training
        if len(sequence_buffer.buffer) >= start_training_episodes and episode % train_frequency == 0:
            print(f"  🔄 Training model...")
            
            total_losses = {}
            num_train_steps = 8
            
            for _ in range(num_train_steps):
                obs_seq, action_seq, reward_seq = sequence_buffer.sample_sequences(batch_size)
                
                if obs_seq is not None:
                    obs_seq = obs_seq.to(device)
                    action_seq = action_seq.to(device)
                    reward_seq = reward_seq.to(device)
                    
                    losses = model.train_step(obs_seq, action_seq, reward_seq)
                    
                    for key, value in losses.items():
                        if key not in total_losses:
                            total_losses[key] = 0
                        total_losses[key] += value
            
            # Log losses
            for key in total_losses:
                total_losses[key] /= num_train_steps
                writer.add_scalar(f'Loss/{key}', total_losses[key], episode)
        
        # Save checkpoints
        if episode % 200 == 0 and episode > 0:
            checkpoint_path = f'saved_models/checkpoints/env_agnostic_{episode}.pth'
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
            torch.save(model.state_dict(), checkpoint_path)
            
            print(f"  📊 Checkpoint {episode}: Success {success_rate:.1f}%, Best {best_reward:.3f}")
    
    print(f"\n🏆 TRAINING COMPLETE!")
    print(f"   Best reward: {best_reward:.3f}")
    print(f"   Success rate: {success_rate:.1f}%")
    print(f"   Successful episodes: {len(smart_env.success_episodes)}")
    
    # Analyze successful episodes
    if smart_env.success_episodes:
        print(f"\n🎯 Success Analysis:")
        success_rewards = [ep['reward'] for ep in smart_env.success_episodes]
        print(f"   Average success reward: {np.mean(success_rewards):.3f}")
        print(f"   Best success reward: {max(success_rewards):.3f}")
    
    final_path = 'saved_models/best_models/env_agnostic_final.pth'
    os.makedirs(os.path.dirname(final_path), exist_ok=True)
    torch.save(model.state_dict(), final_path)
    
    writer.close()
    return model, preprocessor


def action_function(policy):
    """Conservative action function"""
    expected_bounds = [-1, 1]
    action_percent = (policy - expected_bounds[0]) / (expected_bounds[1] - expected_bounds[0])
    bounded_percent = np.minimum(np.maximum(action_percent, 0), 1)
    return env.action_space.low + (env.action_space.high - env.action_space.low) * bounded_percent


if __name__ == "__main__":
    print("🚀 Environment-Agnostic Training")
    print("Problem: Training environment ≠ Evaluation environment")
    print("Solution: Train on original environment with smart exploration\n")
    
    model, preprocessor = environment_agnostic_training()
    
    print("\n🎯 Testing trained model...")
    sai_model = SAICompatibleDreamerV3(model)
    
    print("🔍 Local benchmark...")
    sai.benchmark(sai_model, action_function, Preprocessor)
    
    print("🚀 Submitting to leaderboard...")
    sai.submit("Vedanta_EnvironmentAgnostic_DreamerV3", sai_model, action_function, Preprocessor)
    
    print("🎉 Expected: Training and evaluation scores should match!")