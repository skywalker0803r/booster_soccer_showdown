"""
Success-Focused DreamerV3 Training
Based on NEW understanding: Your agent CAN succeed (+25.418 proves it!)
Problem: Inconsistent performance, not step penalty
Solution: Dense rewards + better training stability
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


class SuccessFocusedWrapper:
    """
    Environment wrapper focused on improving success rate
    Provides dense rewards and stability guidance
    """
    
    def __init__(self, env, preprocessor):
        self.env = env
        self.preprocessor = preprocessor
        self.episode_step = 0
        self.best_ball_dist = float('inf')
        self.stability_score = 0
        self.episode_rewards = []
        self.success_count = 0
        
    def reset(self, **kwargs):
        self.episode_step = 0
        self.best_ball_dist = float('inf')
        self.stability_score = 0
        self.episode_rewards = []
        return self.env.reset(**kwargs)
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.episode_step += 1
        
        # Dense shaping rewards to guide learning
        shaping_reward = 0
        
        # Extract state information
        ball_pos = info.get("ball_xpos_rel_robot", np.zeros(3))
        if len(ball_pos.shape) > 1:
            ball_pos = ball_pos[0]
        ball_dist = np.linalg.norm(ball_pos)
        
        robot_quat = info.get("robot_quat", np.array([0, 0, 0, 1]))
        if len(robot_quat.shape) > 1:
            robot_quat = robot_quat[0]
        robot_upright = 1.0 - abs(robot_quat[2])  # 1.0 = perfectly upright
        
        robot_vel = info.get("robot_velocimeter", np.zeros(3))
        if len(robot_vel.shape) > 1:
            robot_vel = robot_vel[0]
        movement_speed = np.linalg.norm(robot_vel)
        
        ball_vel = info.get("ball_velp_rel_robot", np.zeros(3))
        if len(ball_vel.shape) > 1:
            ball_vel = ball_vel[0]
        ball_speed = np.linalg.norm(ball_vel)
        
        goal_pos = info.get("goal_team_1_rel_robot", np.zeros(3))
        if len(goal_pos.shape) > 1:
            goal_pos = goal_pos[0]
        goal_dist = np.linalg.norm(goal_pos)
        
        # 1. Fundamental Stability Reward (CRITICAL)
        if robot_upright > 0.9:
            stability_reward = 0.02
            self.stability_score += 1
        elif robot_upright > 0.8:
            stability_reward = 0.01
        elif robot_upright > 0.7:
            stability_reward = 0
        else:
            stability_reward = -0.05  # Falling penalty
            self.stability_score = max(0, self.stability_score - 2)
        
        shaping_reward += stability_reward
        
        # 2. Ball Approach Reward (MAIN OBJECTIVE)
        if robot_upright > 0.8:  # Only if stable
            # Progress toward ball
            if ball_dist < self.best_ball_dist - 0.01:  # Made progress
                progress_reward = min((self.best_ball_dist - ball_dist) * 10.0, 0.5)
                shaping_reward += progress_reward
                self.best_ball_dist = ball_dist
            
            # Distance-based rewards
            if ball_dist < 0.3:
                shaping_reward += 0.1  # Very close to ball
            elif ball_dist < 0.6:
                shaping_reward += 0.05  # Close to ball
            elif ball_dist < 1.0:
                shaping_reward += 0.02  # Approaching ball
        
        # 3. Controlled Movement Reward
        if robot_upright > 0.8:
            if 0.2 < movement_speed < 1.5:  # Good controlled movement
                shaping_reward += 0.01
            elif movement_speed > 3.0:  # Too fast/erratic
                shaping_reward -= 0.02
        
        # 4. Ball Contact and Kicking Rewards
        if ball_dist < 0.4 and robot_upright > 0.8:
            # Reward ball contact
            if ball_speed > 0.5:  # Ball is moving
                # Check if ball is moving toward goal
                if goal_dist > 0.1:
                    goal_direction = goal_pos / goal_dist
                    ball_direction = ball_vel / (ball_speed + 1e-8)
                    goal_alignment = np.dot(goal_direction, ball_direction)
                    
                    if goal_alignment > 0.5:  # Good direction
                        kick_reward = min(ball_speed * goal_alignment * 2.0, 1.0)
                        shaping_reward += kick_reward
        
        # 5. Persistence Reward (Surviving longer episodes)
        if self.episode_step > 100 and robot_upright > 0.7:
            persistence_reward = 0.005
            shaping_reward += persistence_reward
        
        # 6. Success Detection and Bonus
        episode_ending = terminated or truncated
        if episode_ending:
            # Give final episode assessment
            episode_bonus = 0
            
            # Survival bonus
            if self.episode_step > 150:
                episode_bonus += 1.0
            if self.episode_step > 300:
                episode_bonus += 2.0
            
            # Stability bonus
            if self.stability_score > self.episode_step * 0.7:
                episode_bonus += 1.0
            
            # Ball interaction bonus
            if self.best_ball_dist < 1.0:
                episode_bonus += 1.0
            if self.best_ball_dist < 0.5:
                episode_bonus += 2.0
            
            # Success detection
            if reward > 0:  # Original positive reward
                episode_bonus += 5.0
                self.success_count += 1
                print(f"🎉 SUCCESS! Episode reward: {reward:.3f} + bonus: {episode_bonus:.3f}")
            
            shaping_reward += episode_bonus
            total_episode_reward = reward + sum(self.episode_rewards) + shaping_reward
            
            print(f"Episode {self.episode_step:3d} steps: Original={reward:8.3f}, "
                  f"Shaped={sum(self.episode_rewards) + shaping_reward:8.3f}, "
                  f"Total={total_episode_reward:8.3f}, Best_ball_dist={self.best_ball_dist:.3f}")
            
            return obs, total_episode_reward, terminated, truncated, info
        else:
            # Store shaping reward for episode
            self.episode_rewards.append(shaping_reward)
            return obs, shaping_reward, terminated, truncated, info
    
    def __getattr__(self, name):
        return getattr(self.env, name)


def success_focused_training():
    """Training focused on improving success rate"""
    
    print("🎯 SUCCESS-FOCUSED DREAMERV3 TRAINING")
    print("="*60)
    print("🔍 Key insight: Your agent CAN succeed (max +25.418)!")
    print("🎯 New goal: Improve success rate and consistency")
    print("📈 Strategy: Dense rewards + stability focus")
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Using device: {device}")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"runs/SuccessFocusedDreamerV3_{timestamp}"
    writer = SummaryWriter(log_dir)
    print(f"📊 TensorBoard: {log_dir}")
    
    # Create success-focused environment
    preprocessor = Preprocessor()
    success_env = SuccessFocusedWrapper(env, preprocessor)
    print("✅ Success-focused wrapper activated")
    
    # Initialize improved model
    model = ImprovedDreamerV3(
        obs_dim=89,
        action_dim=12,
        hidden_dim=256,
        stoch_dim=32,
        discrete_dim=16
    ).to(device)
    
    # Improved sequence buffer from previous training
    from improved_dreamerv3 import ImprovedSequenceBuffer
    sequence_buffer = ImprovedSequenceBuffer(max_size=2000, sequence_length=50)
    
    # Optimized training parameters for success
    num_episodes = 1000
    max_episode_length = 600  # Allow longer episodes for success
    batch_size = 16
    start_training_episodes = 15
    train_frequency = 2
    
    print(f"🎯 Training configuration:")
    print(f"   Episodes: {num_episodes}")
    print(f"   Max length: {max_episode_length}")
    print(f"   Focus: Success rate improvement")
    
    # Tracking
    total_steps = 0
    best_reward = float('-inf')
    success_count = 0
    reward_window = deque(maxlen=100)
    
    for episode in range(num_episodes):
        obs, info = success_env.reset()
        obs = preprocessor.modify_state(obs, info).squeeze()
        
        episode_observations = [obs]
        episode_actions = []
        episode_rewards = []
        episode_reward = 0
        
        agent_state = None
        
        # Adaptive action strategy
        for step in range(max_episode_length):
            if len(sequence_buffer.buffer) < start_training_episodes:
                # Conservative exploration initially
                action = np.random.uniform(-0.3, 0.3, size=12)
            else:
                # Model-based action with reduced noise
                action, agent_state = model.select_action(obs, agent_state, deterministic=False)
                
                # Conservative exploration schedule
                if episode < num_episodes * 0.4:
                    exploration_noise = 0.05 * (1.0 - episode / (num_episodes * 0.4))
                    action += np.random.normal(0, exploration_noise, size=action.shape)
                    action = np.clip(action, -0.8, 0.8)  # Conservative action bounds
            
            # Convert to environment action with safety limits
            env_action = env.action_space.low + (env.action_space.high - env.action_space.low) * (action + 1) / 2
            env_action = np.clip(env_action, env.action_space.low * 0.7, env.action_space.high * 0.7)
            
            # Step environment
            next_obs, reward, terminated, truncated, next_info = success_env.step(env_action)
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
        
        # Track successes
        if episode_reward > 0:
            success_count += 1
        
        # Add to buffer
        if len(episode_actions) > 0:
            sequence_buffer.add_episode(episode_observations[:-1], episode_actions, episode_rewards)
        
        reward_window.append(episode_reward)
        
        # Logging
        writer.add_scalar('Episode/Reward', episode_reward, episode)
        writer.add_scalar('Episode/Length', len(episode_actions), episode)
        writer.add_scalar('Episode/SuccessCount', success_count, episode)
        writer.add_scalar('Episode/SuccessRate', success_count / (episode + 1), episode)
        
        avg_reward = np.mean(reward_window) if len(reward_window) > 0 else 0
        success_rate = success_count / (episode + 1) * 100
        
        print(f"Ep {episode:4d}: R={episode_reward:8.3f}, L={len(episode_actions):3d}, "
              f"Avg={avg_reward:8.3f}, Success={success_count:3d} ({success_rate:5.1f}%)")
        
        # Track best performance
        if episode_reward > best_reward:
            best_reward = episode_reward
            print(f"  🎉 New best reward: {best_reward:.3f}")
        
        # Training
        if len(sequence_buffer.buffer) >= start_training_episodes and episode % train_frequency == 0:
            print(f"  🔄 Training...")
            
            total_losses = {}
            num_train_steps = 6
            
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
            
            # Average and log losses
            for key in total_losses:
                total_losses[key] /= num_train_steps
                writer.add_scalar(f'Loss/{key}', total_losses[key], episode)
        
        # Progress checkpoints
        if episode % 100 == 0 and episode > 0:
            print(f"  📊 Checkpoint {episode}: Success rate {success_rate:.1f}%, Best reward {best_reward:.3f}")
            
            checkpoint_path = f'saved_models/checkpoints/success_focused_{episode}.pth'
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
            torch.save(model.state_dict(), checkpoint_path)
    
    print(f"\n🏆 TRAINING COMPLETE!")
    print(f"   Best reward: {best_reward:.3f}")
    print(f"   Success count: {success_count}/{num_episodes}")
    print(f"   Success rate: {success_count/num_episodes*100:.1f}%")
    
    # Final save
    final_path = 'saved_models/best_models/success_focused_final.pth'
    os.makedirs(os.path.dirname(final_path), exist_ok=True)
    torch.save(model.state_dict(), final_path)
    
    writer.close()
    return model, preprocessor


def action_function(policy):
    """Conservative action function for better stability"""
    expected_bounds = [-1, 1]
    action_percent = (policy - expected_bounds[0]) / (expected_bounds[1] - expected_bounds[0])
    bounded_percent = np.minimum(np.maximum(action_percent, 0), 1)
    
    # Apply conservative scaling for stability
    conservative_action = env.action_space.low + (env.action_space.high - env.action_space.low) * bounded_percent
    return conservative_action * 0.8  # 80% of max action for stability


if __name__ == "__main__":
    print("🚀 Starting Success-Focused Training")
    print("Key insight: Step penalty is NOT the problem!")
    print("Your agent achieved +25.418 - it CAN succeed!")
    print("Focus: Improve success rate and reduce variance\n")
    
    model, preprocessor = success_focused_training()
    
    print("\n🎯 Creating SAI submission...")
    sai_model = SAICompatibleDreamerV3(model)
    
    print("🔍 Local benchmark...")
    sai.benchmark(sai_model, action_function, Preprocessor)
    
    print("🚀 Submitting to leaderboard...")
    sai.submit("Vedanta_SuccessFocused_DreamerV3", sai_model, action_function, Preprocessor)
    
    print("🎉 DONE! Expected improvements:")
    print("   - Higher success rate")
    print("   - More consistent positive rewards")
    print("   - Better stability and ball interaction")
    print("   - Score should improve significantly!")