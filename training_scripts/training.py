from typing import Optional, Callable
import random
from collections import deque

import gymnasium as gym
import numpy as np
import torch
from tqdm import tqdm


class ReplayBuffer:
    def __init__(self, max_size=100000):
        self.buffer = deque(maxlen=max_size)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones),
        )

    def __len__(self):
        return len(self.buffer)


def add_noise(action, noise_scale=0.1, episode_count=0, total_episodes=1000):
    # For sparse reward environments, maintain higher exploration longer
    decay_factor = max(0.2, 1.0 - episode_count / (total_episodes * 1.5))  # Slower decay
    effective_noise = noise_scale * decay_factor
    
    # Add structured exploration for soccer - vary leg coordination
    structured_noise = np.random.normal(0, effective_noise * 0.5, size=action.shape)
    random_noise = np.random.normal(0, effective_noise * 0.5, size=action.shape)
    
    total_noise = structured_noise + random_noise
    return np.clip(action + total_noise, -1, 1)


def training_loop(
    env: gym.Env,
    model,
    action_function: Optional[Callable] = None,
    preprocess_class: Optional[Callable] = None,
    timesteps=1000,
):
    replay_buffer = ReplayBuffer(max_size=200000)  # Larger buffer for more diverse experiences
    preprocessor = preprocess_class()
    batch_size = 128  # Larger batch size for more stable gradients
    update_frequency = 2  # More frequent updates for faster learning
    
    # Check if preprocessor supports reward shaping
    use_reward_shaping = hasattr(preprocessor, 'shape_reward')

    total_steps = 0
    episode_count = 0
    total_reward = 0
    total_original_reward = 0  # Track original reward separately
    
    # Best model tracking
    best_original_reward = float('-inf')
    best_model_state = None
    episodes_since_best = 0
    best_episode = 0

    pbar = tqdm(total=timesteps, desc="Training Progress", unit="steps")

    while total_steps < timesteps:
        done = False
        s, info = env.reset()
        s = preprocessor.modify_state(s, info).squeeze()
        episode_reward = 0
        episode_original_reward = 0  # Track original reward for this episode
        episode_steps = 0
        
        # Reset reward shaper if available
        if use_reward_shaping and hasattr(preprocessor, 'reset_episode'):
            preprocessor.reset_episode()

        while not done and total_steps < timesteps:
            state = torch.from_numpy(np.expand_dims(s, axis=0))
            policy = model(state).detach().numpy()

            if action_function:
                action = action_function(policy)[0].squeeze()
                # Use higher noise for sparse reward environment
                action = add_noise(action, noise_scale=0.4, episode_count=episode_count, total_episodes=timesteps//20)
            else:
                action = model.select_action(s)[0].squeeze()

            new_s, r, terminated, truncated, info = env.step(action)
            new_s = preprocessor.modify_state(new_s, info).squeeze()

            done = terminated or truncated
            
            # Apply reward shaping if available
            if use_reward_shaping:
                shaped_r = preprocessor.shape_reward(new_s, info, r, done)
            else:
                shaped_r = r

            episode_reward += shaped_r
            episode_original_reward += r  # Always track original
            episode_steps += 1

            replay_buffer.add(s, action, shaped_r, new_s, done)
            s = new_s

            total_steps += 1
            pbar.update(1)

            if len(replay_buffer) >= batch_size and total_steps % update_frequency == 0:
                states, actions, rewards, next_states, dones = replay_buffer.sample(
                    batch_size
                )
                critic_loss, actor_loss = model.train(
                    states,
                    actions,
                    rewards.reshape(-1, 1),
                    next_states,
                    dones.reshape(-1, 1),
                    1,
                )

                # Monitor progress for sparse rewards
                if total_steps % 10000 == 0 and total_steps > 0:
                    avg_episode_reward = episode_reward if episode_steps > 0 else -999
                    if avg_episode_reward > -2.0:
                        print(f"\n  Progress: Step {total_steps}, Episode reward: {avg_episode_reward:.3f}")
                    elif total_steps % 50000 == 0:
                        print(f"\n  Learning check: Step {total_steps}, Episode reward: {avg_episode_reward:.3f}")

                # Enhanced description with both rewards
                if use_reward_shaping:
                    pbar.set_description(
                        f"Ep {episode_count} | Shaped: {episode_reward:.2f} | Original: {episode_original_reward:.2f} | C: {critic_loss:.4f} | A: {actor_loss:.4f}"
                    )
                else:
                    pbar.set_description(
                        f"Episode {episode_count} | Reward: {episode_reward:.2f} | Critic: {critic_loss:.4f} | Actor: {actor_loss:.4f}"
                    )

        episode_count += 1
        total_reward += episode_reward
        total_original_reward += episode_original_reward
        episodes_since_best += 1
        
        # Track best model based on original reward (what SAI will actually evaluate)
        if episode_original_reward > best_original_reward:
            best_original_reward = episode_original_reward
            best_model_state = {
                'actor_state_dict': model.actor.state_dict(),
                'critic_state_dict': model.critic.state_dict(),
                'actor_target_state_dict': model.actor_target.state_dict(),
                'critic_target_state_dict': model.critic_target.state_dict(),
                'episode': episode_count,
                'original_reward': episode_original_reward,
                'shaped_reward': episode_reward
            }
            episodes_since_best = 0
            best_episode = episode_count
            print(f"\n🏆 NEW BEST MODEL! Episode {episode_count}")
            print(f"   Original Reward: {episode_original_reward:.4f}")
            print(f"   Shaped Reward: {episode_reward:.4f}")
            print(f"   Saving best model...")
        
        # Periodic detailed logging for correlation analysis
        if episode_count % 100 == 0:
            avg_shaped = total_reward / episode_count
            avg_original = total_original_reward / episode_count
            correlation_ratio = avg_shaped / avg_original if avg_original != 0 else 0
            print(f"\n=== Episode {episode_count} Analysis ===")
            print(f"Average Shaped Reward: {avg_shaped:.4f}")
            print(f"Average Original Reward: {avg_original:.4f}")
            print(f"Shaping Effectiveness Ratio: {correlation_ratio:.2f}")
            print(f"Recent Episode - Shaped: {episode_reward:.4f}, Original: {episode_original_reward:.4f}")
            print(f"Best Model: Episode {best_episode}, Original Reward: {best_original_reward:.4f}")
            print(f"Episodes since best: {episodes_since_best}")
            if abs(episode_original_reward) > 0.1:  # If we're getting meaningful original rewards
                print(f"🎉 PROGRESS: Getting non-zero original rewards!")
            print("=" * 40)

    pbar.close()
    
    # Restore best model for submission
    if best_model_state is not None:
        print(f"\n🎯 RESTORING BEST MODEL for submission:")
        print(f"   From Episode: {best_model_state['episode']}")
        print(f"   Original Reward: {best_model_state['original_reward']:.4f}")
        print(f"   Shaped Reward: {best_model_state['shaped_reward']:.4f}")
        
        model.actor.load_state_dict(best_model_state['actor_state_dict'])
        model.critic.load_state_dict(best_model_state['critic_state_dict'])
        model.actor_target.load_state_dict(best_model_state['actor_target_state_dict'])
        model.critic_target.load_state_dict(best_model_state['critic_target_state_dict'])
        
        # Save best model to disk as backup
        torch.save(best_model_state, 'best_model_checkpoint.pth')
        print(f"   Best model saved to: best_model_checkpoint.pth")
    else:
        print(f"\n⚠️  WARNING: No best model found, using final model")
        print(f"   Final episode reward: {episode_original_reward:.4f}")
    
    env.close()
