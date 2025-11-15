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
    # Decay noise over time for better convergence
    decay_factor = max(0.1, 1.0 - episode_count / total_episodes)
    effective_noise = noise_scale * decay_factor
    noise = np.random.normal(0, effective_noise, size=action.shape)
    return np.clip(action + noise, -1, 1)


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

    total_steps = 0
    episode_count = 0

    pbar = tqdm(total=timesteps, desc="Training Progress", unit="steps")

    while total_steps < timesteps:
        done = False
        s, info = env.reset()
        s = preprocessor.modify_state(s, info).squeeze()
        episode_reward = 0
        episode_steps = 0

        while not done and total_steps < timesteps:
            state = torch.from_numpy(np.expand_dims(s, axis=0))
            policy = model(state).detach().numpy()

            if action_function:
                action = action_function(policy)[0].squeeze()
                # Use decaying noise for better exploration-exploitation balance
                action = add_noise(action, noise_scale=0.2, episode_count=episode_count, total_episodes=timesteps//20)
            else:
                action = model.select_action(s)[0].squeeze()

            new_s, r, terminated, truncated, info = env.step(action)
            new_s = preprocessor.modify_state(new_s, info).squeeze()

            done = terminated or truncated

            episode_reward += r
            episode_steps += 1

            replay_buffer.add(s, action, r, new_s, done)
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

                pbar.set_description(
                    f"Episode {episode_count} | Reward: {episode_reward:.2f} | Critic: {critic_loss:.4f} | Actor: {actor_loss:.4f}"
                )

        episode_count += 1

    pbar.close()
    env.close()
