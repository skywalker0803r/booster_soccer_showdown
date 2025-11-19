# -*- coding: utf-8 -*-
# td3_model.py
# Twin Delayed Deep Deterministic Policy Gradient (TD3) implementation

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class NeuralNetwork(nn.Module):
    def __init__(
        self,
        n_features,
        n_actions,
        neurons,
        activation_function,
        output_activation=None,
    ):
        super().__init__()
        self.n_features = n_features
        self.neurons = neurons
        self.activation_function = activation_function
        self.output_activation = output_activation
        self.n_actions = n_actions

        self.n_layers = len(self.neurons) + 1
        self.layers = torch.nn.ModuleList()
        
        # Build network layers
        for index in range(self.n_layers):
            if index == 0:
                in_dim = n_features
                out_dim = neurons[index]
            elif index == self.n_layers - 1:
                in_dim = neurons[index - 1]
                out_dim = self.n_actions
            else:
                in_dim = neurons[index - 1]
                out_dim = neurons[index]
            self.layers.append(nn.Linear(in_dim, out_dim))

    def forward(self, current_layer):
        model_device = next(self.parameters()).device
        if current_layer.device != model_device:
            current_layer = current_layer.to(model_device)

        if current_layer.dtype != torch.float32:
            current_layer = current_layer.float()
            
        for index, layer in enumerate(self.layers):
            if index < self.n_layers - 1:
                current_layer = self.activation_function(layer(current_layer))
            else:
                # Output layer
                current_layer = layer(current_layer)
                if self.output_activation is not None:
                    current_layer = self.output_activation(current_layer)
        return current_layer


class TD3_FF(torch.nn.Module):
    """
    Twin Delayed Deep Deterministic Policy Gradient (TD3) Algorithm
    Three main improvements over DDPG:
    1. Double Q-Learning (Twin Critic Networks)
    2. Delayed Policy Updates
    3. Target Policy Smoothing
    """
    def __init__(
        self, n_features, action_space, neurons, activation_function, learning_rate,
        policy_delay=2, policy_noise=0.2, noise_clip=0.5
    ):
        super().__init__()
        self.action_space = action_space
        self.learning_rate = learning_rate
        self.gamma = 0.99
        self.tau = 0.005  # TD3 uses larger tau value
        
        # TD3 specific parameters
        self.policy_delay = policy_delay  # Policy update delay
        self.policy_noise = policy_noise  # Target policy noise std
        self.noise_clip = noise_clip     # Noise clipping range
        self.update_counter = 0          # Update counter
        
        action_dim = action_space.shape[0]
        shared_inputs = [neurons, activation_function]
        
        # Actor Network
        self.actor = NeuralNetwork(
            n_features,
            action_dim,
            *shared_inputs,
            F.tanh,  # Output layer uses tanh to constrain action range to [-1, 1]
        )
        
        # 🎯 TD3 Improvement 1: Twin Critic Networks (Double Q-Learning)
        self.critic1 = NeuralNetwork(
            n_features + action_dim, 1, *shared_inputs
        )
        self.critic2 = NeuralNetwork(
            n_features + action_dim, 1, *shared_inputs
        )

        # Target Networks
        self.target_actor = NeuralNetwork(
            n_features,
            action_dim,
            *shared_inputs,
            F.tanh,
        )
        self.target_critic1 = NeuralNetwork(
            n_features + action_dim, 1, *shared_inputs
        )
        self.target_critic2 = NeuralNetwork(
            n_features + action_dim, 1, *shared_inputs
        )

        # Initialize Target networks with same weights as main networks
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())

        # Optimizers
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=self.learning_rate
        )
        self.critic_optimizer = torch.optim.Adam(
            list(self.critic1.parameters()) + list(self.critic2.parameters()), 
            lr=self.learning_rate
        )

    def soft_update_targets(self):
        """Soft update Target network weights (Polyak Averaging)"""
        # Actor
        for target_param, param in zip(
            self.target_actor.parameters(), self.actor.parameters()
        ):
            target_param.data.copy_(
                self.tau * param.data + (1.0 - self.tau) * target_param.data
            )
        
        # Critic1
        for target_param, param in zip(
            self.target_critic1.parameters(), self.critic1.parameters()
        ):
            target_param.data.copy_(
                self.tau * param.data + (1.0 - self.tau) * target_param.data
            )
            
        # Critic2
        for target_param, param in zip(
            self.target_critic2.parameters(), self.critic2.parameters()
        ):
            target_param.data.copy_(
                self.tau * param.data + (1.0 - self.tau) * target_param.data
            )

    @staticmethod
    def backprop(optimizer, loss, max_grad_norm=1.0):
        """Execute backpropagation and gradient clipping"""
        optimizer.zero_grad()
        loss.backward()
        # Gradient clipping to prevent gradient explosion
        for param_group in optimizer.param_groups:
            torch.nn.utils.clip_grad_norm_(param_group["params"], max_grad_norm)
        optimizer.step()

    @staticmethod
    def get_critic_state(state, action):
        """Combine state and action as input for Critic"""
        return torch.cat([state, action], dim=1)

    @staticmethod
    def tensor_to_array(torch_tensor):
        """Convert PyTorch Tensor to numpy array"""
        return torch_tensor.detach().cpu().numpy()

    def forward(self, state):
        """Return only Actor's action output"""
        return self.actor(state).cpu()

    def select_action(self, state_np):
        """Select action during environment interaction"""
        state = torch.tensor(state_np).float().to(next(self.parameters()).device)
        return self.tensor_to_array(self.actor(state))

    def model_update(self, states, actions, rewards, next_states, dones):
        """
        Single update of TD3 model
        Note: inputs states, actions, rewards, next_states, dones are already tensors on correct device
        """
        self.update_counter += 1
        
        # --- Critic Update (update every time) ---
        with torch.no_grad():
            # 🎯 TD3 Improvement 3: Target Policy Smoothing
            next_actions = self.target_actor(next_states)
            
            # Add clipped noise to target actions
            noise = torch.clamp(
                torch.randn_like(next_actions) * self.policy_noise,
                -self.noise_clip, self.noise_clip
            )
            next_actions = torch.clamp(next_actions + noise, -1.0, 1.0)
            
            # 🎯 TD3 Improvement 1: Double Q-Learning (take minimum of two Q values)
            target_q1 = self.target_critic1(
                TD3_FF.get_critic_state(next_states, next_actions)
            )
            target_q2 = self.target_critic2(
                TD3_FF.get_critic_state(next_states, next_actions)
            )
            target_q = torch.min(target_q1, target_q2)
            
            # Bellman Target
            y = rewards + self.gamma * target_q * (1 - dones)

        # Calculate current Q values
        current_q1 = self.critic1(TD3_FF.get_critic_state(states, actions))
        current_q2 = self.critic2(TD3_FF.get_critic_state(states, actions))
        
        # Critic loss (sum of MSE losses from both Critics)
        critic_loss = F.mse_loss(current_q1, y) + F.mse_loss(current_q2, y)
        TD3_FF.backprop(self.critic_optimizer, critic_loss)
        
        actor_loss = None
        
        # 🎯 TD3 Improvement 2: Delayed Policy Updates
        if self.update_counter % self.policy_delay == 0:
            # --- Actor Update (update every policy_delay times) ---
            
            # Calculate optimal actions for current states (predicted by Actor)
            actor_actions = self.actor(states)
            
            # Calculate Actor loss (-Q value, only use first Critic)
            actor_loss = -self.critic1(
                TD3_FF.get_critic_state(states, actor_actions)
            ).mean()
            TD3_FF.backprop(self.actor_optimizer, actor_loss)
            
            # --- Soft update Target networks ---
            self.soft_update_targets()
            
            actor_loss = actor_loss.item()
        else:
            # If Actor is not updated, return None or previous value
            actor_loss = 0.0

        return critic_loss.item(), actor_loss

    def get_statistics(self):
        """Get model statistics"""
        return {
            'update_counter': self.update_counter,
            'policy_delay': self.policy_delay,
            'next_actor_update': self.policy_delay - (self.update_counter % self.policy_delay)
        }


class ReplayBuffer:
    """Standard experience replay buffer (same as DDPG)"""
    def __init__(self, capacity, observation_shape, action_dim):
        self.capacity = capacity
        self.ptr = 0
        self.size = 0
        
        # Use numpy arrays to store experiences
        self.states = np.zeros((capacity, *observation_shape), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        # Adjust shape of rewards and dones
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.next_states = np.zeros((capacity, *observation_shape), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)

    def add(self, state, action, reward, next_state, done):
        """Store single transition (s, a, r, s', d)"""
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr] = float(done)  # Convert to float (0.0 or 1.0)
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        """Randomly sample batch experiences"""
        ind = np.random.randint(0, self.size, size=batch_size)
        
        return (
            self.states[ind],
            self.actions[ind],
            self.rewards[ind],
            self.next_states[ind],
            self.dones[ind],
        )