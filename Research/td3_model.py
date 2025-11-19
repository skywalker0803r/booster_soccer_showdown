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
        
        # 建立網路層
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
                # 輸出層
                current_layer = layer(current_layer)
                if self.output_activation is not None:
                    current_layer = self.output_activation(current_layer)
        return current_layer


class TD3_FF(torch.nn.Module):
    """
    Twin Delayed Deep Deterministic Policy Gradient (TD3) 算法
    相比DDPG的三個主要改進:
    1. Double Q-Learning (雙Critic網路)
    2. Delayed Policy Updates (延遲策略更新)
    3. Target Policy Smoothing (目標策略平滑化)
    """
    def __init__(
        self, n_features, action_space, neurons, activation_function, learning_rate,
        policy_delay=2, policy_noise=0.2, noise_clip=0.5
    ):
        super().__init__()
        self.action_space = action_space
        self.learning_rate = learning_rate
        self.gamma = 0.99
        self.tau = 0.005  # TD3使用較大的tau值
        
        # TD3 特有參數
        self.policy_delay = policy_delay  # 策略更新延遲
        self.policy_noise = policy_noise  # 目標策略噪音標準差
        self.noise_clip = noise_clip     # 噪音裁剪範圍
        self.update_counter = 0          # 更新計數器
        
        action_dim = action_space.shape[0]
        shared_inputs = [neurons, activation_function]
        
        # Actor 網路
        self.actor = NeuralNetwork(
            n_features,
            action_dim,
            *shared_inputs,
            F.tanh,  # 輸出層使用 tanh，將動作範圍約束在 [-1, 1]
        )
        
        # 🎯 TD3改進1: 雙Critic網路 (Double Q-Learning)
        self.critic1 = NeuralNetwork(
            n_features + action_dim, 1, *shared_inputs
        )
        self.critic2 = NeuralNetwork(
            n_features + action_dim, 1, *shared_inputs
        )

        # Target 網路
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

        # 初始化 Target 網路與主網路權重相同
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())

        # 優化器
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=self.learning_rate
        )
        self.critic_optimizer = torch.optim.Adam(
            list(self.critic1.parameters()) + list(self.critic2.parameters()), 
            lr=self.learning_rate
        )

    def soft_update_targets(self):
        """軟更新 Target 網路權重 (Polyak Averaging)"""
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
        """執行反向傳播和梯度裁剪"""
        optimizer.zero_grad()
        loss.backward()
        # 梯度裁剪，防止梯度爆炸
        for param_group in optimizer.param_groups:
            torch.nn.utils.clip_grad_norm_(param_group["params"], max_grad_norm)
        optimizer.step()

    @staticmethod
    def get_critic_state(state, action):
        """將狀態和動作合併為 Critic 的輸入"""
        return torch.cat([state, action], dim=1)

    @staticmethod
    def tensor_to_array(torch_tensor):
        """將 PyTorch Tensor 轉換為 numpy array"""
        return torch_tensor.detach().cpu().numpy()

    def forward(self, state):
        """僅返回 Actor 的動作輸出"""
        return self.actor(state).cpu()

    def select_action(self, state_np):
        """在環境交互時選擇動作"""
        state = torch.tensor(state_np).float().to(next(self.parameters()).device)
        return self.tensor_to_array(self.actor(state))

    def model_update(self, states, actions, rewards, next_states, dones):
        """
        TD3 模型的單次更新
        注意：輸入 states, actions, rewards, next_states, dones 已經是 tensor 且在正確的 device 上
        """
        self.update_counter += 1
        
        # --- Critic 更新 (每次都更新) ---
        with torch.no_grad():
            # 🎯 TD3改進3: Target Policy Smoothing (目標策略平滑化)
            next_actions = self.target_actor(next_states)
            
            # 添加裁剪噪音到目標動作
            noise = torch.clamp(
                torch.randn_like(next_actions) * self.policy_noise,
                -self.noise_clip, self.noise_clip
            )
            next_actions = torch.clamp(next_actions + noise, -1.0, 1.0)
            
            # 🎯 TD3改進1: Double Q-Learning (取兩個Q值的最小值)
            target_q1 = self.target_critic1(
                TD3_FF.get_critic_state(next_states, next_actions)
            )
            target_q2 = self.target_critic2(
                TD3_FF.get_critic_state(next_states, next_actions)
            )
            target_q = torch.min(target_q1, target_q2)
            
            # Bellman Target
            y = rewards + self.gamma * target_q * (1 - dones)

        # 計算當前 Q 值
        current_q1 = self.critic1(TD3_FF.get_critic_state(states, actions))
        current_q2 = self.critic2(TD3_FF.get_critic_state(states, actions))
        
        # Critic 損失 (兩個Critic的MSE損失之和)
        critic_loss = F.mse_loss(current_q1, y) + F.mse_loss(current_q2, y)
        TD3_FF.backprop(self.critic_optimizer, critic_loss)
        
        actor_loss = None
        
        # 🎯 TD3改進2: Delayed Policy Updates (延遲策略更新)
        if self.update_counter % self.policy_delay == 0:
            # --- Actor 更新 (每policy_delay次更新一次) ---
            
            # 計算當前狀態的最佳動作 (由 Actor 預測)
            actor_actions = self.actor(states)
            
            # 計算 Actor 損失 (-Q 值，只使用第一個Critic)
            actor_loss = -self.critic1(
                TD3_FF.get_critic_state(states, actor_actions)
            ).mean()
            TD3_FF.backprop(self.actor_optimizer, actor_loss)
            
            # --- Target 網路軟更新 ---
            self.soft_update_targets()
            
            actor_loss = actor_loss.item()
        else:
            # 如果不更新Actor，返回None或上一次的值
            actor_loss = 0.0

        return critic_loss.item(), actor_loss

    def get_statistics(self):
        """獲取模型統計信息"""
        return {
            'update_counter': self.update_counter,
            'policy_delay': self.policy_delay,
            'next_actor_update': self.policy_delay - (self.update_counter % self.policy_delay)
        }


class ReplayBuffer:
    """標準經驗重放緩衝區 (與DDPG相同)"""
    def __init__(self, capacity, observation_shape, action_dim):
        self.capacity = capacity
        self.ptr = 0
        self.size = 0
        
        # 使用 numpy 陣列儲存經驗
        self.states = np.zeros((capacity, *observation_shape), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        # 調整 rewards 和 dones 的 shape
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.next_states = np.zeros((capacity, *observation_shape), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)

    def add(self, state, action, reward, next_state, done):
        """儲存單次轉變 (s, a, r, s', d)"""
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr] = float(done)  # 轉換為 float (0.0 或 1.0)
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        """隨機採樣批次經驗"""
        ind = np.random.randint(0, self.size, size=batch_size)
        
        return (
            self.states[ind],
            self.actions[ind],
            self.rewards[ind],
            self.next_states[ind],
            self.dones[ind],
        )