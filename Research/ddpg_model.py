# -*- coding: utf-8 -*-
# ddpg_model.py

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


class DDPG_FF(torch.nn.Module):
    """
    前饋 (FeedForward) DDPG 算法的核心模型
    """
    def __init__(
        self, n_features, action_space, neurons, activation_function, learning_rate
    ):
        super().__init__()
        self.action_space = action_space
        self.learning_rate = learning_rate
        self.gamma = 0.99
        self.tau = 0.001
        
        action_dim = action_space.shape[0]

        shared_inputs = [neurons, activation_function]
        
        # Actor 網路
        self.actor = NeuralNetwork(
            n_features,
            action_dim,
            *shared_inputs,
            F.tanh, # 輸出層使用 tanh，將動作範圍約束在 [-1, 1]
        )
        # Critic 網路 (輸入狀態 + 動作)
        self.critic = NeuralNetwork(
            n_features + action_dim, 1, *shared_inputs
        )

        # Target 網路
        self.target_actor = NeuralNetwork(
            n_features,
            action_dim,
            *shared_inputs,
            F.tanh,
        )
        self.target_critic = NeuralNetwork(
            n_features + action_dim, 1, *shared_inputs
        )

        # 初始化 Target 網路與主網路權重相同
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())

        # 優化器
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=self.learning_rate
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=self.learning_rate
        )

    def soft_update_targets(self):
        """軟更新 Target 網路權重 (Polyklak Averaging)"""
        for target_param, param in zip(
            self.target_actor.parameters(), self.actor.parameters()
        ):
            target_param.data.copy_(
                self.tau * param.data + (1.0 - self.tau) * target_param.data
            )

        for target_param, param in zip(
            self.target_critic.parameters(), self.critic.parameters()
        ):
            target_param.data.copy_(
                self.tau * param.data + (1.0 - self.tau) * target_param.data
            )

    @staticmethod
    def backprop(optimizer, loss):
        """執行反向傳播和梯度裁剪"""
        optimizer.zero_grad()
        loss.backward()
        # 梯度裁剪，防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(optimizer.param_groups[0]["params"], 1.0)
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
        DDPG 模型的單次更新
        注意：輸入 states, actions, rewards, next_states, dones 已經是 tensor 且在正確的 device 上
        """
        # --- Critic 更新 ---
        with torch.no_grad():
            # 1. 計算下一個狀態的 Target Action
            next_actions = self.target_actor(next_states)
            # 2. 計算 Target Q 值
            target_q = self.target_critic(
                DDPG_FF.get_critic_state(next_states, next_actions)
            )
            # 3. 計算 Bellman Target (y)
            y = rewards + self.gamma * target_q * (1 - dones)

        # 4. 計算當前 Q 值
        current_q = self.critic(DDPG_FF.get_critic_state(states, actions))
        # 5. Critic 損失 (MSE)
        critic_loss = F.mse_loss(current_q, y)
        DDPG_FF.backprop(self.critic_optimizer, critic_loss)

        # --- Actor 更新 ---
        # 1. 計算當前狀態的最佳動作 (由 Actor 預測)
        actor_actions = self.actor(states)
        # 2. 計算 Actor 損失 (-Q 值)
        actor_loss = -self.critic(
            DDPG_FF.get_critic_state(states, actor_actions)
        ).mean()
        DDPG_FF.backprop(self.actor_optimizer, actor_loss)

        # --- Target 網路軟更新 ---
        self.soft_update_targets()

        return critic_loss.item(), actor_loss.item()


class ReplayBuffer:
    """標準經驗重放緩衝區"""
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
        self.dones[self.ptr] = float(done) # 轉換為 float (0.0 或 1.0)
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