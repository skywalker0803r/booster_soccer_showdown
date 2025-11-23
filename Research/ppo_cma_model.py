# -*- coding: utf-8 -*-
# ppo_cma_model.py
# PPO-CMA 算法實現：結合 PPO 和 CMA-ES 的策略優化算法
# 基於論文: "PPO-CMA: Proximal Policy Optimization with Covariance Matrix Adaptation"

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from collections import deque
import copy


class ActorNetwork(nn.Module):
    """高斯策略網絡 - 輸出動作均值和log標準差"""
    
    def __init__(self, state_dim, action_dim, hidden_dims, activation=nn.ReLU):
        super(ActorNetwork, self).__init__()
        self.action_dim = action_dim
        
        # 構建網絡層
        layers = []
        input_dim = state_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                activation(),
            ])
            input_dim = hidden_dim
        
        # 移除最後的激活函數
        if layers:
            layers = layers[:-1]
        
        self.backbone = nn.Sequential(*layers)
        
        # 分別輸出均值和log標準差
        self.mean_head = nn.Linear(input_dim, action_dim)
        self.log_std_head = nn.Linear(input_dim, action_dim)
        
        # 初始化權重
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            torch.nn.init.constant_(module.bias, 0)
    
    def forward(self, state):
        features = self.backbone(state)
        mean = torch.tanh(self.mean_head(features))  # 限制在 [-1, 1]
        log_std = self.log_std_head(features)
        log_std = torch.clamp(log_std, -20, 2)  # 防止數值不穩定
        
        return mean, log_std
    
    def get_action_and_log_prob(self, state):
        mean, log_std = self.forward(state)
        std = torch.exp(log_std)
        
        # 創建高斯分佈
        distribution = Normal(mean, std)
        action = distribution.sample()
        log_prob = distribution.log_prob(action).sum(dim=-1)
        
        return action, log_prob, mean, log_std
    
    def evaluate_actions(self, state, action):
        mean, log_std = self.forward(state)
        std = torch.exp(log_std)
        
        distribution = Normal(mean, std)
        log_prob = distribution.log_prob(action).sum(dim=-1)
        entropy = distribution.entropy().sum(dim=-1)
        
        return log_prob, entropy


class CriticNetwork(nn.Module):
    """價值函數網絡"""
    
    def __init__(self, state_dim, hidden_dims, activation=nn.ReLU):
        super(CriticNetwork, self).__init__()
        
        layers = []
        input_dim = state_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                activation(),
            ])
            input_dim = hidden_dim
        
        # 移除最後的激活函數
        if layers:
            layers = layers[:-1]
        
        layers.append(nn.Linear(input_dim, 1))
        
        self.network = nn.Sequential(*layers)
        
        # 初始化權重
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            torch.nn.init.constant_(module.bias, 0)
    
    def forward(self, state):
        return self.network(state).squeeze(-1)


class PPOCMABuffer:
    """PPO-CMA 專用經驗緩衝區"""
    
    def __init__(self, capacity, state_dim, action_dim):
        self.capacity = capacity
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.bool_)
        self.log_probs = np.zeros(capacity, dtype=np.float32)
        self.values = np.zeros(capacity, dtype=np.float32)
        self.advantages = np.zeros(capacity, dtype=np.float32)
        self.returns = np.zeros(capacity, dtype=np.float32)
        
        self.ptr = 0
        self.size = 0
        
    def store(self, state, action, reward, next_state, done, log_prob, value):
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr] = done
        self.log_probs[self.ptr] = log_prob
        self.values[self.ptr] = value
        
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        
    def compute_advantages_and_returns(self, gamma=0.99, gae_lambda=0.95, last_value=0):
        """計算 GAE (Generalized Advantage Estimation) 和 returns"""
        advantages = np.zeros_like(self.rewards)
        returns = np.zeros_like(self.rewards)
        
        last_gae = 0
        last_return = last_value
        
        for t in reversed(range(self.size)):
            if t == self.size - 1:
                next_value = last_value
            else:
                next_value = self.values[t + 1]
            
            # TD 誤差
            delta = self.rewards[t] + gamma * next_value * (1 - self.dones[t]) - self.values[t]
            
            # GAE
            advantages[t] = last_gae = delta + gamma * gae_lambda * (1 - self.dones[t]) * last_gae
            
            # Returns
            returns[t] = last_return = self.rewards[t] + gamma * (1 - self.dones[t]) * last_return
        
        self.advantages = advantages
        self.returns = returns
        
        # 正規化 advantages
        if self.size > 1:
            self.advantages = (self.advantages - self.advantages.mean()) / (self.advantages.std() + 1e-8)
    
    def get_all_data(self):
        """獲取所有數據用於訓練"""
        if self.size == 0:
            return None
            
        return {
            'states': torch.FloatTensor(self.states[:self.size]),
            'actions': torch.FloatTensor(self.actions[:self.size]),
            'log_probs': torch.FloatTensor(self.log_probs[:self.size]),
            'advantages': torch.FloatTensor(self.advantages[:self.size]),
            'returns': torch.FloatTensor(self.returns[:self.size]),
            'values': torch.FloatTensor(self.values[:self.size])
        }
    
    def clear(self):
        """清空緩衝區"""
        self.ptr = 0
        self.size = 0


class CovarianceMatrixAdaptation:
    """CMA-ES 協方差矩陣適應模組"""
    
    def __init__(self, parameter_dim, population_size=None, sigma=0.1):
        self.n = parameter_dim
        self.sigma = sigma  # 步長
        
        # 設定 population size (論文建議 4 + floor(3*ln(n)))
        if population_size is None:
            self.lam = 4 + int(3 * np.log(self.n))
        else:
            self.lam = population_size
            
        self.mu = self.lam // 2  # 選擇的個體數量
        
        # 權重設置
        self.weights = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        self.weights = self.weights / np.sum(self.weights)
        self.mu_eff = 1 / np.sum(self.weights**2)
        
        # 協方差矩陣更新參數
        self.c_sigma = (self.mu_eff + 2) / (self.n + self.mu_eff + 5)
        self.c_c = (4 + self.mu_eff/self.n) / (self.n + 4 + 2*self.mu_eff/self.n)
        self.c_1 = 2 / ((self.n + 1.3)**2 + self.mu_eff)
        self.c_mu = min(1 - self.c_1, 2*(self.mu_eff - 2 + 1/self.mu_eff) / ((self.n + 2)**2 + self.mu_eff))
        self.d_sigma = 1 + 2*max(0, np.sqrt((self.mu_eff-1)/(self.n+1)) - 1) + self.c_sigma
        
        # 期望值
        self.chi_n = np.sqrt(self.n) * (1 - 1/(4*self.n) + 1/(21*self.n**2))
        
        # 狀態變量
        self.mean = np.zeros(self.n)
        self.C = np.eye(self.n)  # 協方差矩陣
        self.p_sigma = np.zeros(self.n)  # 進化路徑 for step-size
        self.p_c = np.zeros(self.n)  # 進化路徑 for covariance matrix
        
        self.generation = 0
        
    def generate_offspring(self, mean):
        """生成子代個體"""
        # 確保協方差矩陣正定
        eigenvals, eigenvecs = np.linalg.eigh(self.C)
        eigenvals = np.maximum(eigenvals, 1e-14)
        self.C = eigenvecs @ np.diag(eigenvals) @ eigenvecs.T
        
        # 生成樣本
        samples = np.random.multivariate_normal(np.zeros(self.n), self.C, self.lam)
        offspring = mean + self.sigma * samples
        
        return offspring, samples
        
    def update(self, mean, offspring, fitness_values):
        """根據fitness值更新CMA-ES參數"""
        # 排序選擇最好的個體
        sorted_indices = np.argsort(fitness_values)[::-1]  # 降序排列（假設higher is better）
        selected_offspring = offspring[sorted_indices[:self.mu]]
        
        # 更新均值
        old_mean = self.mean.copy()
        self.mean = np.sum(self.weights[:, np.newaxis] * selected_offspring, axis=0)
        
        # 計算進化路徑
        # Step-size control path
        C_inv_sqrt = self._matrix_sqrt_inv(self.C)
        self.p_sigma = (1 - self.c_sigma) * self.p_sigma + \
                      np.sqrt(self.c_sigma * (2 - self.c_sigma) * self.mu_eff) * \
                      C_inv_sqrt @ (self.mean - old_mean) / self.sigma
        
        # Covariance matrix path
        h_sigma = (np.linalg.norm(self.p_sigma) / 
                   np.sqrt(1 - (1 - self.c_sigma)**(2*(self.generation + 1)))) < \
                   (1.4 + 2/(self.n + 1)) * self.chi_n
        
        self.p_c = (1 - self.c_c) * self.p_c + \
                   h_sigma * np.sqrt(self.c_c * (2 - self.c_c) * self.mu_eff) * \
                   (self.mean - old_mean) / self.sigma
        
        # 更新協方差矩陣
        y_k = (selected_offspring - old_mean) / self.sigma
        sum_w_y_k = np.sum(self.weights[:, np.newaxis, np.newaxis] * 
                          y_k[:, :, np.newaxis] * y_k[:, np.newaxis, :], axis=0)
        
        self.C = (1 - self.c_1 - self.c_mu) * self.C + \
                 self.c_1 * (self.p_c[:, np.newaxis] * self.p_c[np.newaxis, :] + \
                             (1 - h_sigma) * self.c_c * (2 - self.c_c) * self.C) + \
                 self.c_mu * sum_w_y_k
        
        # 更新步長
        self.sigma *= np.exp((self.c_sigma / self.d_sigma) * 
                           (np.linalg.norm(self.p_sigma) / self.chi_n - 1))
        
        self.generation += 1
        
        return self.mean
        
    def _matrix_sqrt_inv(self, matrix):
        """計算矩陣的逆平方根"""
        eigenvals, eigenvecs = np.linalg.eigh(matrix)
        eigenvals = np.maximum(eigenvals, 1e-14)  # 避免數值問題
        return eigenvecs @ np.diag(1.0 / np.sqrt(eigenvals)) @ eigenvecs.T


class PPOCMA:
    """PPO-CMA 主算法類"""
    
    def __init__(self, state_dim, action_dim, hidden_dims, lr_actor=3e-4, lr_critic=1e-3,
                 gamma=0.99, gae_lambda=0.95, clip_epsilon=0.2, entropy_coef=0.01,
                 max_grad_norm=0.5, ppo_epochs=10, batch_size=64, buffer_capacity=2048,
                 cma_population_size=None, cma_sigma=0.1, cma_update_freq=10):
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.ppo_epochs = ppo_epochs
        self.batch_size = batch_size
        self.cma_update_freq = cma_update_freq
        
        # 建立網絡
        self.actor = ActorNetwork(state_dim, action_dim, hidden_dims)
        self.critic = CriticNetwork(state_dim, hidden_dims)
        
        # 優化器
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr_critic)
        
        # 經驗緩衝區
        self.buffer = PPOCMABuffer(buffer_capacity, state_dim, action_dim)
        
        # CMA-ES 模組 (用於策略參數優化)
        # 為避免內存問題，我們只對最後一層參數使用CMA-ES
        last_layer_params = list(self.actor.mean_head.parameters()) + list(self.actor.log_std_head.parameters())
        actor_param_count = sum(p.numel() for p in last_layer_params)
        print(f"CMA-ES參數數量: {actor_param_count}")
        
        self.cma = CovarianceMatrixAdaptation(
            actor_param_count, 
            population_size=cma_population_size,
            sigma=cma_sigma
        )
        self.use_cma = True
        
        # 統計變量
        self.update_counter = 0
        self.total_steps = 0
        self.cma_updates = 0
        
        # 存儲 CMA 相關的候選策略和其適應度
        self.candidate_actors = []
        self.candidate_fitness = []
        
    def get_action(self, state):
        """根據當前策略選擇動作"""
        with torch.no_grad():
            if isinstance(state, np.ndarray):
                state = torch.FloatTensor(state).unsqueeze(0)
            
            action, log_prob, _, _ = self.actor.get_action_and_log_prob(state)
            value = self.critic(state)
            
            return action.cpu().numpy().flatten(), log_prob.cpu().numpy().item(), value.cpu().numpy().item()
    
    def store_transition(self, state, action, reward, next_state, done, log_prob, value):
        """存儲轉移到緩衝區"""
        self.buffer.store(state, action, reward, next_state, done, log_prob, value)
        self.total_steps += 1
    
    def _flatten_parameters(self, model):
        """將模型最後一層參數展平為一維向量"""
        params = []
        for param in list(model.mean_head.parameters()) + list(model.log_std_head.parameters()):
            params.append(param.data.view(-1))
        return torch.cat(params).cpu().numpy()
    
    def _unflatten_parameters(self, model, flat_params):
        """將一維參數向量重新設置到模型最後一層中"""
        flat_params = torch.FloatTensor(flat_params)
        idx = 0
        target_params = list(model.mean_head.parameters()) + list(model.log_std_head.parameters())
        for param in target_params:
            param_length = param.numel()
            param.data = flat_params[idx:idx + param_length].view(param.shape)
            idx += param_length
    
    def update(self):
        """執行PPO-CMA更新"""
        if self.buffer.size < self.batch_size:
            return None, None
        
        # 計算最後的 value 估計
        with torch.no_grad():
            if self.buffer.size > 0:
                last_state = torch.FloatTensor(self.buffer.next_states[self.buffer.size - 1:self.buffer.size])
                last_value = self.critic(last_state).item()
            else:
                last_value = 0
        
        # 計算 advantages 和 returns
        self.buffer.compute_advantages_and_returns(self.gamma, self.gae_lambda, last_value)
        
        # 獲取訓練數據
        batch_data = self.buffer.get_all_data()
        if batch_data is None:
            return None, None
        
        # PPO 更新
        actor_losses = []
        critic_losses = []
        
        for _ in range(self.ppo_epochs):
            # 隨機打亂數據
            indices = torch.randperm(len(batch_data['states']))
            
            for start_idx in range(0, len(indices), self.batch_size):
                end_idx = start_idx + self.batch_size
                batch_indices = indices[start_idx:end_idx]
                
                # 提取批次數據
                states = batch_data['states'][batch_indices]
                actions = batch_data['actions'][batch_indices]
                old_log_probs = batch_data['log_probs'][batch_indices]
                advantages = batch_data['advantages'][batch_indices]
                returns = batch_data['returns'][batch_indices]
                old_values = batch_data['values'][batch_indices]
                
                # 計算新的 log_probs 和 values
                new_log_probs, entropy = self.actor.evaluate_actions(states, actions)
                new_values = self.critic(states)
                
                # PPO Actor 損失
                ratio = torch.exp(new_log_probs - old_log_probs)
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
                actor_loss = -torch.min(surr1, surr2).mean() - self.entropy_coef * entropy.mean()
                
                # PPO Critic 損失
                value_loss_unclipped = (new_values - returns) ** 2
                values_clipped = old_values + torch.clamp(new_values - old_values, 
                                                        -self.clip_epsilon, self.clip_epsilon)
                value_loss_clipped = (values_clipped - returns) ** 2
                critic_loss = 0.5 * torch.max(value_loss_unclipped, value_loss_clipped).mean()
                
                # 更新 Actor
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()
                
                # 更新 Critic
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.critic_optimizer.step()
                
                actor_losses.append(actor_loss.item())
                critic_losses.append(critic_loss.item())
        
        # CMA-ES 更新 (每隔一定步數)
        if self.use_cma and self.update_counter % self.cma_update_freq == 0:
            self._cma_update()
        
        self.update_counter += 1
        
        # 清空緩衝區
        self.buffer.clear()
        
        return np.mean(actor_losses) if actor_losses else 0, np.mean(critic_losses) if critic_losses else 0
    
    def _cma_update(self):
        """執行CMA-ES參數更新"""
        # 獲取當前策略參數
        current_params = self._flatten_parameters(self.actor)
        
        # 生成候選策略參數
        candidate_params, _ = self.cma.generate_offspring(current_params)
        
        # 評估每個候選策略（這裡使用簡化版本）
        fitness_values = []
        
        for params in candidate_params:
            # 創建臨時actor並設置參數
            temp_actor = copy.deepcopy(self.actor)
            self._unflatten_parameters(temp_actor, params)
            
            # 簡化的適應度評估（實際中可能需要完整的episode評估）
            # 這裡使用參數與當前最優參數的距離作為近似
            param_diff = np.linalg.norm(params - current_params)
            fitness = -param_diff  # 負距離作為適應度
            
            fitness_values.append(fitness)
        
        # 更新CMA參數
        self.cma.update(current_params, candidate_params, np.array(fitness_values))
        
        # 可選：使用CMA建議的最佳參數更新actor
        if self.cma_updates % 5 == 0:  # 每5次CMA更新才真正應用
            best_params = self.cma.mean
            self._unflatten_parameters(self.actor, best_params)
        
        self.cma_updates += 1
    
    def get_statistics(self):
        """獲取訓練統計信息"""
        stats = {
            'update_counter': self.update_counter,
            'total_steps': self.total_steps,
            'cma_updates': self.cma_updates,
            'buffer_size': self.buffer.size,
            'use_cma': self.use_cma
        }
        
        if self.use_cma and self.cma is not None:
            stats.update({
                'cma_sigma': self.cma.sigma,
                'cma_generation': self.cma.generation
            })
        else:
            stats.update({
                'cma_sigma': 0.0,
                'cma_generation': 0
            })
        
        return stats
    
    def state_dict(self):
        """獲取模型狀態字典"""
        return {
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'update_counter': self.update_counter,
            'total_steps': self.total_steps,
            'cma_updates': self.cma_updates
        }
    
    def load_state_dict(self, state_dict):
        """載入模型狀態字典"""
        self.actor.load_state_dict(state_dict['actor'])
        self.critic.load_state_dict(state_dict['critic'])
        self.actor_optimizer.load_state_dict(state_dict['actor_optimizer'])
        self.critic_optimizer.load_state_dict(state_dict['critic_optimizer'])
        self.update_counter = state_dict.get('update_counter', 0)
        self.total_steps = state_dict.get('total_steps', 0)
        self.cma_updates = state_dict.get('cma_updates', 0)
    
    def to(self, device):
        """移動模型到指定設備"""
        self.actor = self.actor.to(device)
        self.critic = self.critic.to(device)
        return self