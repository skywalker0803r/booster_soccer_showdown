# -*- coding: utf-8 -*-
# curiosity_module.py
# 基於ICM (Intrinsic Curiosity Module) 的好奇心驅動探索模組

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ICM(nn.Module):
    """
    Intrinsic Curiosity Module (ICM)
    論文: Curiosity-driven Exploration by Self-supervised Prediction
    
    包含三個網絡：
    1. Feature Network: 提取狀態特徵
    2. Forward Model: 預測下一狀態特徵 
    3. Inverse Model: 從狀態特徵預測動作
    """
    
    def __init__(self, state_dim, action_dim, feature_dim=64, hidden_dim=128):
        super(ICM, self).__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.feature_dim = feature_dim
        
        # Feature Network: 提取狀態的緊湊特徵表示
        self.feature_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim),
            nn.ReLU()
        )
        
        # Inverse Model: φ(s_t), φ(s_{t+1}) -> a_t
        self.inverse_net = nn.Sequential(
            nn.Linear(feature_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()  # 動作範圍 [-1, 1]
        )
        
        # Forward Model: φ(s_t), a_t -> φ(s_{t+1})
        self.forward_net = nn.Sequential(
            nn.Linear(feature_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim)
        )
        
        self.optimizer = torch.optim.Adam(self.parameters(), lr=3e-4)
        
    def forward(self, state):
        """提取狀態特徵"""
        return self.feature_net(state)
    
    def get_intrinsic_reward(self, state, action, next_state):
        """
        計算內在獎勵 (好奇心獎勵)
        基於 Forward Model 的預測誤差
        """
        with torch.no_grad():
            # 提取特徵
            phi_state = self.feature_net(state)
            phi_next_state = self.feature_net(next_state)
            
            # Forward Model 預測
            predicted_next_phi = self.forward_net(
                torch.cat([phi_state, action], dim=1)
            )
            
            # 計算預測誤差 (內在獎勵)
            intrinsic_reward = F.mse_loss(
                predicted_next_phi, phi_next_state, reduction='none'
            ).mean(dim=1, keepdim=True)
            
        return intrinsic_reward
    
    def update(self, state_batch, action_batch, next_state_batch):
        """
        更新ICM的三個網絡
        返回: forward_loss, inverse_loss, intrinsic_rewards
        """
        # 提取特徵
        phi_state = self.feature_net(state_batch)
        phi_next_state = self.feature_net(next_state_batch)
        
        # Forward Model Loss
        predicted_next_phi = self.forward_net(
            torch.cat([phi_state, action_batch], dim=1)
        )
        forward_loss = F.mse_loss(predicted_next_phi, phi_next_state)
        
        # Inverse Model Loss  
        predicted_action = self.inverse_net(
            torch.cat([phi_state, phi_next_state], dim=1)
        )
        inverse_loss = F.mse_loss(predicted_action, action_batch)
        
        # 總損失 (論文中建議的權重)
        total_loss = 0.2 * inverse_loss + 0.8 * forward_loss
        
        # 反向傳播
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 0.5)
        self.optimizer.step()
        
        # 計算內在獎勵 (用於記錄)
        with torch.no_grad():
            intrinsic_rewards = F.mse_loss(
                predicted_next_phi, phi_next_state, reduction='none'
            ).mean(dim=1)
        
        return forward_loss.item(), inverse_loss.item(), intrinsic_rewards.mean().item()


class CuriosityDrivenExploration:
    """
    好奇心驅動的探索包裝器
    整合ICM模組與原有的DDPG訓練
    """
    
    def __init__(self, state_dim, action_dim, intrinsic_reward_scale=0.1):
        self.icm = ICM(state_dim, action_dim)
        self.intrinsic_reward_scale = intrinsic_reward_scale
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.icm.to(self.device)
        
        # 統計信息
        self.total_intrinsic_reward = 0.0
        self.update_count = 0
        
    def to(self, device):
        """移動到指定設備"""
        self.device = device
        self.icm.to(device)
        return self
    
    def get_enhanced_reward(self, state, action, next_state, extrinsic_reward):
        """
        計算增強獎勵 = 外在獎勵 + 內在獎勵
        """
        # 確保輸入是正確的tensor格式
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state).float()
        if not isinstance(action, torch.Tensor):
            action = torch.tensor(action).float()
        if not isinstance(next_state, torch.Tensor):
            next_state = torch.tensor(next_state).float()
            
        # 移動到正確設備
        state = state.to(self.device)
        action = action.to(self.device)
        next_state = next_state.to(self.device)
        
        # 如果是單個樣本，添加batch維度
        if len(state.shape) == 1:
            state = state.unsqueeze(0)
            action = action.unsqueeze(0)
            next_state = next_state.unsqueeze(0)
        
        # 計算內在獎勵
        intrinsic_reward = self.icm.get_intrinsic_reward(state, action, next_state)
        intrinsic_reward = intrinsic_reward.cpu().item() * self.intrinsic_reward_scale
        
        # 統計
        self.total_intrinsic_reward += intrinsic_reward
        
        # 增強獎勵
        enhanced_reward = extrinsic_reward + intrinsic_reward
        
        return enhanced_reward, intrinsic_reward
    
    def update_curiosity(self, state_batch, action_batch, next_state_batch):
        """
        更新好奇心模組
        """
        # 確保輸入是tensor格式並在正確設備上
        if not isinstance(state_batch, torch.Tensor):
            state_batch = torch.tensor(state_batch).float()
        if not isinstance(action_batch, torch.Tensor):
            action_batch = torch.tensor(action_batch).float()
        if not isinstance(next_state_batch, torch.Tensor):
            next_state_batch = torch.tensor(next_state_batch).float()
            
        state_batch = state_batch.to(self.device)
        action_batch = action_batch.to(self.device)
        next_state_batch = next_state_batch.to(self.device)
        
        forward_loss, inverse_loss, avg_intrinsic = self.icm.update(
            state_batch, action_batch, next_state_batch
        )
        
        self.update_count += 1
        
        return {
            'forward_loss': forward_loss,
            'inverse_loss': inverse_loss, 
            'avg_intrinsic_reward': avg_intrinsic,
            'total_intrinsic_reward': self.total_intrinsic_reward
        }
    
    def get_statistics(self):
        """獲取好奇心模組統計信息"""
        if self.update_count > 0:
            avg_intrinsic = self.total_intrinsic_reward / self.update_count
        else:
            avg_intrinsic = 0.0
            
        return {
            'total_intrinsic_reward': self.total_intrinsic_reward,
            'average_intrinsic_reward': avg_intrinsic,
            'update_count': self.update_count
        }
    
    def reset_statistics(self):
        """重置統計信息"""
        self.total_intrinsic_reward = 0.0
        self.update_count = 0


# 使用範例
if __name__ == "__main__":
    # 測試好奇心模組
    state_dim = 45
    action_dim = 12
    
    curiosity = CuriosityDrivenExploration(state_dim, action_dim)
    
    # 模擬一些數據
    state = torch.randn(1, state_dim)
    action = torch.randn(1, action_dim)
    next_state = torch.randn(1, state_dim)
    extrinsic_reward = -2.49
    
    enhanced_reward, intrinsic_reward = curiosity.get_enhanced_reward(
        state, action, next_state, extrinsic_reward
    )
    
    print(f"原始獎勵: {extrinsic_reward:.4f}")
    print(f"內在獎勵: {intrinsic_reward:.4f}")
    print(f"增強獎勵: {enhanced_reward:.4f}")