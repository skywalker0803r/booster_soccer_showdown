import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random

# 設備配置
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class RNDNetwork(nn.Module):
    """RND 網絡：包含目標網絡和預測網絡"""
    
    def __init__(self, input_dim, hidden_dim=256):
        super(RNDNetwork, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # 目標網絡（隨機初始化，不更新）
        self.target_network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # 預測網絡（需要訓練）
        self.predictor_network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # 凍結目標網絡參數
        for param in self.target_network.parameters():
            param.requires_grad = False
            
    def forward(self, state):
        """前向傳播"""
        target_output = self.target_network(state)
        predictor_output = self.predictor_network(state)
        return target_output, predictor_output


class RNDModule:
    """Random Network Distillation 模組"""
    
    def __init__(self, input_dim, hidden_dim=256, lr=1e-4, intrinsic_reward_scale=1.0):
        self.input_dim = input_dim
        self.intrinsic_reward_scale = intrinsic_reward_scale
        
        # 初始化 RND 網絡
        self.rnd_network = RNDNetwork(input_dim, hidden_dim).to(DEVICE)
        self.optimizer = optim.Adam(self.rnd_network.predictor_network.parameters(), lr=lr)
        self.criterion = nn.MSELoss()
        
        # 觀測歸一化相關
        self.running_mean = np.zeros(input_dim)
        self.running_var = np.ones(input_dim)
        self.obs_count = 0
        self.epsilon = 1e-8
        
        # 內在獎勵歸一化
        self.reward_history = deque(maxlen=1000)
        
        print(f"RND 模組初始化完成 - 輸入維度: {input_dim}, 隱藏層維度: {hidden_dim}")
    
    def normalize_observation(self, obs):
        """歸一化觀測"""
        if isinstance(obs, torch.Tensor):
            obs_np = obs.cpu().numpy()
        else:
            obs_np = np.array(obs)
        
        # 確保觀測是一維的
        if len(obs_np.shape) > 1:
            obs_np = obs_np.flatten()
            
        # 更新運行統計
        self.obs_count += 1
        delta = obs_np - self.running_mean
        self.running_mean += delta / self.obs_count
        self.running_var += delta * (obs_np - self.running_mean)
        
        # 歸一化
        if self.obs_count > 1:
            std = np.sqrt(self.running_var / (self.obs_count - 1) + self.epsilon)
            normalized_obs = (obs_np - self.running_mean) / std
        else:
            normalized_obs = obs_np
            
        return torch.FloatTensor(normalized_obs).to(DEVICE)
    
    def compute_intrinsic_reward(self, state):
        """計算內在獎勵（好奇心獎勵）"""
        with torch.no_grad():
            # 歸一化觀測 (normalize_observation 內部已處理維度)
            normalized_state = self.normalize_observation(state)
            
            # 轉換為張量並確保正確的批次維度
            if not isinstance(normalized_state, torch.Tensor):
                normalized_state = torch.FloatTensor(normalized_state).to(DEVICE)
            
            # 確保有批次維度
            if len(normalized_state.shape) == 1:
                normalized_state = normalized_state.unsqueeze(0)
            
            # 計算預測誤差
            target_output, predictor_output = self.rnd_network(normalized_state)
            prediction_error = torch.mean((target_output - predictor_output) ** 2, dim=-1)
            
            # 轉換為 numpy 並記錄
            intrinsic_reward = prediction_error.cpu().numpy()
            self.reward_history.extend(intrinsic_reward)
            
            # 歸一化內在獎勵
            if len(self.reward_history) > 10:
                mean_reward = np.mean(self.reward_history)
                std_reward = np.std(self.reward_history) + self.epsilon
                intrinsic_reward = (intrinsic_reward - mean_reward) / std_reward
            
            return intrinsic_reward * self.intrinsic_reward_scale
    
    def update(self, states, batch_size=32):
        """更新 RND 網絡"""
        if len(states) < batch_size:
            return 0.0
            
        # 隨機採樣批次
        batch_indices = random.sample(range(len(states)), min(batch_size, len(states)))
        batch_states = [states[i] for i in batch_indices]
        
        # 轉換為張量並歸一化
        normalized_states = []
        for state in batch_states:
            norm_state = self.normalize_observation(state)
            if not isinstance(norm_state, torch.Tensor):
                norm_state = torch.FloatTensor(norm_state)
            normalized_states.append(norm_state)
        
        state_tensor = torch.stack(normalized_states).to(DEVICE)
        
        # 前向傳播
        target_output, predictor_output = self.rnd_network(state_tensor)
        
        # 計算損失
        loss = self.criterion(predictor_output, target_output.detach())
        
        # 反向傳播
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def get_statistics(self):
        """獲取統計資訊"""
        if len(self.reward_history) > 0:
            return {
                'mean_intrinsic_reward': np.mean(self.reward_history),
                'std_intrinsic_reward': np.std(self.reward_history),
                'obs_count': self.obs_count,
                'reward_scale': self.intrinsic_reward_scale
            }
        return {
            'mean_intrinsic_reward': 0.0,
            'std_intrinsic_reward': 0.0,
            'obs_count': self.obs_count,
            'reward_scale': self.intrinsic_reward_scale
        }
    
    def save(self, filepath):
        """保存 RND 模型"""
        torch.save({
            'rnd_network_state_dict': self.rnd_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'running_mean': self.running_mean,
            'running_var': self.running_var,
            'obs_count': self.obs_count,
            'reward_history': list(self.reward_history)
        }, filepath)
        print(f"RND 模型已保存到: {filepath}")
    
    def load(self, filepath):
        """加載 RND 模型"""
        checkpoint = torch.load(filepath, map_location=DEVICE)
        self.rnd_network.load_state_dict(checkpoint['rnd_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.running_mean = checkpoint['running_mean']
        self.running_var = checkpoint['running_var']
        self.obs_count = checkpoint['obs_count']
        self.reward_history = deque(checkpoint['reward_history'], maxlen=1000)
        print(f"RND 模型已從 {filepath} 加載")


class RNDBuffer:
    """RND 觀測緩存，用於批次更新"""
    
    def __init__(self, maxlen=10000):
        self.buffer = deque(maxlen=maxlen)
    
    def push(self, state):
        """添加狀態到緩存"""
        self.buffer.append(state)
    
    def sample(self, batch_size):
        """採樣批次狀態"""
        if len(self.buffer) < batch_size:
            return list(self.buffer)
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)


# 使用範例
if __name__ == "__main__":
    # 測試 RND 模組
    input_dim = 89  # 假設觀測維度
    rnd = RNDModule(input_dim, intrinsic_reward_scale=0.1)
    
    # 模擬一些觀測
    for i in range(100):
        fake_obs = torch.randn(input_dim)
        intrinsic_reward = rnd.compute_intrinsic_reward(fake_obs)
        print(f"步驟 {i}: 內在獎勵 = {intrinsic_reward[0]:.4f}")
        
        # 更新網絡
        if i % 10 == 0 and i > 0:
            states = [torch.randn(input_dim) for _ in range(32)]
            loss = rnd.update(states)
            stats = rnd.get_statistics()
            print(f"更新 - 損失: {loss:.4f}, 平均內在獎勵: {stats['mean_intrinsic_reward']:.4f}")