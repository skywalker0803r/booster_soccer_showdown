"""
Behavioral Cloning (BC) - 行為克隆
從專家軌跡學習策略，比強化學習更穩定可靠
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import pickle
import os
from datetime import datetime
from pathlib import Path

from sai_rl import SAIClient
from sai_compatible_dreamerv3 import SAICompatibleDreamerV3

class ExpertDataset(Dataset):
    """專家軌跡數據集"""
    
    def __init__(self, trajectories, sequence_length=10):
        self.trajectories = trajectories
        self.sequence_length = sequence_length
        self.samples = []
        
        print(f"🔄 處理 {len(trajectories)} 條軌跡...")
        
        for traj in trajectories:
            observations = np.array(traj['observations'])
            actions = np.array(traj['actions'])
            
            # 創建序列樣本
            for i in range(len(observations) - sequence_length):
                obs_seq = observations[i:i+sequence_length]
                action_seq = actions[i:i+sequence_length]
                
                self.samples.append({
                    'observations': obs_seq,
                    'actions': action_seq,
                    'next_action': actions[i+sequence_length-1]  # 預測目標
                })
        
        print(f"✅ 生成 {len(self.samples)} 個訓練樣本")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        return (
            torch.FloatTensor(sample['observations']),
            torch.FloatTensor(sample['actions']),
            torch.FloatTensor(sample['next_action'])
        )


class BehavioralCloningAgent(nn.Module):
    """行為克隆智能體"""
    
    def __init__(self, obs_dim, action_dim, hidden_dim=256, sequence_length=10):
        super().__init__()
        
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.sequence_length = sequence_length
        
        # 觀察編碼器
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
        
        # 序列處理器 (LSTM)
        self.lstm = nn.LSTM(
            input_size=hidden_dim + action_dim,  # obs + previous_action
            hidden_size=hidden_dim,
            num_layers=2,
            dropout=0.1,
            batch_first=True
        )
        
        # 動作預測器
        self.action_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim),
            nn.Tanh()  # 輸出 [-1, 1]
        )
        
        # 初始化權重
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, 0.01)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def forward(self, observations, actions):
        """
        observations: (batch, sequence_length, obs_dim)
        actions: (batch, sequence_length, action_dim)
        """
        batch_size, seq_len = observations.shape[:2]
        
        # 編碼觀察
        obs_flat = observations.reshape(-1, self.obs_dim)
        obs_encoded = self.obs_encoder(obs_flat)
        obs_encoded = obs_encoded.reshape(batch_size, seq_len, -1)
        
        # 準備LSTM輸入 (obs + previous_action)
        # 第一個時間步的previous_action設為0
        prev_actions = torch.cat([
            torch.zeros(batch_size, 1, self.action_dim, device=actions.device),
            actions[:, :-1]
        ], dim=1)
        
        lstm_input = torch.cat([obs_encoded, prev_actions], dim=-1)
        
        # LSTM處理
        lstm_out, _ = self.lstm(lstm_input)
        
        # 預測動作
        actions_pred = self.action_predictor(lstm_out)
        
        return actions_pred
    
    def select_action(self, obs, hidden_state=None, deterministic=True):
        """單步動作選擇"""
        self.eval()
        with torch.no_grad():
            if isinstance(obs, np.ndarray):
                obs = torch.FloatTensor(obs).unsqueeze(0).unsqueeze(0)  # (1, 1, obs_dim)
            
            # 編碼觀察
            obs_encoded = self.obs_encoder(obs.reshape(-1, self.obs_dim))
            obs_encoded = obs_encoded.reshape(1, 1, -1)
            
            # 如果沒有hidden_state，初始化
            if hidden_state is None:
                prev_action = torch.zeros(1, 1, self.action_dim)
                lstm_input = torch.cat([obs_encoded, prev_action], dim=-1)
                lstm_out, hidden_state = self.lstm(lstm_input)
            else:
                # 使用之前的hidden_state
                prev_action = hidden_state.get('prev_action', torch.zeros(1, 1, self.action_dim))
                lstm_input = torch.cat([obs_encoded, prev_action], dim=-1)
                lstm_out, (h, c) = self.lstm(lstm_input, (hidden_state['h'], hidden_state['c']))
                hidden_state = {'h': h, 'c': c}
            
            # 預測動作
            action = self.action_predictor(lstm_out).squeeze()
            
            # 更新hidden_state
            if hidden_state is None:
                hidden_state = {'h': lstm_out, 'c': lstm_out}
            hidden_state['prev_action'] = action.unsqueeze(0).unsqueeze(0)
            
            return action.cpu().numpy(), hidden_state


def train_behavioral_cloning():
    """訓練行為克隆模型"""
    
    print("🎯 行為克隆訓練開始")
    print("="*50)
    
    # 檢查專家軌跡
    expert_data_path = "expert_data/expert_trajectories.pkl"
    if not os.path.exists(expert_data_path):
        print("❌ 沒有找到專家軌跡數據！")
        print("請先運行 extract_expert_trajectories.py")
        return None
    
    # 載入專家軌跡
    with open(expert_data_path, 'rb') as f:
        trajectories = pickle.load(f)
    
    print(f"✅ 載入 {len(trajectories)} 條專家軌跡")
    
    # 過濾高質量軌跡
    good_trajectories = [t for t in trajectories if t['episode_reward'] > 0]
    if not good_trajectories:
        good_trajectories = [t for t in trajectories if t['episode_reward'] > -10]
    
    print(f"🎯 使用 {len(good_trajectories)} 條高質量軌跡")
    
    # 創建數據集
    sequence_length = 10
    dataset = ExpertDataset(good_trajectories, sequence_length)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, drop_last=True)
    
    # 創建模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = BehavioralCloningAgent(
        obs_dim=89,
        action_dim=12,
        hidden_dim=256,
        sequence_length=sequence_length
    ).to(device)
    
    # 優化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
    
    # TensorBoard
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    writer = SummaryWriter(f"runs/BehavioralCloning_{timestamp}")
    
    # 訓練循環
    num_epochs = 200
    best_loss = float('inf')
    
    print(f"🚀 開始訓練 {num_epochs} epochs...")
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        num_batches = 0
        
        for batch_idx, (obs_seq, action_seq, target_action) in enumerate(dataloader):
            obs_seq = obs_seq.to(device)
            action_seq = action_seq.to(device)
            target_action = target_action.to(device)
            
            optimizer.zero_grad()
            
            # 前向傳播
            predicted_actions = model(obs_seq, action_seq)
            
            # 使用最後一個時間步的預測
            predicted_final = predicted_actions[:, -1]
            
            # 損失函數
            loss = F.mse_loss(predicted_final, target_action)
            
            # 反向傳播
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
        
        scheduler.step()
        avg_loss = epoch_loss / num_batches
        
        # 記錄
        writer.add_scalar('Loss/Train', avg_loss, epoch)
        writer.add_scalar('Learning_Rate', scheduler.get_last_lr()[0], epoch)
        
        print(f"Epoch {epoch:3d}: Loss = {avg_loss:.6f}, LR = {scheduler.get_last_lr()[0]:.2e}")
        
        # 保存最佳模型
        if avg_loss < best_loss:
            best_loss = avg_loss
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
            }
            
            os.makedirs('saved_models/behavioral_cloning', exist_ok=True)
            torch.save(checkpoint, 'saved_models/behavioral_cloning/best_bc_model.pth')
            print(f"  💾 新的最佳模型已保存 (loss: {best_loss:.6f})")
        
        # 驗證
        if epoch % 20 == 0:
            print(f"  🔍 Epoch {epoch} 驗證...")
            evaluate_bc_model(model, device)
    
    writer.close()
    print(f"🎉 訓練完成！最佳損失: {best_loss:.6f}")
    
    return model

def evaluate_bc_model(model, device):
    """評估BC模型"""
    
    # 初始化環境
    sai = SAIClient(comp_id="booster-soccer-showdown", api_key="sai_LFcuaCZiqEkUbNVolQ3wbk5yU7H11jfv")
    env = sai.make_env()
    
    from main_improved_dreamerv3 import Preprocessor
    preprocessor = Preprocessor()
    
    model.eval()
    total_rewards = []
    
    # 測試5個episode
    for episode in range(5):
        obs, info = env.reset()
        obs = preprocessor.modify_state(obs, info).squeeze()
        
        episode_reward = 0
        hidden_state = None
        
        for step in range(400):
            action, hidden_state = model.select_action(obs, hidden_state)
            
            # 轉換動作
            env_action = env.action_space.low + (env.action_space.high - env.action_space.low) * (action + 1) / 2
            
            next_obs, reward, terminated, truncated, next_info = env.step(env_action)
            episode_reward += reward
            
            obs = preprocessor.modify_state(next_obs, next_info).squeeze()
            
            if terminated or truncated:
                break
        
        total_rewards.append(episode_reward)
        print(f"    Episode {episode}: {episode_reward:.3f}")
    
    avg_reward = np.mean(total_rewards)
    print(f"  📊 平均獎勵: {avg_reward:.3f}")
    
    return avg_reward

def action_function(policy):
    """BC模型的動作函數"""
    expected_bounds = [-1, 1]
    action_percent = (policy - expected_bounds[0]) / (expected_bounds[1] - expected_bounds[0])
    bounded_percent = np.minimum(np.maximum(action_percent, 0), 1)
    
    sai = SAIClient(comp_id="booster-soccer-showdown", api_key="sai_LFcuaCZiqEkUbNVolQ3wbk5yU7H11jfv")
    env = sai.make_env()
    return env.action_space.low + (env.action_space.high - env.action_space.low) * bounded_percent

if __name__ == "__main__":
    # 訓練BC模型
    bc_model = train_behavioral_cloning()
    
    if bc_model is not None:
        print("\n🎯 創建SAI提交模型...")
        
        # 創建SAI兼容wrapper
        sai_model = SAICompatibleDreamerV3(bc_model)
        
        # 測試
        sai = SAIClient(comp_id="booster-soccer-showdown", api_key="sai_LFcuaCZiqEkUbNVolQ3wbk5yU7H11jfv")
        from main_improved_dreamerv3 import Preprocessor
        
        print("🔍 本地基準測試...")
        sai.benchmark(sai_model, action_function, Preprocessor)
        
        print("🚀 提交到排行榜...")
        sai.submit("Vedanta_BehavioralCloning", sai_model, action_function, Preprocessor)
        
        print("🎉 完成！BC模型應該比RL更穩定可靠！")