"""
簡化版BC整合：為您現有的main.py添加專家數據預訓練功能
只需最小修改即可整合到您現有的Research/main.py
"""

import numpy as np
import torch
import torch.nn as nn
import os

class ExpertDataLoader:
    """專家數據載入和格式轉換器"""
    
    def __init__(self, expert_data_path):
        self.expert_data_path = expert_data_path
        self.expert_data = self._load_and_convert_data()
    
    def _load_and_convert_data(self):
        """載入並轉換專家數據格式"""
        if not os.path.exists(self.expert_data_path):
            print(f"❌ 專家數據不存在: {self.expert_data_path}")
            return None
            
        print(f"📚 載入專家數據: {self.expert_data_path}")
        data = np.load(self.expert_data_path, allow_pickle=True)
        
        il_observations = data['observations']  # 89維
        expert_actions = data['actions']       # 12維
        
        # 簡化轉換：從89維提取45維
        # 基於IL preprocessor的結構：前42維是機器人狀態，後3維是任務編碼
        converted_observations = []
        
        for obs in il_observations:
            # 提取核心42維機器人狀態
            robot_state = obs[:42]
            task_encoding = obs[-3:]  # 任務編碼
            
            # 構造45維觀測 (42 + 3 = 45)
            # 這裡我們直接使用前42維 + 任務編碼，省略複雜的轉換
            research_obs = np.concatenate([robot_state, task_encoding])
            converted_observations.append(research_obs)
        
        converted_observations = np.array(converted_observations, dtype=np.float32)
        expert_actions = np.array(expert_actions, dtype=np.float32)
        
        print(f"✅ 專家數據載入成功:")
        print(f"   觀測: {converted_observations.shape} (89維→45維)")
        print(f"   動作: {expert_actions.shape}")
        print(f"   Episodes: {np.sum(data['done'])}")
        
        return {
            'observations': converted_observations,
            'actions': expert_actions,
            'episode_count': int(np.sum(data['done']))
        }
    
    def get_data(self):
        """獲取處理後的專家數據"""
        return self.expert_data

class BCPretrainer:
    """行為克隆預訓練器 - 專為您的PPO-CMA系統設計"""
    
    def __init__(self, ppo_agent, expert_data_path, device):
        self.ppo_agent = ppo_agent
        self.device = device
        self.expert_loader = ExpertDataLoader(expert_data_path)
        self.expert_data = self.expert_loader.get_data()
        
        if self.expert_data is None:
            raise ValueError("無法載入專家數據")
        
        # BC專用優化器（只訓練actor）
        self.bc_optimizer = torch.optim.Adam(
            self.ppo_agent.actor.parameters(),
            lr=1e-4,
            weight_decay=1e-5
        )
        
        self.loss_fn = nn.MSELoss()
    
    def pretrain(self, epochs=50, batch_size=256, print_interval=10):
        """執行BC預訓練"""
        if self.expert_data is None:
            print("❌ 沒有專家數據，跳過BC預訓練")
            return None
        
        print(f"🎯 開始BC預訓練 ({epochs} epochs)")
        print(f"   數據量: {len(self.expert_data['observations'])} 樣本")
        print(f"   批次大小: {batch_size}")
        
        observations = torch.tensor(self.expert_data['observations']).to(self.device)
        actions = torch.tensor(self.expert_data['actions']).to(self.device)
        
        dataset_size = len(observations)
        best_loss = float('inf')
        
        # 設置為訓練模式
        self.ppo_agent.actor.train()
        
        for epoch in range(epochs):
            epoch_loss = 0
            num_batches = 0
            
            # 隨機打亂數據
            indices = torch.randperm(dataset_size)
            
            for i in range(0, dataset_size, batch_size):
                end_idx = min(i + batch_size, dataset_size)
                batch_indices = indices[i:end_idx]
                
                batch_obs = observations[batch_indices]
                batch_actions = actions[batch_indices]
                
                # 前向傳播 - 使用actor網絡 (只取mean，忽略log_std)
                predicted_actions, _ = self.ppo_agent.actor(batch_obs)
                
                # 計算損失
                loss = self.loss_fn(predicted_actions, batch_actions)
                
                # 反向傳播
                self.bc_optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.ppo_agent.actor.parameters(), 1.0)
                self.bc_optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
            
            avg_loss = epoch_loss / num_batches
            
            # 保存最佳模型
            if avg_loss < best_loss:
                best_loss = avg_loss
                self.save_bc_model(epoch)
            
            # 定期輸出
            if epoch % print_interval == 0:
                print(f"   Epoch {epoch:3d}: BC Loss = {avg_loss:.6f} (Best: {best_loss:.6f})")
        
        print(f"✅ BC預訓練完成! 最終損失: {best_loss:.6f}")
        
        # 恢復為evaluation模式
        self.ppo_agent.actor.eval()
        
        return best_loss
    
    def save_bc_model(self, epoch):
        """保存BC預訓練模型"""
        save_path = f"bc_pretrained_actor_epoch_{epoch}.pth"
        torch.save({
            'actor_state_dict': self.ppo_agent.actor.state_dict(),
            'epoch': epoch,
            'expert_episodes': self.expert_data['episode_count']
        }, save_path)
    
    def evaluate_bc_performance(self, num_samples=1000):
        """評估BC性能"""
        if self.expert_data is None:
            return None
        
        self.ppo_agent.actor.eval()
        
        with torch.no_grad():
            # 隨機選擇樣本
            indices = torch.randperm(len(self.expert_data['observations']))[:num_samples]
            
            test_obs = torch.tensor(self.expert_data['observations'][indices]).to(self.device)
            test_actions = torch.tensor(self.expert_data['actions'][indices]).to(self.device)
            
            # 預測動作 (只取mean，忽略log_std)
            predicted_actions, _ = self.ppo_agent.actor(test_obs)
            
            # 計算各種誤差
            mse = nn.MSELoss()(predicted_actions, test_actions).item()
            mae = torch.mean(torch.abs(predicted_actions - test_actions)).item()
            
            # 計算每個動作維度的相關係數
            correlations = []
            for dim in range(test_actions.shape[1]):
                pred_dim = predicted_actions[:, dim].cpu().numpy()
                true_dim = test_actions[:, dim].cpu().numpy()
                corr = np.corrcoef(pred_dim, true_dim)[0, 1]
                correlations.append(corr if not np.isnan(corr) else 0)
            
            avg_correlation = np.mean(correlations)
            
        return {
            'mse': mse,
            'mae': mae,
            'avg_correlation': avg_correlation,
            'correlations_per_dim': correlations
        }

def add_bc_pretraining_to_main(expert_data_path="../data/dataset_kick.npz"):
    """
    這個函數展示如何將BC預訓練添加到您現有的main.py
    您只需要在main.py中添加幾行代碼即可
    """
    code_snippet = f'''
# 在您的main.py中，在創建PPO-CMA agent之後添加以下代碼:

# === BC預訓練集成 ===
from simple_bc_integration import BCPretrainer

# 檢查是否有專家數據
expert_data_path = "{expert_data_path}"
if os.path.exists(expert_data_path):
    print("🎯 發現專家數據，開始BC預訓練...")
    
    # 創建BC預訓練器
    bc_pretrainer = BCPretrainer(ppo_cma_agent, expert_data_path, device)
    
    # 執行預訓練
    bc_loss = bc_pretrainer.pretrain(epochs=50)
    
    # 評估BC性能
    bc_performance = bc_pretrainer.evaluate_bc_performance()
    if bc_performance:
        print(f"📊 BC性能評估:")
        print(f"   MSE: {{bc_performance['mse']:.6f}}")
        print(f"   MAE: {{bc_performance['mae']:.6f}}")  
        print(f"   平均相關係數: {{bc_performance['avg_correlation']:.4f}}")
    
    print("✅ BC預訓練完成，開始PPO-CMA訓練...")
else:
    print("⚠️ 未找到專家數據，直接開始PPO-CMA訓練...")

# 然後繼續您原有的訓練循環...
'''
    
    print("📋 整合代碼片段:")
    print(code_snippet)
    
    return code_snippet

if __name__ == "__main__":
    # 示範如何整合
    add_bc_pretraining_to_main()
    
    # 如果要測試BC預訓練
    expert_data_path = "../data/dataset_kick.npz"
    if os.path.exists(expert_data_path):
        print("🧪 測試BC預訓練...")
        
        # 這裡需要您的PPO agent，這只是示範
        # bc_pretrainer = BCPretrainer(your_ppo_agent, expert_data_path, device)
        # bc_pretrainer.pretrain(epochs=10)
    else:
        print(f"請確保專家數據存在: {expert_data_path}")