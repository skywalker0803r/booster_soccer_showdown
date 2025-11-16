"""
DAgger (Dataset Aggregation) - 改進的模仿學習
結合在線收集和離線訓練，比純BC更強大
"""

import torch
import torch.nn as nn
import numpy as np
from collections import deque
import pickle
import os
from datetime import datetime

from behavioral_cloning import BehavioralCloningAgent, ExpertDataset
from sai_rl import SAIClient

class DAggerAgent:
    """DAgger智能體"""
    
    def __init__(self, obs_dim, action_dim, initial_trajectories=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 行為克隆模型
        self.bc_model = BehavioralCloningAgent(obs_dim, action_dim).to(self.device)
        self.optimizer = torch.optim.AdamW(self.bc_model.parameters(), lr=1e-4)
        
        # 數據聚合
        self.all_trajectories = initial_trajectories or []
        self.iteration = 0
        
        # 環境
        self.sai = SAIClient(comp_id="booster-soccer-showdown", api_key="sai_LFcuaCZiqEkUbNVolQ3wbk5yU7H11jfv")
        self.env = self.sai.make_env()
        
        from main_improved_dreamerv3 import Preprocessor
        self.preprocessor = Preprocessor()
        
        print("🎯 DAgger Agent 初始化完成")
    
    def collect_trajectories_with_expert_labels(self, num_episodes=10):
        """使用當前策略收集軌跡，並用專家策略標記"""
        
        print(f"🎮 DAgger Iteration {self.iteration}: 收集 {num_episodes} 個episode...")
        
        new_trajectories = []
        
        for episode in range(num_episodes):
            obs, info = self.env.reset()
            obs = self.preprocessor.modify_state(obs, info).squeeze()
            
            trajectory = {
                'observations': [],
                'expert_actions': [],  # 專家標記的動作
                'policy_actions': [],  # 當前策略的動作
                'rewards': [],
                'episode_reward': 0
            }
            
            hidden_state = None
            
            for step in range(400):
                trajectory['observations'].append(obs.copy())
                
                # 當前策略動作
                if self.iteration == 0 or np.random.random() < 0.3:
                    # 前幾次迭代或隨機時使用專家策略
                    expert_action = self._get_expert_action(obs)
                    policy_action = expert_action
                else:
                    # 使用當前策略
                    policy_action, hidden_state = self.bc_model.select_action(obs, hidden_state)
                    expert_action = self._get_expert_action(obs)
                
                trajectory['policy_actions'].append(policy_action.copy())
                trajectory['expert_actions'].append(expert_action.copy())
                
                # 執行動作（使用策略動作，但學習專家動作）
                env_action = self.env.action_space.low + (self.env.action_space.high - self.env.action_space.low) * (policy_action + 1) / 2
                next_obs, reward, terminated, truncated, next_info = self.env.step(env_action)
                
                trajectory['rewards'].append(reward)
                trajectory['episode_reward'] += reward
                
                obs = self.preprocessor.modify_state(next_obs, next_info).squeeze()
                
                if terminated or truncated:
                    break
            
            new_trajectories.append(trajectory)
            print(f"   Episode {episode}: {trajectory['episode_reward']:.3f}")
        
        # 加入數據集
        self.all_trajectories.extend(new_trajectories)
        print(f"📊 總軌跡數: {len(self.all_trajectories)}")
        
        return new_trajectories
    
    def _get_expert_action(self, obs):
        """獲取專家動作（這裡使用啟發式專家）"""
        
        # 簡單的啟發式專家策略
        # 目標：保持穩定 + 朝球移動
        
        action = np.zeros(12)
        
        # 添加小的隨機擾動保持穩定
        action += np.random.normal(0, 0.05, 12)
        
        # 限制動作範圍
        action = np.clip(action, -0.3, 0.3)
        
        return action
    
    def train_on_aggregated_data(self, num_epochs=50):
        """在聚合數據上訓練"""
        
        print(f"🔄 在聚合數據上訓練...")
        
        # 轉換軌跡格式為BC格式
        bc_trajectories = []
        for traj in self.all_trajectories:
            bc_traj = {
                'observations': traj['observations'],
                'actions': traj['expert_actions'],  # 學習專家動作
                'episode_reward': traj['episode_reward']
            }
            bc_trajectories.append(bc_traj)
        
        # 創建數據集
        dataset = ExpertDataset(bc_trajectories, sequence_length=5)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
        
        # 訓練
        self.bc_model.train()
        for epoch in range(num_epochs):
            epoch_loss = 0
            num_batches = 0
            
            for obs_seq, action_seq, target_action in dataloader:
                obs_seq = obs_seq.to(self.device)
                action_seq = action_seq.to(self.device)
                target_action = target_action.to(self.device)
                
                self.optimizer.zero_grad()
                
                predicted_actions = self.bc_model(obs_seq, action_seq)
                loss = torch.nn.functional.mse_loss(predicted_actions[:, -1], target_action)
                
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
            
            if epoch % 10 == 0:
                print(f"   Epoch {epoch}: Loss = {epoch_loss/num_batches:.6f}")
    
    def run_dagger_iteration(self):
        """運行一次DAgger迭代"""
        
        print(f"\n🎯 DAgger Iteration {self.iteration}")
        print("="*50)
        
        # 1. 收集新軌跡
        new_trajectories = self.collect_trajectories_with_expert_labels(num_episodes=20)
        
        # 2. 在聚合數據上訓練
        self.train_on_aggregated_data(num_epochs=30)
        
        # 3. 評估當前策略
        avg_reward = self.evaluate_policy()
        
        # 4. 保存模型
        self.save_checkpoint()
        
        self.iteration += 1
        
        return avg_reward
    
    def evaluate_policy(self, num_episodes=5):
        """評估當前策略"""
        
        print(f"📊 評估策略...")
        
        self.bc_model.eval()
        rewards = []
        
        for episode in range(num_episodes):
            obs, info = self.env.reset()
            obs = self.preprocessor.modify_state(obs, info).squeeze()
            
            episode_reward = 0
            hidden_state = None
            
            for step in range(400):
                action, hidden_state = self.bc_model.select_action(obs, hidden_state)
                
                env_action = self.env.action_space.low + (self.env.action_space.high - self.env.action_space.low) * (action + 1) / 2
                next_obs, reward, terminated, truncated, next_info = self.env.step(env_action)
                
                episode_reward += reward
                obs = self.preprocessor.modify_state(next_obs, next_info).squeeze()
                
                if terminated or truncated:
                    break
            
            rewards.append(episode_reward)
            print(f"   Episode {episode}: {episode_reward:.3f}")
        
        avg_reward = np.mean(rewards)
        print(f"📈 平均獎勵: {avg_reward:.3f}")
        
        return avg_reward
    
    def save_checkpoint(self):
        """保存檢查點"""
        
        os.makedirs('saved_models/dagger', exist_ok=True)
        
        checkpoint = {
            'iteration': self.iteration,
            'model_state_dict': self.bc_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'trajectories_count': len(self.all_trajectories)
        }
        
        torch.save(checkpoint, f'saved_models/dagger/dagger_iter_{self.iteration}.pth')
        print(f"💾 已保存檢查點: iteration {self.iteration}")

def run_dagger_training():
    """運行完整的DAgger訓練"""
    
    print("🚀 DAgger訓練開始")
    print("="*60)
    
    # 檢查是否有初始專家軌跡
    initial_trajectories = []
    expert_data_path = "expert_data/expert_trajectories.pkl"
    if os.path.exists(expert_data_path):
        with open(expert_data_path, 'rb') as f:
            initial_trajectories = pickle.load(f)
        print(f"✅ 載入 {len(initial_trajectories)} 條初始專家軌跡")
    
    # 創建DAgger agent
    agent = DAggerAgent(obs_dim=89, action_dim=12, initial_trajectories=initial_trajectories)
    
    # 運行多次迭代
    num_iterations = 10
    best_reward = float('-inf')
    
    for iteration in range(num_iterations):
        avg_reward = agent.run_dagger_iteration()
        
        if avg_reward > best_reward:
            best_reward = avg_reward
            print(f"🎉 新的最佳性能: {best_reward:.3f}")
        
        # 早停條件
        if avg_reward > 0:  # 如果獲得正獎勵
            print(f"🎯 達到正獎勵，訓練完成！")
            break
    
    print(f"\n🏆 DAgger訓練完成！")
    print(f"   最佳獎勵: {best_reward:.3f}")
    print(f"   總迭代數: {agent.iteration}")
    
    return agent

if __name__ == "__main__":
    # 運行DAgger訓練
    agent = run_dagger_training()
    
    print("\n🎯 創建最終提交...")
    
    # 提交到SAI
    from sai_compatible_dreamerv3 import SAICompatibleDreamerV3
    from main_improved_dreamerv3 import Preprocessor
    
    sai_model = SAICompatibleDreamerV3(agent.bc_model)
    
    def action_function(policy):
        expected_bounds = [-1, 1]
        action_percent = (policy - expected_bounds[0]) / (expected_bounds[1] - expected_bounds[0])
        bounded_percent = np.minimum(np.maximum(action_percent, 0), 1)
        env = agent.env
        return env.action_space.low + (env.action_space.high - env.action_space.low) * bounded_percent
    
    print("🔍 本地測試...")
    agent.sai.benchmark(sai_model, action_function, Preprocessor)
    
    print("🚀 提交到排行榜...")
    agent.sai.submit("Vedanta_DAgger", sai_model, action_function, Preprocessor)
    
    print("🎉 DAgger模型提交完成！")