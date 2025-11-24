"""
混合訓練系統：結合行為克隆預訓練 + PPO-CMA在線微調
整合專家數據預訓練與您現有的PPO-CMA系統
"""

import numpy as np
import torch
import torch.nn as nn
import copy
from sai_rl import SAIClient 
from ppo_cma_model import PPOCMA
from utils import Preprocessor
from logger import TensorBoardLogger
from curiosity_module import CuriosityDrivenExploration
import sys
sys.path.append('..')
from llm_coach import LLMCoach
from reward_shaper import RewardShaper

class ExpertDataConverter:
    """將imitation learning數據轉換為Research系統格式"""
    
    def __init__(self):
        self.il_preprocessor = self._create_il_preprocessor()
        self.research_preprocessor = Preprocessor()
    
    def _create_il_preprocessor(self):
        """創建IL預處理器"""
        # 導入IL系統的預處理器
        import sys
        sys.path.append('../imitation_learning/scripts')
        from preprocessor import Preprocessor as ILPreprocessor
        return ILPreprocessor()
    
    def convert_observations(self, il_observations):
        """
        將89維IL觀測轉換為45維Research格式
        
        IL格式(89維): [機器人狀態42 + 球信息6 + 目標信息38 + 任務編碼3]
        Research格式(45維): [關節12 + 速度12 + 重力3 + 傳感器9 + 球6 + 任務3]
        """
        converted_obs = []
        
        for obs in il_observations:
            # 從IL觀測中提取基礎信息
            robot_qpos = obs[:12]      # 關節位置 
            robot_qvel = obs[12:24]    # 關節速度
            
            # 提取其他必要信息（需要重構或近似）
            # 由於IL預處理器包含更多信息，我們需要選擇性提取
            proj_gravity = obs[24:27]   # 假設位置3-6是重力投影
            ball_pos = obs[30:33]       # 球的相對位置
            ball_vel = obs[33:36]       # 球的速度
            task_onehot = obs[-3:]      # 任務編碼
            
            # 構造45維向量 (需要補充缺失的傳感器數據)
            # 對於缺失的傳感器數據，使用零填充或從現有數據推斷
            gyro = np.zeros(3)         # 陀螺儀數據 - 需要從原始環境獲取
            accel = np.zeros(3)        # 加速度計數據
            velo = np.zeros(3)         # 速度計數據
            
            research_obs = np.concatenate([
                robot_qpos,    # 12維
                robot_qvel,    # 12維
                proj_gravity,  # 3維
                gyro,          # 3維
                accel,         # 3維  
                velo,          # 3維
                ball_pos,      # 3維
                ball_vel,      # 3維
                task_onehot    # 3維
            ])
            
            converted_obs.append(research_obs)
            
        return np.array(converted_obs)

class BehavioralCloningPretrainer:
    """行為克隆預訓練器"""
    
    def __init__(self, ppo_agent, expert_data_path):
        self.ppo_agent = ppo_agent
        self.expert_data = self._load_expert_data(expert_data_path)
        self.converter = ExpertDataConverter()
        self.bc_loss_fn = nn.MSELoss()
        
        # 創建BC專用優化器
        self.bc_optimizer = torch.optim.Adam(
            ppo_agent.actor.parameters(), 
            lr=1e-4  # BC學習率
        )
        
    def _load_expert_data(self, data_path):
        """載入並轉換專家數據"""
        print(f"📚 載入專家數據: {data_path}")
        data = np.load(data_path, allow_pickle=True)
        
        # 轉換觀測格式
        il_observations = data['observations']
        converted_obs = self.converter.convert_observations(il_observations)
        
        expert_actions = np.array(data['actions'])
        
        print(f"✅ 專家數據載入完成:")
        print(f"   觀測數: {len(converted_obs)} (從89維轉換為45維)")
        print(f"   動作數: {len(expert_actions)}")
        print(f"   Episodes: {np.sum(data['done'])}")
        
        return {
            'observations': converted_obs,
            'actions': expert_actions,
            'done': data['done']
        }
    
    def pretrain(self, epochs=100, batch_size=256):
        """執行行為克隆預訓練"""
        print(f"🎯 開始行為克隆預訓練 ({epochs} epochs)")
        
        observations = torch.tensor(self.expert_data['observations'], dtype=torch.float32)
        actions = torch.tensor(self.expert_data['actions'], dtype=torch.float32)
        
        dataset_size = len(observations)
        device = next(self.ppo_agent.actor.parameters()).device
        observations = observations.to(device)
        actions = actions.to(device)
        
        best_bc_loss = float('inf')
        
        for epoch in range(epochs):
            epoch_loss = 0
            num_batches = 0
            
            # 隨機打亂數據
            indices = torch.randperm(dataset_size)
            
            for i in range(0, dataset_size, batch_size):
                batch_indices = indices[i:i+batch_size]
                batch_obs = observations[batch_indices]
                batch_actions = actions[batch_indices]
                
                # 前向傳播
                predicted_actions = self.ppo_agent.actor(batch_obs)
                
                # 計算BC損失
                bc_loss = self.bc_loss_fn(predicted_actions, batch_actions)
                
                # 反向傳播
                self.bc_optimizer.zero_grad()
                bc_loss.backward()
                self.bc_optimizer.step()
                
                epoch_loss += bc_loss.item()
                num_batches += 1
            
            avg_loss = epoch_loss / num_batches
            
            if avg_loss < best_bc_loss:
                best_bc_loss = avg_loss
                # 保存最佳BC模型
                torch.save(self.ppo_agent.actor.state_dict(), 'best_bc_pretrained_actor.pth')
            
            if epoch % 10 == 0:
                print(f"   Epoch {epoch:3d}: BC Loss = {avg_loss:.6f} (Best: {best_bc_loss:.6f})")
        
        print(f"✅ 行為克隆預訓練完成! 最佳損失: {best_bc_loss:.6f}")
        return best_bc_loss

class HybridTrainer:
    """混合訓練器：BC預訓練 + PPO微調"""
    
    def __init__(self, expert_data_path):
        self.expert_data_path = expert_data_path
        self.setup_environment()
        self.setup_agents()
        
    def setup_environment(self):
        """設置環境（沿用您原有的設置）"""
        self.sai = SAIClient(
            comp_id="booster-soccer-showdown", 
            api_key="sai_LFcuaCZiqEkUbNVolQ3wbk5yU7H11jfv",
        )
        self.env = self.sai.make_env()
        print(f"環境已創建。觀察空間: {self.env.observation_space} | 動作空間: {self.env.action_space}")
        
    def setup_agents(self):
        """設置智能體（沿用您原有的配置）"""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # PPO-CMA配置（從您的main.py）
        self.ppo_cma_agent = PPOCMA(
            state_dim=45,
            action_dim=self.env.action_space.shape[0],
            lr_actor=3e-4,
            lr_critic=1e-3,
            gamma=0.99,
            gae_lambda=0.95,
            clip_epsilon=0.2,
            entropy_coef=0.01,
            hidden_layers=[512, 512, 256],
            buffer_capacity=8192,
            batch_size=1024,
            ppo_epochs=10,
            max_grad_norm=0.5,
            cma_population_size=64,
            cma_sigma=0.1,
            device=device
        )
        
        # 其他組件
        self.curiosity_explorer = CuriosityDrivenExploration(
            state_dim=45,
            action_dim=self.env.action_space.shape[0],
            device=device
        )
        
        self.llm_coach = LLMCoach()
        self.reward_shaper = RewardShaper()
        
        # BC預訓練器
        self.bc_pretrainer = BehavioralCloningPretrainer(
            self.ppo_cma_agent, 
            self.expert_data_path
        )
    
    def train(self, bc_epochs=100, rl_timesteps=1000000):
        """執行混合訓練"""
        print(f"🚀 開始混合訓練：BC預訓練 + PPO-CMA微調")
        
        # 階段1：行為克隆預訓練
        print(f"\n📚 === 階段1：行為克隆預訓練 ===")
        bc_loss = self.bc_pretrainer.pretrain(epochs=bc_epochs)
        
        # 階段2：PPO-CMA在線微調（沿用您的訓練循環）
        print(f"\n🎯 === 階段2：PPO-CMA在線微調 ===")
        self.run_ppo_training(rl_timesteps, initial_bc_loss=bc_loss)
    
    def run_ppo_training(self, total_timesteps, initial_bc_loss):
        """運行PPO訓練（基於您的main.py邏輯）"""
        # 設置logger
        logger = TensorBoardLogger(f"hybrid_bc_ppo_training")
        
        # 初始化變量
        episode_count = 0
        best_reward = -np.inf
        device = self.ppo_cma_agent.device
        
        # 權重配置
        current_weights = {
            'extrinsic': 0.6,   # 降低外在獎勵權重，因為有BC指導
            'intrinsic': 0.3,   # 好奇心獎勵
            'shaped': 0.1       # LLM塑形獎勵
        }
        
        print(f"🎯 開始PPO-CMA微調，初始BC損失: {initial_bc_loss:.6f}")
        print(f"⚖️ 權重配置: {current_weights}")
        
        # 訓練循環
        current_obs, info = self.env.reset()
        state = Preprocessor().modify_state(current_obs, info)[0]
        state = torch.tensor(state).float().to(device)
        
        for t in range(1, total_timesteps + 1):
            # PPO動作選擇（已有BC預訓練的基礎）
            action_probs, values, log_probs = self.ppo_cma_agent.forward(state)
            action = self.ppo_cma_agent.get_action(state.unsqueeze(0)).squeeze()
            
            # 執行動作
            bounded_action = self.action_function(action.cpu().numpy())
            next_obs, extrinsic_reward, done, _, info = self.env.step(bounded_action)
            
            # 獎勵塑形
            intrinsic_reward = self.curiosity_explorer.compute_intrinsic_reward(
                state.unsqueeze(0), action.unsqueeze(0)
            )
            
            shaped_reward = self.reward_shaper.shape_reward(
                extrinsic_reward, state.cpu().numpy(), action.cpu().numpy(), info
            )
            
            # 組合獎勵
            total_reward = (
                current_weights['extrinsic'] * extrinsic_reward + 
                current_weights['intrinsic'] * intrinsic_reward +
                current_weights['shaped'] * shaped_reward
            )
            
            # 處理下一狀態
            if not done:
                next_state = Preprocessor().modify_state(next_obs, info)[0]
                next_state = torch.tensor(next_state).float().to(device)
            else:
                next_state = None
            
            # 存儲經驗
            self.ppo_cma_agent.store_transition(
                state, action, total_reward, next_state, done, log_probs, values
            )
            
            # 更新智能體
            if self.ppo_cma_agent.should_update():
                ppo_info = self.ppo_cma_agent.update()
                curiosity_info = self.curiosity_explorer.update(
                    self.ppo_cma_agent.buffer.states,
                    self.ppo_cma_agent.buffer.actions
                )
                
                # 記錄訓練信息
                logger.log({
                    'ppo/policy_loss': ppo_info.get('policy_loss', 0),
                    'ppo/value_loss': ppo_info.get('value_loss', 0),
                    'curiosity/intrinsic_reward': intrinsic_reward,
                    'reward/extrinsic': extrinsic_reward,
                    'reward/shaped': shaped_reward,
                    'reward/total': total_reward,
                    'training/bc_initialization': initial_bc_loss
                }, step=t)
            
            # 處理episode結束
            if done:
                episode_count += 1
                current_obs, info = self.env.reset()
                state = Preprocessor().modify_state(current_obs, info)[0]
                state = torch.tensor(state).float().to(device)
                
                print(f"Episode {episode_count}: 總獎勵 = {total_reward:.2f}")
            else:
                state = next_state
            
            # 定期報告
            if t % 10000 == 0:
                print(f"步數 {t}: Episode {episode_count}, 獎勵權重 {current_weights}")
        
        logger.close()
        print(f"🏁 混合訓練完成！")
    
    def action_function(self, policy):
        """動作轉換函數（沿用您的實現）"""
        expected_bounds = [-1, 1]
        action_percent = (policy - expected_bounds[0]) / (
            expected_bounds[1] - expected_bounds[0]
        )
        bounded_percent = np.minimum(np.maximum(action_percent, 0), 1)
        return (
            self.env.action_space.low
            + (self.env.action_space.high - self.env.action_space.low) * bounded_percent
        )

def main():
    """主函數"""
    expert_data_path = "../data/dataset_kick.npz"
    
    # 檢查數據存在
    if not os.path.exists(expert_data_path):
        print(f"❌ 專家數據不存在: {expert_data_path}")
        print(f"請確保已收集專家數據")
        return
    
    # 創建混合訓練器
    trainer = HybridTrainer(expert_data_path)
    
    # 執行訓練
    trainer.train(
        bc_epochs=100,        # BC預訓練100個epoch
        rl_timesteps=1000000  # PPO微調100萬步
    )

if __name__ == "__main__":
    import os
    main()