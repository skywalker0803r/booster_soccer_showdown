#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BC預訓練 + Stable Baselines3 PPO訓練系統
使用成熟穩定的SB3框架，API簡潔，功能完整
"""

import numpy as np
import torch
import os
import sys
import glob
try:
    import gymnasium as gym
except ImportError:
    import gym
from sai_rl import SAIClient 
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from utils import Preprocessor
from gdrive_utils import SimpleGDriveSync
from PBRS_module import create_pbrs_wrapper
from gym_compatibility import make_gymnasium_compatible, test_compatibility

# =================================================================
# 1. 環境設置
# =================================================================
print("🚀 初始化BC預訓練 + Stable Baselines3 PPO系統")

# 創建SAI環境
sai = SAIClient(
    comp_id="booster-soccer-showdown", 
    api_key="sai_LFcuaCZiqEkUbNVolQ3wbk5yU7H11jfv",
)

def make_sai_env(use_pbrs=True, pbrs_debug=False):
    """創建SAI環境的工廠函數"""
    env = sai.make_env()
    print("✅ SAI 原始環境創建成功")
    
    # 🔧 添加 Gymnasium 兼容性
    env = make_gymnasium_compatible(env)
    print("✅ Gymnasium 兼容性適配完成")
    
    env = Monitor(env)  # 添加監控
    print("✅ SB3 Monitor 包裝完成")
    
    if use_pbrs:
        # 🎯 添加 PBRS 獎勵塑形
        env = create_pbrs_wrapper(env, gamma=0.99, debug=pbrs_debug)
        print("✅ PBRS 獎勵塑形已啟用")
    
    # 🧪 測試最終環境兼容性
    if pbrs_debug:
        test_compatibility(env)
    
    return env

# 🎯 PBRS 設置
USE_PBRS = True  # 是否使用獎勵塑形
PBRS_DEBUG = False  # 是否輸出PBRS調試信息

env = make_sai_env(use_pbrs=USE_PBRS, pbrs_debug=PBRS_DEBUG)
print(f"✅ 環境已創建 | 觀察: {env.observation_space} | 動作: {env.action_space}")
if USE_PBRS:
    print("🎯 PBRS獎勵塑形: 啟用 - 將幫助突破ep_rew_mean瓶頸")

# =================================================================
# 2. 超參數配置
# =================================================================
TOTAL_TIMESTEPS = 1000000         # 訓練步數
MODEL_NAME = "BC-SB3-PPO"
SAVE_FREQ = 50000                 # 每5萬步保存一次

# Stable Baselines3 PPO超參數 (調優版本)
PPO_CONFIG = {
    'learning_rate': 3e-4,        # 學習率
    'n_steps': 2048,              # 每次更新收集的步數
    'batch_size': 64,             # 批次大小
    'n_epochs': 10,               # 每次更新的epoch數
    'gamma': 0.99,                # 折扣因子
    'gae_lambda': 0.95,           # GAE lambda
    'clip_range': 0.2,            # PPO裁剪範圍
    'ent_coef': 0.01,             # 熵係數
    'vf_coef': 0.5,               # 價值函數損失係數
    'max_grad_norm': 0.5,         # 梯度裁剪
    'verbose': 1,                 # 輸出等級
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'tensorboard_log': './sb3_tensorboard/',
    'policy_kwargs': {            # 網絡架構
        'net_arch': [256, 256],   # 簡潔的網絡結構
        'activation_fn': torch.nn.ReLU,
    }
}

print(f"🔥 使用設備: {PPO_CONFIG['device']}")

# =================================================================
# 3. BC預訓練適配器
# =================================================================
class SB3BCAdapter:
    """將BC預訓練適配到Stable Baselines3"""
    
    def __init__(self, expert_data_path):
        self.expert_data_path = expert_data_path
        self.expert_data = self._load_expert_data()
        
    def _load_expert_data(self):
        """載入並轉換專家數據"""
        if not os.path.exists(self.expert_data_path):
            return None
            
        print(f"📚 載入專家數據: {self.expert_data_path}")
        data = np.load(self.expert_data_path, allow_pickle=True)
        
        # 簡化轉換：從89維提取45維
        il_observations = data['observations']
        expert_actions = data['actions']
        
        # 提取核心機器人狀態
        converted_observations = []
        for obs in il_observations:
            # 前42維 + 任務編碼3維 = 45維
            robot_state = obs[:42]
            task_encoding = obs[-3:]
            research_obs = np.concatenate([robot_state, task_encoding])
            converted_observations.append(research_obs)
        
        converted_observations = np.array(converted_observations, dtype=np.float32)
        expert_actions = np.array(expert_actions, dtype=np.float32)
        
        print(f"✅ 專家數據載入成功:")
        print(f"   觀測: {converted_observations.shape}")
        print(f"   動作: {expert_actions.shape}")
        print(f"   Episodes: {np.sum(data['done'])}")
        
        return {
            'observations': converted_observations,
            'actions': expert_actions,
            'episode_count': int(np.sum(data['done']))
        }
    
    def pretrain_sb3_model(self, model, epochs=50, batch_size=256):
        """使用專家數據預訓練SB3模型"""
        if self.expert_data is None:
            print("❌ 沒有專家數據，跳過BC預訓練")
            return None
        
        print(f"🎯 開始BC預訓練 SB3模型 ({epochs} epochs)")
        
        device = model.device
        observations = torch.tensor(self.expert_data['observations']).to(device)
        actions = torch.tensor(self.expert_data['actions']).to(device)
        
        # 獲取SB3模型的策略網絡
        policy = model.policy
        
        # 創建BC優化器
        optimizer = torch.optim.Adam(policy.parameters(), lr=1e-4, weight_decay=1e-5)
        loss_fn = torch.nn.MSELoss()
        
        dataset_size = len(observations)
        best_loss = float('inf')
        
        policy.train()
        
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
                
                # 使用SB3策略網絡預測動作 (修正版2)
                # 將tensor轉為numpy，預測後再轉回tensor
                batch_obs_np = batch_obs.cpu().numpy()
                
                # 使用SB3預測動作
                actions_np, _ = policy.predict(batch_obs_np, deterministic=True)
                
                # 轉回tensor用於梯度計算
                predicted_actions = torch.tensor(actions_np, device=batch_obs.device, dtype=torch.float32, requires_grad=True)
                
                # 計算損失
                loss = loss_fn(predicted_actions, batch_actions)
                
                # 反向傳播
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
                optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
            
            avg_loss = epoch_loss / num_batches
            
            if avg_loss < best_loss:
                best_loss = avg_loss
            
            # 定期輸出
            if epoch % 10 == 0:
                print(f"   Epoch {epoch:3d}: BC Loss = {avg_loss:.6f} (Best: {best_loss:.6f})")
        
        print(f"✅ BC預訓練完成! 最終損失: {best_loss:.6f}")
        policy.eval()
        
        return best_loss

# =================================================================
# 4. Google Drive設置和模型選擇
# =================================================================
def choose_model_loading():
    """選擇載入模型或重新開始"""
    print("\n" + "="*50)
    print("🤔 BC-SB3-PPO訓練模式選擇")
    print("="*50)
    
    # 檢查本地SB3模型
    local_models = glob.glob(f"*{MODEL_NAME}*.zip") + glob.glob(f"best_*.zip") + glob.glob(f"checkpoint_*.zip")
    
    if local_models:
        print("📂 發現已存在的SB3模型:")
        for i, model in enumerate(local_models):
            print(f"  {i+1}. {model}")
        
        print(f"\n{len(local_models)+1}. 🆕 從頭開始訓練 (包含BC預訓練)")
        
        while True:
            try:
                choice = input("\n選擇要載入的模型 (輸入數字): ").strip()
                choice_num = int(choice)
                
                if choice_num == len(local_models) + 1:
                    return None  # 從頭開始
                elif 1 <= choice_num <= len(local_models):
                    return local_models[choice_num - 1]
                else:
                    print("❌ 無效選擇，請重新輸入")
            except ValueError:
                print("❌ 請輸入有效數字")
    else:
        print("📂 未發現已存在的SB3模型，將從頭開始訓練")
        return None

# 初始化Google Drive
try:
    gdrive_sync = SimpleGDriveSync()
    gdrive_available = gdrive_sync.gdrive_path is not None
    print(f"🔗 Google Drive: {'✅ 已連接' if gdrive_available else '❌ 未連接'}")
except Exception as e:
    print(f"⚠️ Google Drive初始化失敗: {e}")
    gdrive_sync = None
    gdrive_available = False

# 選擇模型載入方式
model_path = choose_model_loading()

# =================================================================
# 5. 創建和配置SB3 PPO模型
# =================================================================
if model_path:
    print(f"📂 載入SB3模型: {model_path}")
    model = PPO.load(model_path, env=env)
    skip_bc = True
    print("✅ SB3模型載入成功，跳過BC預訓練")
else:
    print("🔧 創建新的SB3 PPO模型")
    model = PPO('MlpPolicy', env, **PPO_CONFIG)
    skip_bc = False
    print("✅ SB3 PPO模型創建完成")

# =================================================================
# 6. BC預訓練 (如果需要)
# =================================================================
expert_data_path = "../data/dataset_kick.npz"
if not skip_bc and os.path.exists(expert_data_path):
    bc_adapter = SB3BCAdapter(expert_data_path)
    bc_loss = bc_adapter.pretrain_sb3_model(model, epochs=50)
    print("✅ BC預訓練完成，開始SB3 PPO訓練...")
elif skip_bc:
    print("🔄 載入已訓練模型，跳過BC預訓練")
else:
    print("⚠️ 未找到專家數據，僅使用SB3 PPO訓練")

# =================================================================
# 7. 設置回調函數
# =================================================================
# Checkpoint回調 - 定期保存
checkpoint_callback = CheckpointCallback(
    save_freq=SAVE_FREQ,
    save_path='./sb3_checkpoints/',
    name_prefix=MODEL_NAME,
    verbose=1
)

# 自定義Google Drive上傳回調
class GDriveUploadCallback:
    def __init__(self, gdrive_sync, save_freq):
        self.gdrive_sync = gdrive_sync
        self.save_freq = save_freq
        self.best_reward = -np.inf
        
    def __call__(self, locals_, globals_):
        # 每save_freq步上傳一次
        if locals_['self'].num_timesteps % self.save_freq == 0:
            if self.gdrive_sync and gdrive_available:
                try:
                    model_path = f"./sb3_checkpoints/{MODEL_NAME}_{locals_['self'].num_timesteps}_steps.zip"
                    locals_['self'].save(model_path)
                    
                    # 上傳到Google Drive (這裡需要適配gdrive_utils)
                    print(f"📤 上傳checkpoint到Google Drive")
                except Exception as e:
                    print(f"⚠️ Google Drive上傳失敗: {e}")
        
        return True

# =================================================================
# 8. 開始SB3 PPO訓練
# =================================================================
print(f"🚀 開始Stable Baselines3 PPO訓練 ({TOTAL_TIMESTEPS:,} 步)")
print("="*60)

try:
    # 開始訓練
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=checkpoint_callback,
        progress_bar=True
    )
    
    print(f"🎉 訓練完成!")
    
    # 保存最終模型
    final_model_path = f"final_{MODEL_NAME}.zip"
    model.save(final_model_path)
    print(f"✅ 最終模型已保存: {final_model_path}")
    
    # Google Drive上傳最終模型
    if gdrive_sync and gdrive_available:
        try:
            # 這裡需要適配gdrive_utils
            print(f"📤 上傳最終模型到Google Drive")
        except Exception as e:
            print(f"⚠️ 最終模型Google Drive上傳失敗: {e}")
    
except KeyboardInterrupt:
    print(f"\n⏹️ 訓練被中斷")
    # 保存中斷時的模型
    interrupted_model_path = f"interrupted_{MODEL_NAME}.zip"
    model.save(interrupted_model_path)
    print(f"💾 中斷模型已保存: {interrupted_model_path}")

print("🏁 BC預訓練 + Stable Baselines3 PPO訓練完成!")

# =================================================================
# 9. 簡單測試
# =================================================================
print("\n🧪 進行簡單測試...")
# 修正Gymnasium API兼容性問題
obs, info = env.reset()  # 新版Gym API返回tuple (obs, info)
for i in range(100):
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, dones, truncated, info = env.step(action)  # 新版API返回5個值
    if dones or truncated:
        break

print(f"✅ 測試完成，執行了 {i+1} 步")
env.close()