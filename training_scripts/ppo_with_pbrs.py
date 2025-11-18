# ppo_with_pbrs.py

import argparse
import os
import sys
from datetime import datetime
import numpy as np
import gymnasium as gym
from gymnasium.spaces import Box, Discrete
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback # 即使沒用，也保留
from collections import deque # 即使沒用，也保留
from stable_baselines3.common.logger import configure
import torch
from typing import Dict, Any, Union, Tuple

from sai_rl import SAIClient

# --- 外部模組匯入 ---
# 確保這些檔案存在於相同目錄下
try:
    from log_callback import DetailedLogCallback
    from hrl_wrapper import HierarchicalWrapper
    from pbrs_wrapper import PBRSWrapper, make_pbrs_env # 導入 PBRS 相關功能
except ImportError as e:
    print(f"❌ 錯誤: 無法匯入所需模組。請確保 'log_callback.py', 'hrl_wrapper.py', 'pbrs_wrapper.py' 存在。錯誤: {e}")
    sys.exit(1)


# --- 全域常數 ---
_FLOAT_EPS = np.finfo(np.float64).eps
MODEL_DIR = "low_level_models" # LL Policy 的儲存目錄
HRL_MODEL_DIR = "hrl_models"   # HL Policy 的儲存目錄
MOVE_POLICY_PATH = os.path.join(MODEL_DIR, "move_policy_final.zip")
KICK_POLICY_PATH = os.path.join(MODEL_DIR, "kick_policy_final.zip")
HL_POLICY_PREFIX = "hl_policy"


# --- Preprocessor ---
class Preprocessor():
    """
    用於在訓練前修改環境狀態 (Observation) 的預處理器。
    這是為了處理 SAI 環境返回的狀態並讓模型更容易學習。
    """
    def get_task_onehot(self, info):
        if 'task_index' in info:
            # 確保返回的是 numpy array
            return np.array([info['task_index']])
        else:
            return np.array([])

    def quat_rotate_inverse(self, q: np.ndarray, v: np.ndarray):
        # 這是您的四元數旋轉邏輯，用於將狀態轉換到機器人座標系下
        q_w = q[:,[-1]]
        q_vec = q[:,:3]
        a = v * (2.0 * q_w**2 - 1.0)
        b = np.cross(q_vec, v) * (q_w * 2.0)
        # 確保點積結果可以正確廣播
        dot_product = np.sum(q_vec * v, axis=1, keepdims=True)
        c = q_vec * (dot_product * 2.0)
        return a - b + c 

    def modify_state(self, obs, info):
        """
        修改環境觀察狀態 (Observation)。
        例如: 將相對座標從世界座標轉換到機器人座標系，並追加任務 one-hot 編碼。
        """
        if len(obs.shape) == 1:
            obs = np.expand_dims(obs, axis=0)

        # 這裡應該包含您的實際狀態修改邏輯...
        # 由於原始邏輯不完整，這裡只保留 Task One-Hot 處理
        task_onehot = self.get_task_onehot(info)
        if len(task_onehot.shape) == 1:
            task_onehot = np.expand_dims(task_onehot, axis=0)

        if task_onehot.size > 0:
            # 將任務 one-hot 追加到狀態上
            return np.hstack((obs, task_onehot))
        else:
            return obs


# --- 環境建立函數 ---
def make_env(sai: SAIClient, stage: str, num_envs: int, config: Dict[str, Any]) -> gym.Env:
    """
    創建環境，並根據 stage 應用適當的 Wrapper (PBRS 或 HRL)。
    """
    if stage == 'hrl':
        print("--- 建立 HRL 環境 (HierarchicalWrapper) ---")
        
        # HRL 需要一個底層環境來進行包裝
        # 假設 LL 訓練是使用 'LowerT1GoaliePenaltyKick-v0' 或類似的基礎環境
        ll_env_id = "LowerT1GoaliePenaltyKick-v0" 
        
        def env_factory():
            # 建立單一底層環境
            env = sai.make_env(env_id=ll_env_id) # 使用 SAIClient 建立底層環境
            
            # 將底層環境包裝進 HierarchicalWrapper
            hrl_env = HierarchicalWrapper(
                env=env, # 傳入實際的環境實例
                ll_steps=config['ll_steps']
            )
            return hrl_env

        # 建立向量化環境
        vec_env = make_vec_env(env_factory, n_envs=num_envs)
        return vec_env
    
    # move 或 kick 階段使用 PBRSWrapper (這需要 pbrs_wrapper.py)
    print(f"--- 建立 {stage.upper()} 環境 (PBRSWrapper) ---")
    return make_pbrs_env(
        sai=sai,
        comp_id="booster-soccer-showdown", # 應從 config 或 args 取得
        stage=stage,
        num_envs=num_envs,
        config=config
    )


# --- 主函數 ---
def main():
    parser = argparse.ArgumentParser(description='PPO with PBRS/HRL Training Script')
    parser.add_argument('--stage', type=str, required=True, choices=['move', 'kick', 'hrl'], help='Training stage: move, kick, or hrl')
    parser.add_argument('--config_path', type=str, default=None, help='Path to an optional JSON config file to override default settings')
    parser.add_argument('--mode', type=str, default='new', choices=['new', 'continue'], help='Training mode: new or continue')
    args = parser.parse_args()
    stage = args.stage
    mode = args.mode

    # --- 預設訓練配置 (可以從 config_path 加載 JSON 覆蓋) ---
    default_config = {
        'lr': 3e-4, 
        'n_steps': 2048, 
        'batch_size': 64, 
        'gamma': 0.99, 
        'n_epochs': 10,
        'ent_coef': 0.01,
        'clip_range': 0.2,
        'total_timesteps': 5000000,
        'log_interval': 100000, 
        'net_arch': [256,256,128,128,64],
        # PBRS 參數 (只在 'move' 和 'kick' 階段使用)
        'k1': 0.5, 
        'k2': 1.0, 
        # HRL 參數 (只在 'hrl' 階段使用)
        'll_steps': 10 
    }
    config = default_config
    # 這裡可以加入讀取 JSON 覆蓋配置的邏輯

    print("--- 初始化 SAI Client (請將 api_key 替換為您的金鑰) ---")
    # 注意: comp_id 和 api_key 應從安全地方加載
    sai = SAIClient(comp_id="booster-soccer-showdown", api_key="sai_LFcuaCZiqEkUbNVolQ3wbk5yU7H11jfv") 

    # --- 路徑設定 ---
    if stage == 'hrl':
        save_path = HRL_MODEL_DIR
        save_prefix = HL_POLICY_PREFIX
    else:
        save_path = MODEL_DIR
        save_prefix = f"{stage}_policy"

    # 確保儲存目錄存在
    os.makedirs(save_path, exist_ok=True)
    
    # 日誌目錄
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_name = f"{stage}_{timestamp}_{save_prefix}"
    log_dir = os.path.join("runs", log_name)
    os.makedirs(log_dir, exist_ok=True)
    
    # 建立環境
    num_envs = 1 # 您可能需要根據需要調整
    env = make_env(sai, stage, num_envs, config)
    
    # 建立日誌記錄器
    new_logger = configure(log_dir, ["csv", "tensorboard"])

    # --- 模型初始化或繼續訓練 ---
    policy_kwargs = dict(net_arch=config['net_arch'])
    
    if mode == 'new':
        print(f"--- 開始新的訓練: {stage.upper()} ---")
        if stage == 'hrl':
            # HRL 使用離散動作空間
            policy = "MlpPolicy" # 假設您的 HL Policy 是 MlpPolicy
            action_space = Discrete(2) # 假設有 2 個技能 (Move, Kick)
        else:
            # Move/Kick 使用連續動作空間
            policy = "MlpPolicy" 
            action_space = env.action_space # 連續動作空間
        
        # 由於環境中應用了 Preprocessor/Wrapper，PPO 需接收正確的空間
        model = PPO(
            policy, 
            env, 
            learning_rate=config['lr'], 
            n_steps=config['n_steps'], 
            batch_size=config['batch_size'], 
            gamma=config['gamma'], 
            n_epochs=config['n_epochs'],
            ent_coef=config['ent_coef'], 
            clip_range=config['clip_range'],
            verbose=2, 
            tensorboard_log=log_dir,
            policy_kwargs=policy_kwargs,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
    else:
        model_to_load = os.path.join(save_path, f"{save_prefix}_final.zip")
        print(f"--- 繼續訓練，載入模型: {model_to_load} ---")
        # 載入時需要 custom_objects 來正確反序列化 Wrapper，但在 PPO 這裡先設為 {}
        model = PPO.load(model_to_load, env=env, custom_objects={}, device="cuda" if torch.cuda.is_available() else "cpu")
        # 重新設定日誌記錄器以繼續記錄
        model.set_logger(new_logger)


    # --- 訓練資訊輸出 ---
    print("\n-----------------------------")
    print(f"Training Stage: {stage.upper()}")
    print(f"Training Mode: {mode.upper()}")
    print(f"Policy: {model.policy_class}")
    print(f"Timesteps per update: {config['n_steps']}")
    print(f"Total Timesteps: {config['total_timesteps']}")
    print(f"Learning Rate: {config['lr']}")
    print(f"Gamma: {config['gamma']}")
    print(f"Clip Range: {model.clip_range}")
    if stage != 'hrl':
        print("\nPBRS Hyperparameters (在環境初始化時生效):")
        print(f"  k1 (agent-ball): {config['k1']}")
        print(f"  k2 (ball-goal): {config['k2']}")
    else:
        print(f"\nHRL LL Steps: {config['ll_steps']}")

    print(f"\nTensorBoard Log: {log_dir}")
    print(f"Models will be saved in: {save_path}")
    print("-----------------------------\n")

    # --- Model Training ---
    callback = DetailedLogCallback(
        save_path=save_path, 
        save_prefix=save_prefix, 
        log_interval=config['log_interval'],
        verbose=1
    )

    try:
        model.learn(total_timesteps=config['total_timesteps'], callback=callback, reset_num_timesteps=(mode=='new'))
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
    finally:
        final_model_path = os.path.join(save_path, f"{save_prefix}_final.zip")
        model.save(final_model_path)
        print(f"\nFinal model saved to: {final_model_path}")
        env.close()

    print("\nTraining complete.")
    print(f"\n✅ 要檢視日誌，請運行此命令: tensorboard --logdir=runs")


if __name__ == '__main__':
    main()