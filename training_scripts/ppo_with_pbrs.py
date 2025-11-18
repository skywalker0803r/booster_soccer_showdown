# ppo_with_pbrs.py
import shutil # 用於檔案複製
import argparse
import os
import sys
from datetime import datetime
import numpy as np
import gymnasium as gym
from gymnasium.spaces import Box, Discrete
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.logger import configure
import torch
from typing import Dict, Any, Union, Tuple
# 💡 新增 VecNormalize 導入
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv 

from sai_rl import SAIClient

# --- 外部模組匯入 ---
try:
    from log_callback import DetailedLogCallback
    from hrl_wrapper import HierarchicalWrapper
    # 確保 make_pbrs_env 存在於 pbrs_wrapper.py
    from pbrs_wrapper import make_pbrs_env 
except ImportError as e:
    print(f"❌ 錯誤: 無法匯入所需模組。請確保 'log_callback.py', 'hrl_wrapper.py', 'pbrs_wrapper.py' 存在。錯誤: {e}")
    sys.exit(1)


# --- 全域常數 ---
_FLOAT_EPS = np.finfo(np.float64).eps
MODEL_DIR = "low_level_models" # LL Policy 的儲存目錄
HRL_MODEL_DIR = "hrl_models"   # HL Policy 的儲存目錄


# --- 預設超參數配置 (調整以提高穩定性和性能) ---
default_config: Dict[str, Any] = {
    # PPO Core
    'policy': 'MlpPolicy',
    'n_steps': 2048,           # Rollout buffer size
    'batch_size': 256,         # Minibatch size for gradient updates
    'gamma': 0.99,             # Discount factor
    'learning_rate': 3e-4,     # Initial learning rate
    'n_epochs': 10,            # Number of epochs for PPO
    'gae_lambda': 0.95,        # GAE 參數
    'clip_range': 0.2,         # Clipping parameter

    # 💡 調整網絡結構 (增加容量)
    'policy_kwargs': dict(net_arch=dict(pi=[512, 512, 256], vf=[512, 512, 256])), 

    # 💡 調整熵係數 (降低隨機性，鼓勵收斂)
    'ent_coef': 0.005,         

    # Training and Logging
    'total_timesteps': 10_000_000,
    'log_interval': 10000, 

    # PBRS Parameters (Low-Level Only)
    'k1': 10.0,  # 接近球的潛力係數 (agent-ball)
    'k2': 5.0,   # 踢向目標的潛力係數 (ball-goal)
    'k3': 2.0,   # 💡 新增角度引導係數 (在 kick 階段生效，鼓勵機器人站在好的位置踢球)
    
    # HRL Parameters (High-Level Only)
    'll_steps': 10, # 每個高階時間步執行的低階步數
}


# --- 環境建立函數 (增加 VecNormalize) ---
def make_env(
    sai: SAIClient,
    comp_id: str,
    stage: str,
    num_envs: int,
    config: Dict[str, Any],
) -> gym.Env:
    """
    建立向量化環境，並根據 stage 應用 Wrapper，最後應用 VecNormalize。
    """
    if stage in ('move', 'kick'):
        env = make_pbrs_env(
            sai=sai, 
            comp_id=comp_id, 
            stage=stage, 
            num_envs=num_envs, 
            config=config # config 包含 k1, k2, k3
        )
    elif stage == 'hrl':
        # 建立基礎環境並包裹 HierarchicalWrapper
        def env_fn():
            # 假設 HRL 使用的基礎環境與 kick 訓練相似
            base_env = sai.make_env()
            # HierarchicalWrapper 會在內部處理 LL Policy 的載入
            return HierarchicalWrapper(base_env, ll_steps=config['ll_steps'])

        env = DummyVecEnv([env_fn] * num_envs)

    else:
        raise ValueError(f"不支援的階段: {stage}")

    # 💡 應用觀察空間和獎勵正規化 (大幅提高穩定性)
    # 我們在這裡使用 True/True 進行正規化
    env = VecNormalize(
        env, 
        norm_obs=True, 
        norm_reward=True, 
        clip_obs=10.,
        gamma=config['gamma']
    )

    return env


# --- 主邏輯 (Main Logic) ---
def main(stage: str, mode: str, num_envs: int = 1):
    # 初始化 SAI Client
    sai = SAIClient(
        comp_id="booster-soccer-showdown", # ⚠️ 請替換為您的比賽 ID
        api_key="sai_LFcuaCZiqEkUbNVolQ3wbk5yU7H11jfv",        # ⚠️ 請替換為您的 API Key
    )

    config = default_config

    # --- 訓練環境與模型準備 ---
    print("🛠️ 正在初始化環境和模型...")
    env = make_env(sai, comp_id="booster-soccer-showdown", stage=stage, num_envs=num_envs, config=config)

    # 設置日誌和模型儲存路徑
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_prefix = f"ppo_{stage}_{current_time}"
    
    if stage == 'hrl':
        base_dir = HRL_MODEL_DIR
    else:
        base_dir = MODEL_DIR
        
    save_path = os.path.join(base_dir, save_prefix)
    log_dir = os.path.join("runs", save_prefix)
    
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # 設定 logger 以確保日誌輸出到指定目錄
    new_logger = configure(log_dir, ["stdout", "csv", "tensorboard"])

    # 初始化或載入模型
    if mode == 'new':
        print(f"🔄 創建新的 {stage.upper()} PPO 模型...")
        model = PPO(
            config['policy'],
            env,
            verbose=1,
            tensorboard_log=log_dir,
            learning_rate=config['learning_rate'],
            n_steps=config['n_steps'] // num_envs, 
            batch_size=config['batch_size'],
            gamma=config['gamma'],
            n_epochs=config['n_epochs'],
            clip_range=config['clip_range'],
            gae_lambda=config['gae_lambda'],
            ent_coef=config['ent_coef'],
            policy_kwargs=config['policy_kwargs'],
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        model.set_logger(new_logger)
    elif mode == 'continue':
        # 載入邏輯需要額外的檔案路徑處理，這裡需要使用者處理路徑和 VecNormalize 統計數據
        raise NotImplementedError("Continue mode requires specifying a model path and handling VecNormalize stats loading.")
    else:
        raise ValueError(f"不支援的模式: {mode}")

    # --- 訓練參數摘要 (省略了原文件中的部分輸出，但確保核心參數可見) ---
    print("\n-----------------------------")
    print(f"STAGE: {stage.upper()} | ENVS: {num_envs}")
    print(f"Learning Rate: {config['learning_rate']} | Gamma: {config['gamma']}")
    print(f"Ent Coef: {config['ent_coef']} | Total Timesteps: {config['total_timesteps']}")
    if stage != 'hrl':
        print(f"PBRS: k1={config['k1']}, k2={config['k2']}, k3={config['k3']}")
    else:
        print(f"HRL LL Steps: {config['ll_steps']}")
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
        # 💡 保存最終模型和 VecNormalize 統計數據
        final_model_path = os.path.join(save_path, f"{save_prefix}_final.zip")
        model.save(final_model_path)
        print(f"\n✅ Final model saved to {final_model_path}")

        # --- 新增邏輯: 將模型複製到 HRL 預期的固定路徑 (FIX) ---
        if stage in ['move', 'kick']:
            # HRL 預期的固定路徑：low_level_models/move_policy_final.zip 或 kick_policy_final.zip
            hrl_target_path = os.path.join(MODEL_DIR, f"{stage}_policy_final.zip") 
            try:
                shutil.copyfile(final_model_path, hrl_target_path)
                print(f"✅ Copied {stage} model to HRL fixed path: {hrl_target_path}")
            except Exception as e:
                # 雖然不應發生，但保留錯誤處理
                print(f"❌ Warning: Failed to copy model to HRL fixed path: {e}")
        # -----------------------------------------------------------
        
        # 保存 VecNormalize 統計數據 (對推論很重要)
        stats_path = os.path.join(save_path, f"vec_normalize_{stage}.pkl")
        env.save(stats_path)
        print(f"✅ VecNormalize stats saved to {stats_path}")
        
        env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="PPO Training Script with PBRS/HRL")
    parser.add_argument('--stage', type=str, required=True, choices=['move', 'kick', 'hrl'],
                        help="訓練階段: move (移動), kick (踢球), hrl (分層)")
    parser.add_argument('--mode', type=str, default='new', choices=['new', 'continue'],
                        help="訓練模式: new (新的訓練), continue (繼續訓練)")
    parser.add_argument('--num_envs', type=int, default=1,
                        help="向量化環境數量")
    
    args = parser.parse_args()
    main(args.stage, args.mode, args.num_envs)