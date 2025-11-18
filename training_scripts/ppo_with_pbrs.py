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
from stable_baselines3.common.callbacks import BaseCallback
from collections import deque
from stable_baselines3.common.logger import configure
import torch
from typing import Dict, Any, Union, Tuple

from sai_rl import SAIClient

# --- HRL Wrapper 匯入 ---
# 確保 hrl_wrapper.py 檔案與本文件在同一目錄
try:
    from hrl_wrapper import HierarchicalWrapper
except ImportError:
    print("❌ 錯誤: 無法匯入 hrl_wrapper。請確保 hrl_wrapper.py 存在並在正確的路徑。")
    sys.exit(1)


# --- 全域常數 ---
_FLOAT_EPS = np.finfo(np.float64).eps
MODEL_DIR = "low_level_models" # LL Policy 的儲存目錄
HRL_MODEL_DIR = "hrl_models"   # HL Policy 的儲存目錄
MOVE_POLICY_PATH = os.path.join(MODEL_DIR, "move_policy_final.zip")
KICK_POLICY_PATH = os.path.join(MODEL_DIR, "kick_policy_final.zip")
HL_POLICY_PREFIX = "hrl_high_level_policy"


# --- 1. Preprocessor (您的原始碼) ---
class Preprocessor():
    """用於將任務 One-Hot 向量加入到觀察狀態中。"""
    def get_task_onehot(self, info: Dict[str, Any]) -> np.ndarray:
        if 'task_index' in info:
            return info['task_index']
        else:
            return np.array([])

    def quat_rotate_inverse(self, q: np.ndarray, v: np.ndarray) -> np.ndarray:
        if len(q.shape) == 1: q = np.expand_dims(q, axis=0)
        if len(v.shape) == 1: v = np.expand_dims(v, axis=0)
            
        q_w = q[:,[-1]]
        q_vec = q[:,:3]
        a = v * (2.0 * q_w**2 - 1.0)
        b = np.cross(q_vec, v) * (q_w * 2.0)
        c = q_vec * (np.sum(q_vec * v, axis=1).reshape(-1,1) * 2.0)    
        return a - b + c 

    def modify_state(self, obs: np.ndarray, info: Dict[str, Any]) -> np.ndarray:
        if len(obs.shape) == 1:
            obs = np.expand_dims(obs, axis=0)

        task_onehot = self.get_task_onehot(info)
        if len(task_onehot.shape) == 1:
            task_onehot = np.expand_dims(task_onehot, axis=0)
        
        if task_onehot.size > 0:
            return np.hstack((obs, task_onehot))
        else:
            return obs

# --- 2. 獎勵塑形環境包裝器 (PBRS Wrapper) ---
class PBRSWrapper(gym.Wrapper):
    """
    實作基於勢能的獎勵塑形 (Potential-Based Reward Shaping)。
    只在低階策略訓練時使用。
    """
    def __init__(self, env, k1: float, k2: float):
        super().__init__(env)
        self.k1 = k1  # Agent to Ball 係數
        self.k2 = k2  # Ball to Goal 係數
        self.last_potential = 0.0
        # PPO 預設 gamma=0.99，這裡固定使用 0.99
        self.gamma = 0.99 

    def _get_potential(self, info: Dict[str, Any]) -> float:
        """計算當前的勢能 (Potential)。"""
        if 'ball_xpos_rel_robot' not in info or 'goal_team_1_rel_ball' not in info:
             return 0.0

        d_agent_ball = np.linalg.norm(info['ball_xpos_rel_robot'])
        d_ball_goal = np.linalg.norm(info['goal_team_1_rel_ball'])
        
        # 勢能函數 V(s) = -k1 * d(agent, ball) - k2 * d(ball, goal)
        potential = -self.k1 * d_agent_ball - self.k2 * d_ball_goal
        return float(potential)

    def reset(self, **kwargs) -> Tuple[np.ndarray, Dict[str, Any]]:
        obs, info = self.env.reset(**kwargs)
        self.last_potential = self._get_potential(info)
        return obs, info

    def step(self, action: Union[int, np.ndarray]) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        current_potential = self._get_potential(info)
        
        # 獎勵塑形項 F(s, s') = gamma * V(s') - V(s)
        shaping_reward = self.gamma * current_potential - self.last_potential
        
        self.last_potential = current_potential

        # 加入塑形獎勵
        reward += shaping_reward
        
        return obs, reward, terminated, truncated, info

# --- 3. 訓練日誌回調函數 ---
class DetailedLogCallback(BaseCallback):
    """用於記錄詳細訓練指標的回調函數。"""
    def __init__(self, save_path: str, save_prefix: str, log_interval: int = 100000, verbose: int = 1):
        super().__init__(verbose)
        self.save_path = save_path
        self.save_prefix = save_prefix
        self.log_interval = log_interval
        self.ep_rewards = deque(maxlen=100) # 追蹤最近 100 個回合
        self.ep_lengths = deque(maxlen=100)
        self.current_ep_reward = 0
        self.current_ep_length = 0
        self.last_save_timesteps = 0

    def _on_step(self) -> bool:
        # PPO 的 self.locals['dones'] 和 self.locals['rewards'] 已經是批次 (在這裡是 VecEnv=1)
        reward = self.locals['rewards'][0] 
        done = self.locals['dones'][0]

        self.current_ep_reward += reward
        self.current_ep_length += 1
        
        if done:
            self.ep_rewards.append(self.current_ep_reward)
            self.ep_lengths.append(self.current_ep_length)
            self.current_ep_reward = 0
            self.current_ep_length = 0
        
        # 每隔 log_interval 步數記錄並儲存
        if self.num_timesteps - self.last_save_timesteps >= self.log_interval:
            mean_reward = np.mean(self.ep_rewards) if self.ep_rewards else 0
            mean_length = np.mean(self.ep_lengths) if self.ep_lengths else 0
            
            # 記錄到 TensorBoard
            self.logger.record('rollout/ep_rew_mean', mean_reward)
            self.logger.record('rollout/ep_len_mean', mean_length)
            self.logger.dump(self.num_timesteps)

            # 儲存模型
            save_model_path = os.path.join(self.save_path, f"{self.save_prefix}_{self.num_timesteps}.zip")
            self.model.save(save_model_path)
            
            self.last_save_timesteps = self.num_timesteps

        return True


# --- 4. 訓練函數 ---

def train_model(config: Dict[str, Any], sai_client: SAIClient, stage: str = 'move', mode: str = 'new'):
    
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(HRL_MODEL_DIR, exist_ok=True)

    # 設置儲存路徑和日誌目錄
    if stage == 'hrl':
        save_path = HRL_MODEL_DIR
        save_prefix = HL_POLICY_PREFIX
    else:
        save_path = MODEL_DIR
        save_prefix = f"{stage}_policy"
        
    log_dir = os.path.join("logs", stage, datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(log_dir, exist_ok=True)
    new_logger = configure(log_dir, ["stdout", "tensorboard"])
    
    # --- 環境創建和包裝邏輯 ---
    # 創建底層環境並傳入 Preprocessor
    base_env = sai_client.make_env(preprocessor=Preprocessor()) 

    if stage == 'hrl':
        # HRL 訓練：使用 HierarchicalWrapper
        env = HierarchicalWrapper(base_env, ll_steps=config['ll_steps']) 
        print(f"\n--- 環境: HRL High-Level Training --- (HL Steps={config['ll_steps']})")
        print(f"高階動作空間: {env.action_space}")
        
    elif stage in ['move', 'kick']:
        # LL 訓練：使用 PBRSWrapper
        env = PBRSWrapper(
            base_env, 
            k1=config['k1'], 
            k2=config['k2']
        )
        print(f"\n--- 環境: LL {stage.upper()} Policy Training ---")
        print(f"低階動作空間: {env.action_space}")

    # --- PPO 模型初始化 ---
    
    if mode == 'new':
        print(f"--- 開始新的訓練: {save_prefix} ---")
        model = PPO(
            "MlpPolicy",
            env,
            verbose=0,
            tensorboard_log=log_dir,
            policy_kwargs=dict(net_arch=config['net_arch']),
            learning_rate=config['lr'],
            n_steps=config['n_steps'],
            batch_size=config['batch_size'],
            gamma=config['gamma'],
            n_epochs=config['n_epochs'],
            ent_coef=config['ent_coef'], 
            clip_range=config['clip_range'],
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
    else:
        model_to_load = os.path.join(save_path, f"{save_prefix}_final.zip")
        print(f"--- 繼續訓練，載入模型: {model_to_load} ---")
        model = PPO.load(model_to_load, env=env, custom_objects={})

    model.set_logger(new_logger)
    
    print("\n-----------------------------")
    print(f"Training Stage: {stage.upper()}")
    print(f"Total Timesteps: {config['total_timesteps']}")
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
    print(f"To view logs, run: tensorboard --logdir={os.path.join('logs', stage)}")

# --- 5. 主程式 ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PPO Training Script with PBRS and HRL Support')
    parser.add_argument('--comp_id', type=str, required=True, help='SAI Competition ID')
    parser.add_argument('--stage', type=str, default='move', choices=['move', 'kick', 'hrl'], help='Training stage: move (LL), kick (LL), or hrl (HL)')
    parser.add_argument('--mode', type=str, default='new', choices=['new', 'continue'], help='Training mode: new or continue')
    args = parser.parse_args()

    # --- 預設訓練配置 ---
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

    print("--- 初始化 SAI Client ---")
    sai = SAIClient(comp_id=args.comp_id)

    # 確保儲存目錄存在
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(HRL_MODEL_DIR, exist_ok=True)

    # --- 執行訓練前的檢查 ---
    if args.stage == 'hrl':
        if not os.path.exists(MOVE_POLICY_PATH) or not os.path.exists(KICK_POLICY_PATH):
            print(f"\n🚨🚨 警告: HRL 訓練需要低階模型。")
            print(f"請先訓練並儲存: {MOVE_POLICY_PATH} 和 {KICK_POLICY_PATH}")
            sys.exit(1)
        
        train_model(default_config, sai, stage='hrl', mode=args.mode)
    
    elif args.stage == 'move':
        train_model(default_config, sai, stage='move', mode=args.mode)

    elif args.stage == 'kick':
        train_model(default_config, sai, stage='kick', mode=args.mode)

    print("\n所有操作完成。")