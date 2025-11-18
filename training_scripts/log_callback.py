# log_callback.py

import os
from collections import deque
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.results_plotter import load_results, ts2xy

class DetailedLogCallback(BaseCallback):
    """
    自定義的回調函數，用於在訓練過程中記錄詳細的指標到 TensorBoard，
    並在達到最佳平均回報時保存模型。
    """
    def __init__(self, save_path: str, save_prefix: str, log_interval: int, verbose: int = 0):
        super().__init__(verbose)
        self.save_path = save_path
        self.save_prefix = save_prefix
        self.log_interval = log_interval
        self.best_mean_reward = -np.inf
        # 用於儲存最近 N 個平均回報，以平滑曲線
        self.mean_reward_buffer = deque(maxlen=10) 
        self.num_timesteps_last_log = 0

    def _init_callback(self) -> None:
        # 在訓練開始時初始化
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        # 紀錄當前訓練步數
        current_timesteps = self.num_timesteps
        
        # 每隔 log_interval 步檢查並記錄
        if (current_timesteps - self.num_timesteps_last_log) >= self.log_interval:
            self.num_timesteps_last_log = current_timesteps
            
            # --- 獲取並記錄訓練指標 (Training Metrics) ---
            # Stable-Baselines3 會自動在訓練期間將指標寫入 logger
            # 我們在這裡手動添加一些自定義的紀錄 (如果需要)
            
            # 例如，您可以紀錄學習率
            # lr = self.model.param_groups[0]['lr'] # PPO 沒有 param_groups
            # self.logger.record('train/learning_rate', lr)
            
            # 獲取並記錄環境回報 (Episode Rewards)
            if self.model.ep_info_buffer:
                # 獲取最近完成的 episode 的回報
                last_ep_rewards = [ep_info['r'] for ep_info in self.model.ep_info_buffer]
                if last_ep_rewards:
                    mean_reward = np.mean(last_ep_rewards)
                    
                    # 使用緩衝區計算平滑平均回報
                    self.mean_reward_buffer.append(mean_reward)
                    smooth_mean_reward = np.mean(self.mean_reward_buffer)
                    
                    # 記錄指標
                    self.logger.record('rollout/ep_reward_mean_raw', mean_reward)
                    self.logger.record('rollout/ep_reward_mean_smooth', smooth_mean_reward)
                    
                    if self.verbose > 0:
                        print(f"Timestep: {current_timesteps} | Mean Reward (Raw): {mean_reward:.4f} | Mean Reward (Smooth): {smooth_mean_reward:.4f}")

                    # --- 模型保存邏輯 ---
                    if smooth_mean_reward > self.best_mean_reward:
                        self.best_mean_reward = smooth_mean_reward
                        
                        # 保存最佳模型
                        path = os.path.join(self.save_path, f"{self.save_prefix}_best.zip")
                        self.model.save(path)
                        if self.verbose > 0:
                            print(f"⭐ Saving new best model with mean reward {self.best_mean_reward:.4f} to {path}")
            
            # 確保指標被寫入 TensorBoard
            self.logger.dump(current_timesteps)

        return True # 返回 True 繼續訓練