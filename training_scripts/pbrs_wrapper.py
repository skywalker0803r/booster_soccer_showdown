# pbrs_wrapper.py

import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box
from typing import Dict, Any, Union, Tuple
from stable_baselines3.common.env_util import make_vec_env
from sai_rl import SAIClient # 導入 SAIClient 用於 make_pbrs_env

# --- 1. PBRS Preprocessor ---
class PBRSPreprocessor:
    """
    用於計算 PBRS 所需特徵的預處理器。
    它假設 SAI 環境 info 字典中包含 'ball_xpos_rel_robot' 和 'goal_team_0_rel_ball'。
    """
    def __init__(self, k1: float, k2: float):
        self.k1 = k1
        self.k2 = k2

    def get_features(self, info: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
        """從 info 中提取球和目標的位置，用於計算 potential function。"""
        # 球相對於機器人的位置 (只取 x, y，假設 z 不重要)
        agent_to_ball_pos = info.get('ball_xpos_rel_robot', np.zeros(3))[:2]
        
        # 目標相對於球的位置 (只取 x, y)
        ball_to_goal_pos = info.get('goal_team_0_rel_ball', np.zeros(3))[:2]

        return agent_to_ball_pos, ball_to_goal_pos

    def compute_potential(self, info: Dict[str, Any]) -> float:
        """根據 PBRS 公式計算潛在函式 V(s) 的值。"""
        agent_to_ball, ball_to_goal = self.get_features(info)
        
        # 距離計算 (L2 範數)
        dist_agent_ball = np.linalg.norm(agent_to_ball)
        dist_ball_goal = np.linalg.norm(ball_to_goal)
        
        # Potential Function V(s) = - (k1 * dist_agent_ball + k2 * dist_ball_goal)
        potential = - (self.k1 * dist_agent_ball + self.k2 * dist_ball_goal)
        return potential


# --- 2. PBRS 環境包裝器 (Wrapper) ---
class PBRSWrapper(gym.Wrapper):
    """
    實作 Potential-Based Reward Shaping (PBRS) 的環境包裝器。
    新的獎勵 R' = R + gamma * V(s') - V(s)。
    """
    def __init__(self, env: gym.Env, k1: float, k2: float, gamma: float):
        super().__init__(env)
        self.gamma = gamma
        self.preprocessor = PBRSPreprocessor(k1, k2)
        self.prev_potential = 0.0

    def reset(self, **kwargs) -> Tuple[Union[np.ndarray, Dict[str, np.ndarray]], Dict[str, Any]]:
        """重置環境，並初始化潛在函式的值。"""
        obs, info = self.env.reset(**kwargs)
        # 計算初始狀態的潛在函式值 V(s_0)
        self.prev_potential = self.preprocessor.compute_potential(info)
        return obs, info

    def step(self, action: Union[int, np.ndarray]) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """執行一步，並根據 PBRS 公式計算新的獎勵。"""
        obs, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated

        # 計算新狀態 s' 的潛在函式值 V(s')
        new_potential = self.preprocessor.compute_potential(info)
        
        # 應用 Reward Shaping
        if not done:
            # R' = R + gamma * V(s') - V(s)
            shaped_reward = reward + self.gamma * new_potential - self.prev_potential
        else:
            # 終止狀態下，V(s') 通常為 0，則 R' = R - V(s)
            shaped_reward = reward - self.prev_potential
            
        # 更新 V(s) 準備下一個時間步
        self.prev_potential = new_potential
        
        return obs, shaped_reward, terminated, truncated, info

# --- 3. 輔助函數 (供 ppo_with_pbrs.py 調用) ---
def make_pbrs_env(
    sai: SAIClient,
    comp_id: str,
    stage: str,
    num_envs: int,
    config: Dict[str, Any],
) -> gym.Env:
    """
    建立向量化環境，並應用 PBRSWrapper。
    """
    # 根據 stage 確定環境 ID
    if stage == 'move':
        env_id = "LowerT1GoaliePenaltyKick-v0"
    elif stage == 'kick':
        env_id = "LowerT1KickToTarget-v0" # 假設 kick 訓練用的是目標踢球環境
    else:
        raise ValueError(f"不支援的階段: {stage}")

    def env_factory():
        # 建立單一環境
        env = sai.make_env(env_id=env_id)
        
        # 應用 PBRSWrapper
        env = PBRSWrapper(
            env, 
            k1=config['k1'], 
            k2=config['k2'], 
            gamma=config['gamma']
        )
        return env

    # 建立向量化環境
    vec_env = make_vec_env(env_factory, n_envs=num_envs)
    return vec_env

if __name__ == '__main__':
    # 簡單的測試 PBRSWrapper
    print("pbrs_wrapper.py 檔案已準備好被匯入。")