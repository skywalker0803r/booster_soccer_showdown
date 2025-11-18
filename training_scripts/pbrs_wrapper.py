# pbrs_wrapper.py

import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box
from typing import Dict, Any, Union, Tuple, Callable
from sai_rl import SAIClient 
from stable_baselines3.common.vec_env import DummyVecEnv

# --- 全域常數 ---
_FLOAT_EPS = np.finfo(np.float32).eps # 使用 float32 精度


# --- 1. PBRS Preprocessor ---
class PBRSPreprocessor:
    """
    用於計算 PBRS 所需特徵的預處理器。
    增加角度項的計算以提高踢球引導。
    """
    # 💡 接受 k3 參數
    def __init__(self, stage: str, num_envs: int, k1: float, k2: float, k3: float = 0.0):
        self.stage = stage
        self.num_envs = num_envs
        self.k1 = k1
        self.k2 = k2
        self.k3 = k3

    def get_features(self, info: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
        """從 info 中提取球和目標的位置，用於計算 potential function。"""
        
        # 處理 info 中可能缺失的 key，並確保形狀正確 (num_envs, dim)
        default_pos = np.zeros((self.num_envs, 3), dtype=np.float32)
        
        # 球相對於機器人的位置 (只取 x, y)
        try:
            agent_to_ball_pos = info.get('ball_xpos_rel_robot', default_pos)[:, :2]
        except:
            agent_to_ball_pos = info.get('ball_xpos_rel_robot', default_pos)[:2]
        
        # 目標相對於球的位置 (只取 x, y)
        try:
            ball_to_goal_pos = info.get('goal_team_0_rel_ball', default_pos)[:, :2]
        except:
            ball_to_goal_pos = info.get('goal_team_0_rel_ball', default_pos)[:2]

        return agent_to_ball_pos, ball_to_goal_pos
    
    def compute_potential(self, info: Dict[str, Any]) -> np.ndarray:
        """
        計算潛在函數 V(s)。
        V(s) = - (k1 * dist_agent_ball + k2 * dist_ball_goal) [Move Phase]
        V(s) = - (k1 * dist_agent_ball + k2 * dist_ball_goal) + (k3 * cos_angle) [Kick Phase]
        """
        vec_agent_to_ball, vec_ball_to_goal = self.get_features(info)
        
        # 距離項 (L2 Norm)
        dist_agent_ball = np.linalg.norm(vec_agent_to_ball)
        dist_ball_goal = np.linalg.norm(vec_ball_to_goal)

        # 💡 角度項 (用於 kick 階段)
        potential_value = - (self.k1 * dist_agent_ball) - (self.k2 * dist_ball_goal)
        
        if self.stage == 'kick' and self.k3 > _FLOAT_EPS:
            # 確保向量長度不為零
            norm_agent_to_ball = dist_agent_ball[:, None] + _FLOAT_EPS
            norm_ball_to_goal = dist_ball_goal[:, None] + _FLOAT_EPS
            
            # 單位向量
            unit_agent_to_ball = vec_agent_to_ball / norm_agent_to_ball
            unit_ball_to_goal = vec_ball_to_goal / norm_ball_to_goal

            # 內積 (cos 夾角) - 機器人到球的方向與球到目標的方向夾角
            # 鼓勵機器人站在球的後面
            cos_angle = np.sum(unit_agent_to_ball * unit_ball_to_goal, axis=1)
            
            # 將 cos_angle 項加到潛力函數中，最大化 cos_angle（趨近於 1）
            potential_value += (self.k3 * cos_angle)
            
        # 確保 potential 是 (num_envs,) 的形狀
        return potential_value.astype(np.float32)


# --- 2. PBRS 環境包裝 (Wrapper) ---
class PBRSWrapper(gym.Wrapper):
    
    # 💡 接受 k3 參數
    def __init__(self, env: gym.Env, stage: str, num_envs: int, gamma: float = 0.99, k1: float = 10.0, k2: float = 5.0, k3: float = 0.0):
        super().__init__(env)
        self.gamma = gamma
        self.num_envs = num_envs
        # 💡 初始化 Preprocessor 時傳遞所有參數
        self.preprocessor = PBRSPreprocessor(stage, num_envs, k1, k2, k3)
        self.prev_potential = np.zeros(num_envs, dtype=np.float32) # 初始化為零向量
        
    def reset(self, **kwargs) -> Tuple[np.ndarray, Dict[str, Any]]:
        obs, info = self.env.reset(**kwargs)
        # info 已經是向量化環境的格式
        self.prev_potential = self.preprocessor.compute_potential(info)
        return obs, info

    def step(self, action: Union[int, np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        done = terminated | truncated # 向量化的終止條件
        
        # 計算 V(s')
        new_potential = self.preprocessor.compute_potential(info)
        
        # 應用 Reward Shaping
        shaped_reward = reward.copy()
        
        # R' = R + gamma * V(s') - V(s)
        # 對於未結束的環境: V(s') 會被計算
        shaped_reward[~done] += self.gamma * new_potential[~done] - self.prev_potential[~done]
        # 對於已結束的環境: V(s') = 0，因此 R' = R - V(s)
        shaped_reward[done] += - self.prev_potential[done]
            
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
    if stage == 'move':
        env_id = "LowerT1GoaliePenaltyKick-v0"
    elif stage == 'kick':
        env_id = "LowerT1KickToTarget-v0" 
    else:
        raise ValueError(f"不支援的階段: {stage}")
        
    # 定義一個建立環境的函數
    def env_fn():
        env = sai.make_env(env_id, comp_id=comp_id)
        # 傳遞所有 config 參數
        return PBRSWrapper(
            env, 
            stage=stage, 
            num_envs=1, # 每個獨立環境的 num_envs 都是 1
            gamma=config['gamma'], 
            k1=config['k1'], 
            k2=config['k2'], 
            k3=config['k3'] # 💡 傳遞 k3
        )

    # 使用 DummyVecEnv 封裝多個環境實例
    env = DummyVecEnv([env_fn] * num_envs)
    return env