# hrl_wrapper.py

import gymnasium as gym
from gymnasium.spaces import Discrete, Box
import numpy as np
import os
from stable_baselines3 import PPO 
from typing import Union, Tuple, Dict, Any, Optional

# --- 全域常數/路徑 (確保與 ppo_with_pbrs.py 中的定義一致) ---
MODEL_DIR = "low_level_models"
MOVE_POLICY_PATH = os.path.join(MODEL_DIR, "move_policy_final.zip")
KICK_POLICY_PATH = os.path.join(MODEL_DIR, "kick_policy_final.zip")

# --- 1. 低階策略/技能控制器 (載入並使用已訓練好的模型) ---
class SkillPolicy:
    """
    用於管理和執行低階 Move (0) 和 Kick (1) 策略的類別。
    """
    def __init__(self):
        self.move_model: PPO = self._load_policy(MOVE_POLICY_PATH, "Move")
        self.kick_model: PPO = self._load_policy(KICK_POLICY_PATH, "Kick")
        print("✅ SkillPolicy 載入成功。")

    def _load_policy(self, path: str, name: str) -> PPO:
        """載入單一 PPO 模型。"""
        try:
            # PPO.load 會自動將模型設定為推論模式
            model = PPO.load(path)
            return model
        except Exception as e:
            print(f"❌ 無法載入 {name} 模型: {path}. 請確保低階模型已訓練並存在。錯誤: {e}")
            raise

    def predict(self, obs: np.ndarray, skill_id: int) -> np.ndarray:
        """根據 skill_id 選擇並執行低階動作。"""
        model = self.move_model if skill_id == 0 else self.kick_model
        # ⚠️ 注意：這裡的 obs 必須是未經過 HRL 擴展的原始觀察，因為低階模型是獨立訓練的。
        action, _ = model.predict(obs, deterministic=True)
        return action
    
    
# --- 2. HRL 環境包裝 (Wrapper) ---
class HierarchicalWrapper(gym.Wrapper):
    
    def __init__(self, env: gym.Env, ll_steps: int):
        super().__init__(env)
        self.ll_steps = ll_steps
        self.skill_policy = SkillPolicy() # 載入低階策略
        self.current_obs: Optional[np.ndarray] = None # 儲存未擴展的原始觀察
        self.current_skill = 0  # 當前技能 ID
        self.last_skill = 0
        
        # 動作空間：離散的技能 ID (0: Move, 1: Kick)
        # 由於這是 HRL 的頂層，這個 action_space 代表高層動作空間
        self.action_space = Discrete(2) 
        
        # 💡 修復: 為了滿足 _augment_obs 中的屬性存取，明確定義它。
        # 雖然 self.action_space 已經是 Discrete(2)，但為了相容錯誤追溯中的命名，我們新增此屬性。
        self.action_space_high_level = self.action_space  # <--- 關鍵修復點

        # 💡 擴展觀察空間：原始觀察 + [當前技能 ID (2維 1-hot), 技能執行進度 (1維 float)]
        original_obs_space = self.env.observation_space.shape[0]
        # 技能 ID (2維 1-hot) + 技能進度 (1維 float) = 3 維
        new_obs_dim = original_obs_space + self.action_space_high_level.n + 1 
        
        # 由於 VecEnv 會將多個環境的輸出堆疊，所以這裡的 shape 只需要 (new_obs_dim,)
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(new_obs_dim,), dtype=np.float32)

    @property
    def num_envs(self) -> int:
        """HRLWrapper 應該被包裹在 DummyVecEnv 中，所以這裡 num_envs 應為 1"""
        return 1 

    def _augment_obs(self, obs: np.ndarray, skill_id: int, progress: float) -> np.ndarray:
        """
        將技能 ID (1-hot) 和進度添加到觀察狀態中。
        :param obs: 原始觀察狀態 (1D)。
        ...
        """
        # 創建 1-hot 技能數組
        # 💡 修復: 現在 self.action_space_high_level 已經存在
        num_high_level_actions = self.action_space_high_level.n
        skill_one_hot = np.zeros(num_high_level_actions, dtype=np.float32)
        skill_one_hot[skill_id] = 1.0
        
        # 將所有數組保持為 1D 進行拼接
        progress_scalar = np.array([progress], dtype=np.float32)
        
        # 在 1D 上拼接 (軸 0)
        # 拼接後 shape: (original_obs_dim + num_skills + 1,)
        return np.concatenate([obs, skill_one_hot, progress_scalar], axis=0).astype(np.float32)
    
    def _check_skill_termination(self, skill_id: int, info: Dict[str, Any]) -> bool:
        """
        實作單一環境的內部技能終止條件。
        返回一個布林值，表示是否達到內部終止條件。
        """
        
        # info 中的 key (如 'ball_xpos_rel_robot') 是 (1, dim) 的 NumPy 數組
        
        # 距離 (L2 norm)
        agent_to_ball_dist = np.linalg.norm(info['ball_xpos_rel_robot'][0, :2])
        
        if skill_id == 0: # Move 技能：到達球附近即成功終止
            MOVE_SUCCESS_THRESHOLD = 0.3 
            return agent_to_ball_dist < MOVE_SUCCESS_THRESHOLD
            
        elif skill_id == 1: # Kick 技能：球被踢出即成功終止 (檢查球的速度)
            
            # 假設 info 中有 'ball_xvel' (球的絕對速度)
            if 'ball_xvel' in info:
                 ball_speed = np.linalg.norm(info['ball_xvel'][0])
            else:
                 # 如果環境沒有提供速度信息，則不觸發內部終止
                 return False
            
            KICK_SUCCESS_SPEED = 1.0 # 例如，球速超過 1.0 m/s
            return ball_speed > KICK_SUCCESS_SPEED
            
        return False

    def reset(self, **kwargs) -> Tuple[np.ndarray, Dict[str, Any]]:
        obs, info = self.env.reset(**kwargs)
        self.current_obs = obs # 儲存未擴展的原始觀察 (用於 LL Policy)
        self.last_skill = 0 
        self.current_skill = 0 
        
        # 返回擴展後的觀察狀態 (初始技能 0, 進度 0.0)
        return self._augment_obs(obs, 0, 0.0), info

    def step(self, action: Union[int, np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
        
        # 確保動作是整數 (Skill ID)
        skill_id = int(action.item()) if isinstance(action, np.ndarray) else int(action)
        
        # 💡 累積獎勵必須是 NumPy 陣列 (1,)，以符合 DummyVecEnv 接口
        accumulated_reward = np.zeros((self.num_envs,), dtype=np.float32) 
        
        # 💡 懲罰技能切換：如果技能發生變化，施加小的負獎勵 (避免 chattering)
        SWITCH_PENALTY = -0.05 
        
        if skill_id != self.current_skill: 
            accumulated_reward += SWITCH_PENALTY
        
        self.last_skill = self.current_skill
        self.current_skill = skill_id

        final_obs = self.current_obs
        final_info = None
        
        # 終止和截斷必須是 NumPy 陣列 (1,)
        terminated = np.zeros((self.num_envs,), dtype=bool)
        truncated = np.zeros((self.num_envs,), dtype=bool)
        
        # --- 執行 N 個低階時間步 (LL Steps) ---
        for i in range(self.ll_steps):
            
            # 1. 低階策略推論：使用未擴展的原始觀察 (current_obs)
            ll_action = self.skill_policy.predict(self.current_obs, self.current_skill)
            
            # 2. 執行環境步驟 (obs, reward, terminated, truncated, info 都是 (1,) 陣列)
            obs, reward, terminated_ll, truncated_ll, info = self.env.step(ll_action)

            # 3. 更新累積獎勵和當前狀態
            # 💡 累積獎勵是陣列加法
            accumulated_reward += reward 
            self.current_obs = obs # 將新的原始觀察狀態儲存

            # 4. 檢查內部技能終止條件 (針對當前單一環境)
            internal_terminate = self._check_skill_termination(self.current_skill, info)
            
            # 5. 如果達到外部終止或內部終止，則結束 LL Steps
            if terminated_ll[0] or truncated_ll[0] or internal_terminate:
                terminated = terminated_ll # 保持 NumPy 陣列 (1,) 格式
                truncated = truncated_ll   # 保持 NumPy 陣列 (1,) 格式
                break # 終止 LL 循環

        # 6. 處理最終狀態和觀察 
        final_obs = self.current_obs
        final_info = info
        
        # 計算最終的進度 
        progress = (i + 1) / self.ll_steps
        
        # 返回擴展後的觀察狀態 (obs, reward, terminated, truncated 都是 (1,) 陣列)
        return self._augment_obs(final_obs, self.current_skill, progress), accumulated_reward, terminated, truncated, final_info