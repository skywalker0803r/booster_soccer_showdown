# hrl_wrapper.py

import gymnasium as gym
from gymnasium.spaces import Discrete
import numpy as np
import os
from stable_baselines3 import PPO 
from typing import Union, Tuple, Dict, Any

# --- 全域常數/路徑 (確保與 ppo_with_pbrs.py 中的定義一致) ---
# NOTE: 這些檔案必須在 HRL 訓練前就存在
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
            model = PPO.load(path, device="cpu") 
            model.policy.set_training_mode(False) 
            return model
        except Exception as e:
            # 使用 FileNotFoundError 提醒用戶
            raise FileNotFoundError(
                f"❌ 載入 {name} Policy 失敗，請確保已訓練並儲存模型: {path}. 錯誤: {e}"
            )

    def predict(self, obs: np.ndarray, skill_id: int) -> np.ndarray:
        """
        根據 Skill ID 執行對應的低階策略，並返回連續動作。
        obs 應該是 Preprocessor 處理後的狀態。
        """
        if skill_id == 0:
            # Skill 0: Move
            model = self.move_model
        elif skill_id == 1:
            # Skill 1: Kick
            model = self.kick_model
        else:
            # 錯誤處理: 返回零動作 (假設動作空間形狀與 Move Policy 一致)
            return np.zeros(self.move_model.action_space.shape, dtype=np.float32)

        # 使用 deterministic=True 確保推論時的動作是確定性的
        # 由於 SAI 的環境是單一環境，obs 應該是 (obs_dim,) 或 (1, obs_dim)
        action, _ = model.predict(obs, deterministic=True)
        return action # 返回連續動作 (action.shape: (action_dim,))

# --- 2. 高階環境包裝器 (High-Level Wrapper) ---
class HierarchicalWrapper(gym.Wrapper):
    """
    將連續動作環境轉換為離散技能選擇環境，用於訓練高階策略。
    
    高階動作空間: Discrete(2) -> [0: Move, 1: Kick]
    """
    def __init__(self, env: gym.Env, ll_steps: int = 10):
        super().__init__(env)
        
        self.ll_steps = ll_steps
        self.current_obs = None # 儲存當前 Preprocessor 處理後的觀察
        self.current_skill = 0  # 追蹤當前正在執行的技能 ID

        # 改變動作空間：從連續動作 (Box) 變為離散技能選擇 (Discrete(2))
        self.action_space = Discrete(2) 
        
        # 初始化低階策略管理器 (載入訓練好的模型)
        self.skill_policy = SkillPolicy()

    def reset(self, **kwargs) -> Tuple[np.ndarray, Dict[str, Any]]:
        """重置底層環境，並返回第一個 HL 狀態。"""
        obs, info = self.env.reset(**kwargs)
        
        # 儲存初始狀態，供第一個 LL step 使用
        self.current_obs = obs 
        self.current_skill = 0 
        
        return obs, info

    def step(self, action: Union[int, np.ndarray]) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        執行一個高階時間步 (High-Level Step)。
        
        一個 HL Step 包含 N 個 (self.ll_steps) LL Step。
        高階動作 `action` 是離散的技能 ID (0 或 1)。
        """
        # 確保動作是整數 (Skill ID)
        skill_id = int(action.item()) if isinstance(action, np.ndarray) else int(action)
        self.current_skill = skill_id

        accumulated_reward = 0.0
        final_obs = self.current_obs
        final_info = None
        terminated = False
        truncated = False
        
        # --- 執行 N 個低階時間步 (LL Steps) ---
        for i in range(self.ll_steps):
            
            # 1. 低階策略推論：使用上一個時間步的觀察狀態 (current_obs)
            ll_action = self.skill_policy.predict(self.current_obs, self.current_skill)
            
            # 2. 執行環境步驟
            obs, reward, terminated, truncated, info = self.env.step(ll_action)

            # 3. 更新累積獎勵和當前狀態
            accumulated_reward += reward
            self.current_obs = obs # 將新的觀察狀態儲存起來，供下一個 LL step 使用
            
            # 4. 追蹤最終狀態和狀態
            final_obs = obs
            final_info = info

            # 5. 檢查終止條件：如果底層環境終止或截斷，則立即退出循環
            if terminated or truncated:
                break
        
        # --- 返回高階時間步的結果 ---
        # 返回 final_obs 作為下一個 HL 動作的輸入
        return final_obs, accumulated_reward, terminated, truncated, final_info