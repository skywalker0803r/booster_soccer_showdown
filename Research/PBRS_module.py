# -*- coding: utf-8 -*-
import numpy as np

def calculate_potential(state_45: np.ndarray) -> float:
    """
    計算基於保持直立、站穩和球控制的勢能函數 Phi(s)。
    
    45維狀態結構 (來自 utils.py Preprocessor.modify_state):
    [0:12]   Joint Positions (12)
    [12:24]  Joint Velocities (12) 
    [24:27]  Projected Gravity (3)     ← 直立控制
    [27:30]  Robot Gyro (3)           ← 角速度穩定
    [30:33]  Robot Accelerometer (3)   ← 加速度穩定
    [33:36]  Robot Velocimeter (3)     ← 線速度控制
    [36:39]  Ball Position (3)         ← 球距離控制
    [39:42]  Ball Velocity (3)         ← 球速度控制
    [42:45]  Task One-Hot (3)
    
    輸出: 浮點數勢能值 [-1.0, 1.0]
    """
    
    # 1. 直立穩定勢能 (Projected Gravity)
    proj_grav = state_45[24:27]  # ✅ 修正索引
    target_grav = np.array([0.0, 0.0, -1.0])
    # 點積: 越直立越接近1.0
    grav_potential = np.dot(proj_grav, target_grav) 
    
    # 2. 線速度穩定勢能 (Robot Velocimeter) 
    robot_velo = state_45[33:36]  # ✅ 修正索引
    # 懲罰過度移動，鼓勵穩定控制
    velo_penalty = -0.03 * np.sum(robot_velo**2)  # 調整係數
    
    # 3. 🆕 角速度穩定勢能 (Robot Gyro)
    robot_gyro = state_45[27:30]  
    # 懲罰過度旋轉，鼓勵平衡
    gyro_penalty = -0.02 * np.sum(robot_gyro**2)
    
    # 4. 🆕 球距離控制勢能 (Ball Position)
    ball_pos = state_45[36:39]
    ball_distance = np.linalg.norm(ball_pos[:2])  # 只考慮xy距離
    # 鼓勵接近球，但不要太近
    optimal_distance = 1.0  # 最佳踢球距離
    distance_reward = -0.1 * abs(ball_distance - optimal_distance)
    
    # 5. 🆕 關節速度懲罰 (Joint Velocities)
    joint_velo = state_45[12:24]
    # 避免關節劇烈運動
    joint_penalty = -0.01 * np.sum(joint_velo**2)
    
    
    # 6. 總勢能組合 - 分層權重設計
    stability_component = grav_potential + velo_penalty + gyro_penalty + joint_penalty  # 穩定性
    ball_component = distance_reward  # 球控制
    
    # 🎯 階段性權重: 先學穩定，再學球控制
    stability_weight = 0.7  # 穩定性佔70%
    ball_weight = 0.3      # 球控制佔30%
    
    total_potential = stability_weight * stability_component + ball_weight * ball_component
    
    # 7. 規模調整 - 匹配原始獎勵規模
    K = 0.4  # 調整係數，避免過度影響原始獎勵
    scaled_potential = K * total_potential
    
    # 8. 最終勢能裁剪 - 防止數值溢出
    final_potential = np.clip(scaled_potential, -1.0, 0.8)
    
    return final_potential


def create_pbrs_wrapper(env, gamma=0.99, debug=False):
    """
    創建 PBRS 包裝器的便利函數
    
    Args:
        env: 原始環境
        gamma: 折扣因子
        debug: 是否輸出調試信息
    
    Returns:
        包裝後的環境
    """
    return PBRSWrapper(env, gamma=gamma, debug=debug)


class PBRSWrapper:
    """
    PBRS (Potential-Based Reward Shaping) 環境包裝器
    
    根據 Ng, Harada & Russell (1999) 的理論，
    使用勢能函數進行獎勵塑形，保證最優策略不變性
    """
    
    def __init__(self, env, gamma=0.99, debug=False):
        self.env = env
        self.gamma = gamma
        self.debug = debug
        self.prev_potential = 0.0
        self.step_count = 0
        self.total_shaped_reward = 0.0
        
    def reset(self, **kwargs):
        """重置環境"""
        obs, info = self.env.reset(**kwargs)
        
        # 使用 Preprocessor 處理觀測
        from utils import Preprocessor
        preprocessor = Preprocessor()
        processed_obs = preprocessor.modify_state(obs, info)
        
        # 計算初始勢能
        self.prev_potential = calculate_potential(processed_obs[0])
        self.step_count = 0
        self.total_shaped_reward = 0.0
        
        if self.debug:
            print(f"🔄 PBRS Reset - Initial potential: {self.prev_potential:.3f}")
            
        return obs, info
    
    def step(self, action):
        """環境步進"""
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # 處理觀測
        from utils import Preprocessor
        preprocessor = Preprocessor()
        processed_obs = preprocessor.modify_state(obs, info)
        
        # 計算新勢能
        current_potential = calculate_potential(processed_obs[0])
        
        # PBRS 獎勵塑形: R' = R + γ*Φ(s') - Φ(s)
        if terminated or truncated:
            # Episode結束時，Φ(s') = 0
            shaped_reward = reward - self.prev_potential
        else:
            shaped_reward = reward + self.gamma * current_potential - self.prev_potential
        
        # 更新狀態
        self.prev_potential = current_potential
        self.step_count += 1
        self.total_shaped_reward += (shaped_reward - reward)
        
        if self.debug and self.step_count % 100 == 0:
            print(f"📊 Step {self.step_count}: Original={reward:.3f}, "
                  f"Shaped={shaped_reward:.3f}, Potential={current_potential:.3f}")
        
        return obs, shaped_reward, terminated, truncated, info
    
    def __getattr__(self, name):
        """代理到原始環境"""
        return getattr(self.env, name)