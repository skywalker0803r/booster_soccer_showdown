# -*- coding: utf-8 -*-
# utils.py

import numpy as np

class Preprocessor():

    def get_task_onehot(self, info):
        # 保持不變：獲取 task_index 的 one-hot 編碼 (3 維)
        if 'task_index' in info:
            return info['task_index']
        else:
            # 如果沒有，則返回一個 3 維的零向量
            return np.array([0, 0, 0]) 

    # 保持 quat_rotate_inverse 不變
    def quat_rotate_inverse(self, q: np.ndarray, v: np.ndarray):
        q_w = q[:,[-1]]
        q_vec = q[:,:3]
        a = v * (2.0 * q_w**2 - 1.0)
        b = np.cross(q_vec, v) * (q_w * 2.0)
        c = q_vec * (np.dot(q_vec, v).reshape(-1,1) * 2.0)    
        return a - b + c 

    def modify_state(self, obs, info):
        
        if len(obs.shape) == 1:
            obs = np.expand_dims(obs, axis=0)
        
        # 處理 info 數據擴展
        if len(info["robot_quat"].shape) == 1:
            info["robot_quat"] = np.expand_dims(info["robot_quat"], axis = 0)
            info["robot_gyro"] = np.expand_dims(info["robot_gyro"], axis = 0)
            info["robot_accelerometer"] = np.expand_dims(info["robot_accelerometer"], axis = 0)
            info["robot_velocimeter"] = np.expand_dims(info["robot_velocimeter"], axis = 0)
            
        # 計算 Project Gravity
        quat = info["robot_quat"]
        project_gravity = self.quat_rotate_inverse(quat, np.array([0.0, 0.0, -1.0]))

        # *** 創建 42 維的基礎狀態 ***
        # 12 (qpos) + 12 (qvel) + 3 (proj_grav) + 3 (gyro) + 3 (accel) + 3 (velo) + 3 (ball_xpos) + 3 (ball_velp) = 42
        # 注意: obs 的維度來自 SAI 環境原始輸出
        base_state_42 = np.hstack((
                         obs[:,:12],                 # Joint Positions (12)
                         obs[:,12:24],                # Joint Velocities (12)
                         project_gravity,             # Projected Gravity (3)
                         info["robot_gyro"],          # Robot Gyro (3)
                         info["robot_accelerometer"], # Robot Accelerometer (3)
                         info["robot_velocimeter"],   # Robot Velocimeter (3)
                         obs[:, 24:27],               # Ball Position (Relative to Robot) (3)
                         obs[:, 27:30],               # Ball Linear Velocity (Relative to Robot) (3)
                         ))

        # *** 附加 Task One-Hot 3 維，總計 45 維 ***
        task_onehot = self.get_task_onehot(info)
        if len(task_onehot.shape) == 1:
            task_onehot = np.expand_dims(task_onehot, axis=0)

        # 最終輸出是 45 維
        return np.hstack((base_state_42, task_onehot))

def calculate_potential(state_45: np.ndarray) -> float:
    """
    計算基於保持直立和站穩的勢能函數 Phi(s)。
    
    輸入: 45 維的 Agent 狀態向量 (來自 Preprocessor.modify_state 的輸出)。
    輸出: 浮點數勢能值。
    """
    
    # 1. 重力投影 (Projected Gravity) 勢能: 鼓勵直立
    # 正確索引: [24:27]
    proj_grav = state_45[24:27] 
    target_grav = np.array([0.0, 0.0, -1.0])
    # 點積: 越直立 (越接近 -1)，值越接近 1.0 (正勢能)
    # grav_potential 範圍在 [-1.0, 1.0]
    grav_potential = np.dot(proj_grav, target_grav) 
    
    
    # 2. 線速度 (Robot Velocimeter) 勢能: 鼓勵站穩
    # 正確索引: [33:36]
    robot_velo = state_45[33:36] 
    # 懲罰速度平方 L2 範數。由於 velo 可能很大，需要調整係數。
    # 假設最大速度 L2 範數約為 4.0，則懲罰項約為 -8.0。
    # 我們將係數設置為 -0.05，使其範圍約為 [0, -0.4]
    velo_penalty = -0.05 * np.sum(robot_velo**2)
    
    
    # *** 修正：調整勢能規模使其與原始獎勵 $R_{raw}$ 匹配 ***
    
    # 總勢能 = 姿勢穩定 (佔大頭) + 站穩懲罰
    total_potential = grav_potential + velo_penalty
    
    # 勢能規模調整係數 K。K=0.5 使 $\Phi(s)$ 範圍約在 $[-1.0, 0.5]$ 附近。
    K = 0.5
    
    # 3. 總勢能
    # 最終勢能 $\Phi(s)$
    total_potential = K * total_potential
    
    # 將最終勢能截斷在 $[-1.0, 0.5]$，與原始獎勵 $R_{raw}$ 的規模接近。
    # 這一步對於穩定 Q-值學習至關重要。
    final_potential = np.clip(total_potential, -1.0, 0.5)
    
    return final_potential