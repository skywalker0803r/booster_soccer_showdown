# -*- coding: utf-8 -*-
import numpy as np

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