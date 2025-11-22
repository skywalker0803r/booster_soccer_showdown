# -*- coding: utf-8 -*-
# reward_shaper.py
"""
獎勵塑形模組：物理層
接收 env 的原始數據，根據權重計算密集的 Shaped Reward
基於實際環境分析的結果實作
"""

import numpy as np

class RewardShaper:
    def __init__(self):
        """
        初始化獎勵塑形器
        """
        pass
    
    def quat_rotate_inverse(self, q: np.ndarray, v: np.ndarray):
        """
        四元數反向旋轉（從 utils.py 借用）
        用於計算投影重力
        """
        if len(q.shape) == 1:
            q = np.expand_dims(q, axis=0)
        if len(v.shape) == 1:
            v = np.expand_dims(v, axis=0)
            
        q_w = q[:, [-1]]
        q_vec = q[:, :3]
        a = v * (2.0 * q_w**2 - 1.0)
        b = np.cross(q_vec, v) * (q_w * 2.0)
        c = q_vec * (np.dot(q_vec, v.T).T.reshape(-1, 1) * 2.0)    
        return (a - b + c).flatten()

    def compute_reward(self, info, obs, weights):
        """
        計算密集獎勵 (Dense Reward)
        
        Args:
            info: env.step() 返回的 info 字典
            obs: env.step() 返回的 raw observation (45維)
            weights: LLMCoach 返回的權重字典
        
        Returns:
            float: 計算出的塑形獎勵
        """
        reward = 0.0
        
        try:
            # 確保obs是numpy array且為1維
            if isinstance(obs, list):
                obs = np.array(obs)
            obs_flat = obs.flatten()
            
            # =================================================================
            # 1. 平衡獎勵 (Balance Reward)
            # =================================================================
            balance_reward = 0.0
            
            # 1.1 懲罰陀螺儀角速度過大 (防止晃動)
            if 'robot_gyro' in info:
                gyro_penalty = np.sum(np.square(info['robot_gyro']))
                balance_reward -= gyro_penalty * 0.1
            
            # 1.2 獎勵保持直立 (使用投影重力)
            if 'robot_quat' in info:
                robot_quat = info['robot_quat']
                gravity_vector = np.array([0.0, 0.0, -1.0])
                project_gravity = self.quat_rotate_inverse(robot_quat, gravity_vector)
                
                # 計算與目標重力方向的點積 (越直立越接近-1)
                # 我們希望 project_gravity 接近 [0, 0, -1]
                upright_score = -project_gravity[2]  # 取Z分量的負值
                upright_reward = max(0, upright_score)  # 只有當直立時才給獎勵
                balance_reward += upright_reward * 0.5
            
            # 1.3 懲罰過大的線性速度 (鼓勵穩定)
            if 'robot_velocimeter' in info:
                velocity_penalty = np.sum(np.square(info['robot_velocimeter']))
                balance_reward -= velocity_penalty * 0.05
            
            # 1.4 懲罰過大的加速度變化 (平滑移動)
            if 'robot_accelerometer' in info:
                accel_penalty = np.sum(np.square(info['robot_accelerometer'])) * 0.001
                balance_reward -= accel_penalty
            
            # 應用平衡權重
            reward += weights.get('balance', 0) * balance_reward
            
            # =================================================================
            # 2. 進度獎勵 (Progress Reward)
            # =================================================================
            progress_reward = 0.0
            
            # 2.1 與球的距離獎勵 (基於環境分析: obs[24:27]是球位置)
            if len(obs_flat) >= 27:
                # 根據環境分析，obs[24:27]是球相對機器人的位置
                ball_relative_pos = obs_flat[24:27] 
                dist_to_ball = np.linalg.norm(ball_relative_pos)
                
                # 距離越近獎勵越高 (使用倒數函數)
                if dist_to_ball > 0:
                    proximity_reward = 1.0 / (1.0 + dist_to_ball)
                    progress_reward += proximity_reward * 0.5
                
                # 額外獎勵：當非常接近球時 (< 1.0m)
                if dist_to_ball < 1.0:
                    close_reward = (1.0 - dist_to_ball) * 0.3
                    progress_reward += close_reward
            
            # 2.2 移動獎勵 (基於velocimeter，但只在有進度方向時)
            if 'robot_velocimeter' in info:
                # 獎勵朝向球的移動
                if len(obs_flat) >= 27:
                    ball_relative_pos = obs_flat[24:27] 
                    if np.linalg.norm(ball_relative_pos) > 0:
                        # 計算朝向球的方向
                        ball_direction = ball_relative_pos / np.linalg.norm(ball_relative_pos)
                        # 計算速度在球方向上的投影
                        velocity = info['robot_velocimeter']
                        velocity_towards_ball = np.dot(velocity, ball_direction)
                        if velocity_towards_ball > 0:  # 只獎勵朝向球的移動
                            progress_reward += velocity_towards_ball * 0.2
            
            # 2.3 球移動獎勵 (如果球在朝目標移動)
            if 'ball_velp_rel_robot' in info:
                ball_velocity = info['ball_velp_rel_robot']
                # 簡單的假設：X軸正方向是朝向目標
                if ball_velocity[0] > 0:  # 球向前移動
                    ball_progress_reward = ball_velocity[0] * 0.1
                    progress_reward += ball_progress_reward
            
            # 應用進度權重
            reward += weights.get('progress', 0) * progress_reward
            
            # =================================================================
            # 3. 能量效率獎勵 (Energy Efficiency)
            # =================================================================
            energy_penalty = 0.0
            
            # 3.1 懲罰關節速度過大 (obs[12:24]是關節速度)
            if len(obs_flat) >= 24:
                joint_velocities = obs_flat[12:24]
                joint_velocity_penalty = np.sum(np.square(joint_velocities)) * 0.01
                energy_penalty += joint_velocity_penalty
            
            # 3.2 懲罰關節位置偏離中性位置過遠 (obs[0:12]是關節位置)
            if len(obs_flat) >= 12:
                joint_positions = obs_flat[0:12]
                # 假設中性位置接近0（根據初始觀察）
                joint_position_penalty = np.sum(np.square(joint_positions)) * 0.005
                energy_penalty += joint_position_penalty
            
            # 應用能量權重 (注意這是懲罰，所以是減法)
            reward -= weights.get('energy', 0) * energy_penalty
            
            # =================================================================
            # 4. 獎勵範圍限制和歸一化
            # =================================================================
            
            # 將獎勵限制在合理範圍內，避免破壞TD3的Critic估計
            reward = np.clip(reward, -2.0, 2.0)
            
        except Exception as e:
            # 防止索引錯誤或其他異常導致訓練崩潰
            print(f"⚠️ RewardShaper警告: {e}")
            reward = 0.0
            
        return reward
    
    def get_reward_breakdown(self, info, obs, weights):
        """
        獲取獎勵的詳細分解，用於調試和分析
        
        Returns:
            dict: 包含各部分獎勵的詳細字典
        """
        breakdown = {
            'total': 0.0,
            'balance': 0.0,
            'progress': 0.0,
            'energy': 0.0,
            'components': {}
        }
        
        try:
            obs_flat = np.array(obs).flatten()
            
            # 計算各部分獎勵 (簡化版，用於分析)
            
            # 平衡部分
            balance_reward = 0.0
            if 'robot_gyro' in info:
                gyro_penalty = np.sum(np.square(info['robot_gyro']))
                breakdown['components']['gyro_penalty'] = -gyro_penalty * 0.1
                balance_reward += breakdown['components']['gyro_penalty']
            
            # 進度部分  
            progress_reward = 0.0
            if len(obs_flat) >= 27:
                ball_relative_pos = obs_flat[24:27]
                dist_to_ball = np.linalg.norm(ball_relative_pos)
                if dist_to_ball > 0:
                    proximity_reward = 1.0 / (1.0 + dist_to_ball) * 0.5
                    breakdown['components']['ball_proximity'] = proximity_reward
                    progress_reward += proximity_reward
            
            breakdown['balance'] = balance_reward * weights.get('balance', 0)
            breakdown['progress'] = progress_reward * weights.get('progress', 0)
            breakdown['total'] = breakdown['balance'] + breakdown['progress'] + breakdown['energy']
            
        except Exception as e:
            print(f"⚠️ RewardShaper breakdown警告: {e}")
            
        return breakdown