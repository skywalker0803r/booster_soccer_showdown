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