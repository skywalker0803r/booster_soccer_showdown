# -*- coding: utf-8 -*-
"""
Gym to Gymnasium 兼容性適配器
解決 SB3 期望 Gymnasium 但 SAI 使用 Gym 的問題
"""

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import Wrapper
    USING_GYMNASIUM = True
    print("🔧 使用 Gymnasium")
except ImportError:
    import gym
    from gym import Wrapper
    USING_GYMNASIUM = False
    print("🔧 使用 OpenAI Gym")


class GymToGymnasiumWrapper(Wrapper):
    """
    將 OpenAI Gym 環境轉換為 Gymnasium 兼容格式
    主要處理 step() 返回值的差異
    """
    
    def __init__(self, env):
        super().__init__(env)
        
    def step(self, action):
        """
        Gym: (obs, reward, done, info)
        Gymnasium: (obs, reward, terminated, truncated, info)
        """
        if hasattr(self.env, 'step'):
            result = self.env.step(action)
            
            if len(result) == 4:
                # 舊版 Gym 格式: (obs, reward, done, info)
                obs, reward, done, info = result
                # 將 done 拆分為 terminated 和 truncated
                # 簡單策略: 如果 episode 結束就設為 terminated
                terminated = done
                truncated = False
                return obs, reward, terminated, truncated, info
            elif len(result) == 5:
                # 已經是新格式或 Gymnasium
                return result
            else:
                raise ValueError(f"意外的 step() 返回值長度: {len(result)}")
        else:
            raise AttributeError("環境沒有 step() 方法")
    
    def reset(self, **kwargs):
        """
        確保 reset() 返回 (obs, info) 格式
        """
        result = self.env.reset(**kwargs)
        
        if isinstance(result, tuple) and len(result) == 2:
            # 已經是 (obs, info) 格式
            return result
        else:
            # 舊格式，只返回 obs
            return result, {}


def make_gymnasium_compatible(env):
    """
    讓任何環境兼容 SB3 的 Gymnasium 要求
    """
    if USING_GYMNASIUM:
        # 如果已經使用 Gymnasium，直接包裝確保格式正確
        return GymToGymnasiumWrapper(env)
    else:
        # 如果使用 OpenAI Gym，需要轉換
        return GymToGymnasiumWrapper(env)


# 兼容性測試函數
def test_compatibility(env):
    """測試環境兼容性"""
    print(f"🧪 測試環境兼容性...")
    
    # 測試 reset
    try:
        reset_result = env.reset()
        if isinstance(reset_result, tuple) and len(reset_result) == 2:
            print("✅ reset() 格式正確: (obs, info)")
        else:
            print(f"⚠️ reset() 格式: {type(reset_result)}")
    except Exception as e:
        print(f"❌ reset() 測試失敗: {e}")
    
    # 測試 step
    try:
        obs, info = env.reset()
        action = env.action_space.sample()
        step_result = env.step(action)
        
        if len(step_result) == 5:
            print("✅ step() 格式正確: (obs, reward, terminated, truncated, info)")
        else:
            print(f"⚠️ step() 格式長度: {len(step_result)}")
    except Exception as e:
        print(f"❌ step() 測試失敗: {e}")
    
    # 測試空間
    try:
        obs_space = env.observation_space
        action_space = env.action_space
        print(f"✅ 觀察空間: {obs_space}")
        print(f"✅ 動作空間: {action_space}")
    except Exception as e:
        print(f"❌ 空間測試失敗: {e}")