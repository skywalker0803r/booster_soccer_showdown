"""
調試專家控制 - 檢查為什麼機器人不響應鍵盤
"""

import numpy as np
from sai_rl import SAIClient
import time
import sys
import os

# 添加路徑
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_action_effects():
    """測試不同動作對機器人的影響"""
    
    print("🔍 測試機器人動作響應")
    print("="*50)
    
    # 初始化環境
    sai = SAIClient(comp_id="booster-soccer-showdown", api_key="sai_LFcuaCZiqEkUbNVolQ3wbk5yU7H11jfv")
    env = sai.make_env()
    
    print(f"📊 動作空間信息:")
    print(f"   Shape: {env.action_space.shape}")
    print(f"   Low: {env.action_space.low}")
    print(f"   High: {env.action_space.high}")
    print(f"   動作維度: {env.action_space.shape[0]}")
    
    # 重置環境
    obs, info = env.reset()
    
    # 測試不同強度的動作
    test_actions = [
        ("零動作", np.zeros(12)),
        ("小幅前進", np.array([0.1, 0, 0, 0, 0, 0, 0.1, 0, 0, 0, 0, 0])),
        ("中幅前進", np.array([0.3, 0, 0, 0, 0, 0, 0.3, 0, 0, 0, 0, 0])),
        ("大幅前進", np.array([0.8, 0, 0, 0, 0, 0, 0.8, 0, 0, 0, 0, 0])),
        ("最大前進", np.array([1.0, 0, 0, 0, 0, 0, 1.0, 0, 0, 0, 0, 0])),
        ("隨機動作", np.random.uniform(-0.5, 0.5, 12)),
    ]
    
    for name, normalized_action in test_actions:
        print(f"\n🧪 測試: {name}")
        print(f"   歸一化動作: {normalized_action}")
        
        # 轉換為環境動作
        env_action = env.action_space.low + (env.action_space.high - env.action_space.low) * (normalized_action + 1) / 2
        print(f"   環境動作: {env_action}")
        
        # 重置環境
        obs, info = env.reset()
        
        # 執行動作並觀察效果
        print(f"   執行結果:")
        for step in range(10):  # 執行10步看效果
            next_obs, reward, terminated, truncated, next_info = env.step(env_action)
            
            # 提取機器人位置信息
            robot_pos = next_info.get("robot_xpos", np.zeros(3))
            if len(robot_pos.shape) > 1:
                robot_pos = robot_pos[0]
            
            if step % 3 == 0:  # 每3步打印一次
                print(f"     Step {step}: 位置={robot_pos}, 獎勵={reward:.3f}")
            
            if terminated or truncated:
                print(f"     Episode在第{step}步結束")
                break
                
            time.sleep(0.1)
        
        input(f"   按回車測試下一個動作...")
    
    env.close()
    print("\n✅ 動作測試完成")

def test_keyboard_mapping():
    """測試鍵盤映射是否正確"""
    
    print("\n🎮 測試鍵盤映射")
    print("="*30)
    
    # 模擬simple_expert_collector的鍵盤映射
    key_mapping = {
        'w': ('move_forward', 0.1),
        's': ('move_backward', -0.1),
        'a': ('turn_left', 0.1),
        'd': ('turn_right', -0.1),
        'q': ('left_leg_up', 0.2),
        'e': ('right_leg_up', 0.2),
    }
    
    def process_command(action, command, value):
        """模擬命令處理"""
        action = action.copy()
        
        if command == 'move_forward':
            action[0] += value  # 左髖
            action[6] += value  # 右髖
        elif command == 'move_backward':
            action[0] += value
            action[6] += value
        elif command == 'turn_left':
            action[6:12] += value
            action[0:6] -= value * 0.5
        elif command == 'turn_right':
            action[0:6] += value
            action[6:12] -= value * 0.5
        elif command == 'left_leg_up':
            action[1] += value
        elif command == 'right_leg_up':
            action[7] += value
            
        return action
    
    # 測試每個按鍵的效果
    base_action = np.zeros(12)
    
    for key, (command, value) in key_mapping.items():
        test_action = process_command(base_action, command, value)
        print(f"   按鍵 '{key}' ({command}): {test_action}")
        
        # 檢查是否有變化
        if np.allclose(test_action, base_action):
            print(f"     ⚠️  警告: 按鍵 '{key}' 沒有產生動作變化!")

def enhanced_keyboard_control_test():
    """增強版鍵盤控制測試"""
    
    print("\n🚀 增強版鍵盤控制測試")
    print("="*40)
    
    sai = SAIClient(comp_id="booster-soccer-showdown", api_key="sai_LFcuaCZiqEkUbNVolQ3wbk5yU7H11jfv")
    env = sai.make_env()
    
    # 重置環境
    obs, info = env.reset()
    
    print("🎮 手動測試模式:")
    print("   輸入數字1-6測試不同動作:")
    print("   1: 輕微前進")
    print("   2: 中等前進") 
    print("   3: 強烈前進")
    print("   4: 左轉")
    print("   5: 右轉")
    print("   6: 隨機動作")
    print("   q: 退出")
    
    predefined_actions = {
        '1': np.array([0.1, 0, 0, 0, 0, 0, 0.1, 0, 0, 0, 0, 0]),  # 輕微前進
        '2': np.array([0.3, 0, 0, 0, 0, 0, 0.3, 0, 0, 0, 0, 0]),  # 中等前進
        '3': np.array([0.8, 0, 0, 0, 0, 0, 0.8, 0, 0, 0, 0, 0]),  # 強烈前進
        '4': np.array([-0.2, 0, 0, 0, 0, 0, 0.2, 0, 0, 0, 0, 0]), # 左轉
        '5': np.array([0.2, 0, 0, 0, 0, 0, -0.2, 0, 0, 0, 0, 0]), # 右轉
        '6': np.random.uniform(-0.5, 0.5, 12),                     # 隨機
    }
    
    while True:
        try:
            choice = input("\n選擇動作 (1-6, q退出): ").strip()
            
            if choice == 'q':
                break
                
            if choice in predefined_actions:
                action = predefined_actions[choice]
                print(f"執行動作: {action}")
                
                # 轉換為環境動作
                env_action = env.action_space.low + (env.action_space.high - env.action_space.low) * (action + 1) / 2
                
                # 執行5步觀察效果
                for step in range(5):
                    next_obs, reward, terminated, truncated, next_info = env.step(env_action)
                    
                    # 提取狀態信息
                    robot_pos = next_info.get("robot_xpos", "未知")
                    ball_pos = next_info.get("ball_xpos_rel_robot", "未知")
                    
                    print(f"  Step {step+1}: 獎勵={reward:.3f}")
                    
                    if terminated or truncated:
                        print(f"  Episode結束")
                        obs, info = env.reset()  # 重置
                        break
                
            else:
                print("無效選擇，請輸入1-6或q")
                
        except KeyboardInterrupt:
            break
    
    env.close()
    print("測試結束")

if __name__ == "__main__":
    print("🔧 機器人控制調試工具")
    print("="*60)
    
    print("\n選擇測試模式:")
    print("1. 動作效果測試 (自動)")
    print("2. 鍵盤映射測試")
    print("3. 增強版手動控制測試")
    
    choice = input("請選擇 (1-3): ").strip()
    
    if choice == '1':
        test_action_effects()
    elif choice == '2':
        test_keyboard_mapping()
    elif choice == '3':
        enhanced_keyboard_control_test()
    else:
        print("無效選擇")
        
    print("\n💡 調試建議:")
    print("1. 如果機器人完全不動 → 檢查動作轉換公式")
    print("2. 如果動作太小 → 增加動作強度 (0.1 → 0.5)")
    print("3. 如果機器人摔倒 → 降低動作強度")
    print("4. 如果控制不響應 → 檢查鍵盤庫安裝")