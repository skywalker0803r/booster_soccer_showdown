"""
簡單鍵盤測試 - 排查鍵盤響應問題
"""

import keyboard
import time
import numpy as np

def test_keyboard_detection():
    """測試鍵盤檢測是否正常"""
    
    print("🎮 鍵盤檢測測試")
    print("="*40)
    print("請按以下按鍵測試:")
    print("W, A, S, D, Q, E, Space, ESC")
    print("按住按鍵約1秒，然後鬆開")
    print("ESC退出測試")
    print()
    
    detected_keys = set()
    
    def on_key_event(event):
        if event.event_type == keyboard.KEY_DOWN:
            print(f"✅ 檢測到按鍵: {event.name}")
            detected_keys.add(event.name)
            
            if event.name == 'esc':
                return False  # 停止監聽
    
    # 開始監聽
    print("開始監聽鍵盤...")
    keyboard.hook(on_key_event)
    keyboard.wait('esc')
    keyboard.unhook_all()
    
    print(f"\n📊 測試結果:")
    print(f"檢測到的按鍵: {sorted(detected_keys)}")
    
    # 檢查關鍵按鍵
    required_keys = {'w', 'a', 's', 'd', 'q', 'e', 'space'}
    missing_keys = required_keys - detected_keys
    
    if missing_keys:
        print(f"❌ 缺失的按鍵: {missing_keys}")
        print("💡 建議:")
        print("  1. 以管理員身份運行")
        print("  2. 檢查keyboard庫版本: pip install keyboard==0.13.5")
        print("  3. 嘗試pynput庫作為替代")
    else:
        print("✅ 所有關鍵按鍵都能檢測到！")
    
    return len(missing_keys) == 0

def test_action_generation():
    """測試動作生成邏輯"""
    
    print("\n🤖 動作生成測試")
    print("="*30)
    
    # 模擬KeyboardController的邏輯
    action = np.zeros(12)
    
    test_commands = [
        ('move_forward', 0.1),
        ('turn_left', 0.1),
        ('left_leg_up', 0.2),
        ('kick', 0.5),
    ]
    
    for command, value in test_commands:
        print(f"\n測試命令: {command} (值: {value})")
        old_action = action.copy()
        
        # 模擬命令處理
        if command == 'move_forward':
            action[0] += value * 3  # 左髖
            action[6] += value * 3  # 右髖
            action[1] += value * 2  # 左膝
            action[7] += value * 2  # 右膝
        elif command == 'turn_left':
            action[6:12] += value
            action[0:6] -= value * 0.5
        elif command == 'left_leg_up':
            action[1] += value
        elif command == 'kick':
            action[1:3] += value
            action[7:9] += value * 0.5
        
        # 應用衰減
        action *= 0.95
        action = np.clip(action, -1.0, 1.0)
        
        print(f"  動作變化: {action - old_action}")
        print(f"  當前動作: {action}")
        
        # 檢查是否有變化
        if np.allclose(action, old_action):
            print(f"  ⚠️  警告: 命令 {command} 沒有產生動作變化!")

if __name__ == "__main__":
    print("🔧 鍵盤響應問題診斷工具")
    print("="*50)
    
    # 首先測試鍵盤檢測
    print("Step 1: 測試鍵盤檢測...")
    keyboard_ok = test_keyboard_detection()
    
    if keyboard_ok:
        print("\n✅ 鍵盤檢測正常")
        
        # 測試動作生成
        test_action_generation()
        
        print("\n💡 如果鍵盤檢測正常但機器人不動:")
        print("1. 檢查是否需要以管理員身份運行")
        print("2. 確認焦點在控制台窗口，不在渲染窗口")
        print("3. 嘗試增加動作強度")
        print("4. 檢查是否有其他程序佔用鍵盤")
        
    else:
        print("\n❌ 鍵盤檢測有問題")
        print("💡 建議解決方案:")
        print("1. 以管理員身份運行 Python")
        print("2. 重新安裝 keyboard 庫:")
        print("   pip uninstall keyboard")
        print("   pip install keyboard==0.13.5")
        print("3. 或嘗試替代方案（pynput庫）")