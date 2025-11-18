"""
簡化版專家軌跡收集器
直接使用SAI環境，避免booster_control依賴問題
使用鍵盤直接控制動作輸出
"""

import numpy as np
import pickle
import os
from datetime import datetime
import keyboard  # pip install keyboard
from sai_rl import SAIClient
import time
import threading
import queue


class KeyboardController:
    """簡單的鍵盤控制器"""
    
    def __init__(self):
        self.action = np.zeros(12)  # 12關節動作
        self.running = True
        self.command_queue = queue.Queue()
        
        # 按鍵映射 - 增強動作強度
        self.key_mapping = {
            # 基礎移動 (大幅增強)
            'w': ('move_forward', 0.3),     # 從0.1增加到0.3
            's': ('move_backward', -0.3),   # 從-0.1增加到-0.3
            'a': ('turn_left', 0.3),        # 從0.1增加到0.3
            'd': ('turn_right', -0.3),      # 從-0.1增加到-0.3
            
            # 腿部控制 (增強)
            'q': ('left_leg_up', 0.4),      # 從0.2增加到0.4
            'e': ('right_leg_up', 0.4),     # 從0.2增加到0.4
            'z': ('left_leg_down', -0.4),   # 從-0.2增加到-0.4
            'c': ('right_leg_down', -0.4),  # 從-0.2增加到-0.4
            
            # 踢球動作 (保持較大)
            'space': ('kick', 0.6),         # 從0.5增加到0.6
            
            # 重置
            'r': ('reset', 0),
        }
        
        print("🎮 鍵盤控制器初始化")
        print("📖 控制說明:")
        print("   W/S: 前進/後退")
        print("   A/D: 左轉/右轉") 
        print("   Q/E: 左腿/右腿抬起")
        print("   Z/C: 左腿/右腿放下")
        print("   Space: 踢球")
        print("   R: 重置動作")
        print("   ESC: 退出")
        
    def start_keyboard_listener(self):
        """啟動鍵盤監聽 - 改為持續按鍵模式"""
        
        self.pressed_keys = set()  # 當前按下的鍵
        
        def on_key_press(event):
            if event.name == 'esc':
                print("🚪 ESC按下，準備退出...")
                self.running = False
                return
            
            if event.name in self.key_mapping and event.name not in self.pressed_keys:
                self.pressed_keys.add(event.name)
                print(f"🎮 按鍵按下: {event.name}")
        
        def on_key_release(event):
            if event.name in self.pressed_keys:
                self.pressed_keys.remove(event.name)
                print(f"🎮 按鍵釋放: {event.name}")
        
        keyboard.on_press(on_key_press)
        keyboard.on_release(on_key_release)
        
        print("✅ 鍵盤監聽已啟動 (持續按鍵模式)")
        print("💡 按住 W/A/S/D 鍵會持續產生動作")
        print("🎮 鬆開按鍵會停止動作")
    
    def get_action(self):
        """獲取當前動作 - 基於持續按鍵"""
        
        # 重置動作（每次重新計算）
        self.action = np.zeros(12)
        
        # 處理當前按下的所有按鍵
        active_commands = []
        for key_name in self.pressed_keys:
            if key_name in self.key_mapping:
                command, value = self.key_mapping[key_name]
                self._process_command(command, value)
                active_commands.append(f"{command}({value})")
        
        # 顯示活動命令
        if active_commands:
            print(f"🎮 活動命令: {', '.join(active_commands)}")
            print(f"📊 合成動作: {self.action}")
        
        # 限制動作範圍
        self.action = np.clip(self.action, -1.0, 1.0)
        
        return self.action.copy()
    
    def _process_command(self, command, value):
        """處理控制命令 - 修復動作強度和映射"""
        
        if command == 'reset':
            self.action = np.zeros(12)
            
        elif command == 'move_forward':
            # 前進：髖關節前屈 + 膝蓋彎曲 (增強動作強度)
            self.action[0] += value * 8   # 左髖前屈 (大幅增強)
            self.action[6] += value * 8   # 右髖前屈
            self.action[1] += value * 5   # 左膝彎曲
            self.action[7] += value * 5   # 右膝彎曲
            # 添加踝關節支撐
            self.action[2] += value * 3   # 左踝
            self.action[8] += value * 3   # 右踝
            
        elif command == 'move_backward':
            # 後退：髖關節後伸
            self.action[0] += value * 8   # 負值表示後伸
            self.action[6] += value * 8
            self.action[1] += value * 3   # 輕微膝蓋彎曲保持平衡
            self.action[7] += value * 3
            
        elif command == 'turn_left':
            # 左轉：右腿更多動作，左腿減少
            self.action[6] += value * 6   # 右髖
            self.action[7] += value * 4   # 右膝  
            self.action[0] -= value * 2   # 左髖減少
            self.action[1] -= value * 1   # 左膝減少
            
        elif command == 'turn_right':
            # 右轉：左腿更多動作，右腿減少  
            self.action[0] += value * 6   # 左髖
            self.action[1] += value * 4   # 左膝
            self.action[6] -= value * 2   # 右髖減少
            self.action[7] -= value * 1   # 右膝減少
            
        elif command == 'left_leg_up':
            # 左腿抬起：髖關節屈曲 + 膝關節彎曲
            self.action[0] += value * 8   # 左髖屈曲
            self.action[1] += value * 6   # 左膝彎曲
            
        elif command == 'right_leg_up':
            # 右腿抬起
            self.action[6] += value * 8   # 右髖屈曲  
            self.action[7] += value * 6   # 右膝彎曲
            
        elif command == 'left_leg_down':
            # 左腿向下：髖關節伸展
            self.action[0] += value * 6   # 負值表示伸展
            self.action[1] += value * 4
            
        elif command == 'right_leg_down':
            # 右腿向下
            self.action[6] += value * 6
            self.action[7] += value * 4
            
        elif command == 'kick':
            # 踢球動作：快速有力的腿部擺動
            self.action[0] += value * 10  # 左髖大幅前擺
            self.action[1] += value * 8   # 左膝快速伸展
            self.action[6] += value * 3   # 右腿支撐
            self.action[7] += value * 2   # 右膝輕微彎曲保持平衡


class SimpleExpertCollector:
    """簡化版專家軌跡收集器"""
    
    def __init__(self, save_dir="expert_data"):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # SAI環境 (啟用視覺化)
        print("🔄 初始化SAI環境...")
        self.sai = SAIClient(comp_id="booster-soccer-showdown", api_key="sai_LFcuaCZiqEkUbNVolQ3wbk5yU7H11jfv")
        self.env = self.sai.make_env()
        
        # 強制啟用渲染
        print("🎨 啟用視覺化渲染...")
        self.env.unwrapped.render_mode = "human"
        try:
            # 嘗試渲染一幀
            self.env.render()
            print("✅ 視覺化窗口應該已開啟")
        except Exception as e:
            print(f"⚠️ 視覺化可能有問題: {e}")
            print("💡 如果看不到畫面，請檢查顯示驅動")
        
        # Preprocessor
        import sys
        current_dir = os.path.dirname(os.path.abspath(__file__))
        sys.path.append(current_dir)
        from main_improved_dreamerv3 import Preprocessor
        self.preprocessor = Preprocessor()
        
        # 鍵盤控制器
        self.keyboard_controller = KeyboardController()
        
        # 數據收集
        self.all_trajectories = []
        self.episode_count = 0
        
        print("✅ 簡化版專家收集器初始化完成")
    
    def collect_episode(self):
        """收集單個episode"""
        
        self.episode_count += 1
        print(f"\n🎮 Episode {self.episode_count} 開始...")
        print("💡 使用鍵盤控制機器人，ESC退出")
        
        # 重置環境
        obs, info = self.env.reset()
        obs_processed = self.preprocessor.modify_state(obs, info).squeeze()
        
        # Episode數據
        episode_data = {
            'observations': [obs_processed],
            'actions': [],
            'rewards': [],
            'episode_reward': 0,
            'start_time': time.time()
        }
        
        # Episode循環
        step = 0
        while self.keyboard_controller.running and step < 800:
            
            # 獲取鍵盤動作
            action_normalized = self.keyboard_controller.get_action()
            
            # 如果動作不為零，顯示詳細信息
            if np.any(np.abs(action_normalized) > 0.001):
                print(f"🎮 歸一化動作 [-1,1]: {action_normalized}")
            
            # 轉換為環境動作
            env_action = self.env.action_space.low + (self.env.action_space.high - self.env.action_space.low) * (action_normalized + 1) / 2
            
            # 顯示最終環境動作
            if np.any(np.abs(action_normalized) > 0.001):
                print(f"🔧 環境動作: {env_action}")
                print(f"📏 動作範圍: [{self.env.action_space.low[0]:.1f}, {self.env.action_space.high[0]:.1f}]")
            
            # 執行動作
            next_obs, reward, terminated, truncated, next_info = self.env.step(env_action)
            next_obs_processed = self.preprocessor.modify_state(next_obs, next_info).squeeze()
            
            # 記錄數據
            episode_data['actions'].append(action_normalized)
            episode_data['rewards'].append(reward)
            episode_data['episode_reward'] += reward
            episode_data['observations'].append(next_obs_processed)
            
            step += 1
            
            # 顯示實時信息
            if step % 50 == 0:
                print(f"   Step {step}, 當前獎勵: {episode_data['episode_reward']:.3f}")
            
            if terminated or truncated:
                print(f"   Episode在第{step}步結束")
                break
            
            # 控制頻率
            time.sleep(0.05)  # 20Hz
        
        # Episode結束
        duration = time.time() - episode_data['start_time']
        reward = episode_data['episode_reward']
        
        print(f"\n📊 Episode {self.episode_count} 結果:")
        print(f"   獎勵: {reward:.3f}")
        print(f"   步數: {len(episode_data['actions'])}")
        print(f"   時長: {duration:.1f}秒")
        
        # 詢問是否保存
        save_decision = self._ask_save_decision(reward)
        
        if save_decision:
            self._save_episode(episode_data)
        
        return episode_data if save_decision else None
    
    def _ask_save_decision(self, reward):
        """詢問保存決定"""
        
        if reward > 0:
            print("🎉 正獎勵！強烈建議保存")
            default = 'y'
        elif reward > -10:
            print("😐 表現還可以，建議保存")
            default = 'y'
        else:
            print("😞 表現較差")
            default = 'n'
        
        while True:
            try:
                choice = input(f"💾 保存這個episode嗎？[{default}/other]: ").strip().lower()
                if not choice:
                    choice = default.lower()
                
                if choice in ['y', 'yes']:
                    return True
                elif choice in ['n', 'no']:
                    return False
                else:
                    print("請輸入 y/yes 或 n/no")
                    
            except KeyboardInterrupt:
                print("\n中斷保存詢問，不保存此episode")
                return False
    
    def _save_episode(self, episode_data):
        """保存episode"""
        
        # 轉換為numpy
        trajectory = {
            'observations': np.array(episode_data['observations'][:-1]),  # 移除最後一個obs
            'actions': np.array(episode_data['actions']),
            'rewards': np.array(episode_data['rewards']),
            'episode_reward': episode_data['episode_reward'],
            'collection_time': datetime.now().isoformat(),
            'episode_id': self.episode_count
        }
        
        self.all_trajectories.append(trajectory)
        
        # 自動保存
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"simple_expert_trajectories_{timestamp}.pkl"
        filepath = os.path.join(self.save_dir, filename)
        
        with open(filepath, 'wb') as f:
            pickle.dump(self.all_trajectories, f)
        
        print(f"✅ Episode已保存！總軌跡數: {len(self.all_trajectories)}")
        
        # 複製到標準位置供BC使用
        standard_path = os.path.join(self.save_dir, "expert_trajectories.pkl")
        with open(standard_path, 'wb') as f:
            pickle.dump(self.all_trajectories, f)
    
    def run_collection(self):
        """運行收集流程"""
        
        print("🚀 開始專家軌跡收集")
        print("="*50)
        
        # 啟動鍵盤監聽
        self.keyboard_controller.start_keyboard_listener()
        
        try:
            while self.keyboard_controller.running:
                # 收集episode
                episode = self.collect_episode()
                
                if not self.keyboard_controller.running:
                    break
                
                # 詢問是否繼續
                try:
                    continue_choice = input("\n🔄 收集下一個episode嗎？[Y/n]: ").strip().lower()
                    if continue_choice in ['n', 'no']:
                        break
                except KeyboardInterrupt:
                    break
        
        except KeyboardInterrupt:
            print("\n⚠️ 收集被中斷")
        
        finally:
            keyboard.unhook_all()
            self.env.close()
            
            # 最終統計
            if self.all_trajectories:
                rewards = [t['episode_reward'] for t in self.all_trajectories]
                print(f"\n📊 收集完成統計:")
                print(f"   總軌跡數: {len(self.all_trajectories)}")
                print(f"   平均獎勵: {np.mean(rewards):.3f}")
                print(f"   最佳獎勵: {max(rewards):.3f}")
                print(f"   軌跡已保存到: {self.save_dir}")
                print(f"\n💡 現在可以運行: python behavioral_cloning.py")
            else:
                print("\n😞 沒有收集到軌跡")


if __name__ == "__main__":
    print("🎯 簡化版專家軌跡收集器")
    print("⚠️  需要先安裝: pip install keyboard")
    print("🎮 使用鍵盤直接控制機器人動作")
    
    try:
        collector = SimpleExpertCollector()
        collector.run_collection()
    except ImportError as e:
        print(f"❌ 導入錯誤: {e}")
        print("💡 請安裝缺失的包: pip install keyboard")
    except Exception as e:
        print(f"❌ 運行錯誤: {e}")
        import traceback
        traceback.print_exc()