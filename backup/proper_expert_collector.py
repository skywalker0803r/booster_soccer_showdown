"""
正確的專家軌跡收集器
使用真正的機器人運動學控制，而不是瞎猜關節映射
"""

import numpy as np
import pickle
import os
from datetime import datetime
import keyboard
from sai_rl import SAIClient
import time
import sys

# 嘗試導入機器人控制系統
try:
    # 添加路徑
    current_dir = os.path.dirname(os.path.abspath(__file__))
    booster_control_path = os.path.join(os.path.dirname(current_dir), 'booster_control')
    sys.path.insert(0, booster_control_path)
    
    from t1_utils import LowerT1JoyStick
    ROBOT_CONTROL_AVAILABLE = True
    print("✅ 成功導入機器人控制系統")
except Exception as e:
    print(f"⚠️ 無法導入機器人控制系統: {e}")
    ROBOT_CONTROL_AVAILABLE = False


class RobotCommandGenerator:
    """機器人指令生成器"""
    
    def __init__(self):
        # 基礎運動命令 (se3格式)
        self.base_command = np.zeros(6)  # [x, y, z, roll, pitch, yaw]
        self.movement_scale = 0.5  # 運動縮放因子
        
        # 命令映射
        self.command_mapping = {
            # 基礎移動 (在機器人座標系中)
            'move_forward': np.array([0.5, 0, 0, 0, 0, 0]),     # X正方向
            'move_backward': np.array([-0.5, 0, 0, 0, 0, 0]),   # X負方向
            'move_left': np.array([0, 0.3, 0, 0, 0, 0]),        # Y正方向
            'move_right': np.array([0, -0.3, 0, 0, 0, 0]),      # Y負方向
            'turn_left': np.array([0, 0, 0, 0, 0, 0.3]),        # 逆時針轉
            'turn_right': np.array([0, 0, 0, 0, 0, -0.3]),      # 順時針轉
            
            # 複合動作
            'forward_left': np.array([0.5, 0.2, 0, 0, 0, 0.1]), # 前進+左轉
            'forward_right': np.array([0.5, -0.2, 0, 0, 0, -0.1]), # 前進+右轉
            
            # 重心調整
            'lean_forward': np.array([0, 0, 0, 0, 0.1, 0]),     # 向前傾
            'lean_back': np.array([0, 0, 0, 0, -0.1, 0]),       # 向後傾
            'lean_left': np.array([0, 0, 0, 0.1, 0, 0]),        # 向左傾
            'lean_right': np.array([0, 0, 0, -0.1, 0, 0]),      # 向右傾
            
            # 高度調整
            'stand_up': np.array([0, 0, 0.1, 0, 0, 0]),         # 站高一點
            'crouch_down': np.array([0, 0, -0.1, 0, 0, 0]),     # 蹲低一點
        }
        
        print("🤖 機器人指令生成器初始化")
        print("📖 可用指令:")
        for cmd in self.command_mapping.keys():
            print(f"   {cmd}")
    
    def get_command(self, command_name, intensity=1.0):
        """獲取機器人指令"""
        if command_name in self.command_mapping:
            return self.command_mapping[command_name] * intensity * self.movement_scale
        else:
            return np.zeros(6)


class ProperKeyboardController:
    """使用真正機器人控制邏輯的鍵盤控制器"""
    
    def __init__(self, env):
        self.env = env
        self.running = True
        self.pressed_keys = set()
        
        # 機器人控制系統
        if ROBOT_CONTROL_AVAILABLE:
            self.robot_controller = LowerT1JoyStick(env.unwrapped)
            print("✅ 使用真正的機器人控制系統")
        else:
            self.robot_controller = None
            print("❌ 回退到命令生成器")
        
        # 指令生成器
        self.command_generator = RobotCommandGenerator()
        
        # 按鍵到指令映射
        self.key_to_command = {
            'w': 'move_forward',
            's': 'move_backward', 
            'a': 'turn_left',
            'd': 'turn_right',
            'q': 'move_left',
            'e': 'move_right',
            'r': 'lean_forward',
            'f': 'lean_back',
            't': 'stand_up',
            'g': 'crouch_down',
            'z': 'forward_left',
            'c': 'forward_right',
        }
        
        print("🎮 真正的機器人控制器初始化")
        print("📖 控制說明:")
        print("   W/S: 前進/後退")
        print("   A/D: 左轉/右轉")
        print("   Q/E: 橫向移動")
        print("   R/F: 前傾/後傾")
        print("   T/G: 站高/蹲低")
        print("   Z/C: 前進轉彎")
        print("   ESC: 退出")
    
    def start_keyboard_listener(self):
        """啟動鍵盤監聽"""
        
        def on_key_press(event):
            if event.name == 'esc':
                print("🚪 ESC按下，準備退出...")
                self.running = False
                return
            
            if event.name in self.key_to_command and event.name not in self.pressed_keys:
                self.pressed_keys.add(event.name)
                command_name = self.key_to_command[event.name]
                print(f"🎮 按鍵按下: {event.name} → {command_name}")
        
        def on_key_release(event):
            if event.name in self.pressed_keys:
                self.pressed_keys.remove(event.name)
                print(f"🎮 按鍵釋放: {event.name}")
        
        keyboard.on_press(on_key_press)
        keyboard.on_release(on_key_release)
        
        print("✅ 鍵盤監聽已啟動")
    
    def get_robot_action(self, observation, info):
        """獲取機器人動作 - 使用真正的控制邏輯"""
        
        # 合成SE3指令
        combined_command = np.zeros(6)
        active_commands = []
        
        for key_name in self.pressed_keys:
            if key_name in self.key_to_command:
                command_name = self.key_to_command[key_name]
                command = self.command_generator.get_command(command_name)
                combined_command += command
                active_commands.append(command_name)
        
        if active_commands:
            print(f"🎮 活動指令: {', '.join(active_commands)}")
            print(f"📊 合成SE3指令: {combined_command}")
        
        # 使用機器人控制器轉換為關節動作
        if self.robot_controller is not None:
            try:
                # 使用真正的機器人控制系統
                joint_action = self.robot_controller.get_actions(combined_command, observation, info)
                if isinstance(joint_action, tuple):
                    joint_action = joint_action[0]  # 取第一個元素
                
                print(f"🤖 機器人關節動作: {joint_action}")
                return joint_action
                
            except Exception as e:
                print(f"⚠️ 機器人控制器錯誤: {e}")
                # 回退到零動作
                return np.zeros(self.env.action_space.shape[0])
        else:
            # 如果沒有機器人控制器，返回零動作
            return np.zeros(self.env.action_space.shape[0])


class ProperExpertCollector:
    """使用正確機器人控制的專家軌跡收集器"""
    
    def __init__(self, save_dir="expert_data"):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # SAI環境
        print("🔄 初始化SAI環境...")
        self.sai = SAIClient(comp_id="booster-soccer-showdown", api_key="sai_LFcuaCZiqEkUbNVolQ3wbk5yU7H11jfv")
        self.env = self.sai.make_env()
        
        # 視覺化
        print("🎨 啟用視覺化...")
        self.env.unwrapped.render_mode = "human"
        try:
            self.env.render()
            print("✅ 視覺化窗口已開啟")
        except Exception as e:
            print(f"⚠️ 視覺化問題: {e}")
        
        # Preprocessor
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        from main_improved_dreamerv3 import Preprocessor
        self.preprocessor = Preprocessor()
        
        # 機器人控制器
        self.keyboard_controller = ProperKeyboardController(self.env)
        
        # 數據收集
        self.all_trajectories = []
        self.episode_count = 0
        
        print("✅ 正確的專家收集器初始化完成")
    
    def collect_episode(self):
        """收集episode"""
        
        self.episode_count += 1
        print(f"\n🎮 Episode {self.episode_count} 開始...")
        
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
            
            # 獲取機器人動作（使用真正的控制系統）
            robot_action = self.keyboard_controller.get_robot_action(obs, info)
            
            # 確保動作維度正確
            if robot_action.shape[0] != self.env.action_space.shape[0]:
                print(f"⚠️ 動作維度不匹配: {robot_action.shape} vs {self.env.action_space.shape}")
                robot_action = np.zeros(self.env.action_space.shape[0])
            
            # 執行動作
            next_obs, reward, terminated, truncated, next_info = self.env.step(robot_action)
            next_obs_processed = self.preprocessor.modify_state(next_obs, next_info).squeeze()
            
            # 記錄數據 (轉換為歸一化動作供BC使用)
            normalized_action = 2 * (robot_action - self.env.action_space.low) / (self.env.action_space.high - self.env.action_space.low) - 1
            
            episode_data['actions'].append(normalized_action)
            episode_data['rewards'].append(reward)
            episode_data['episode_reward'] += reward
            episode_data['observations'].append(next_obs_processed)
            
            step += 1
            
            # 實時信息
            if step % 100 == 0:
                print(f"   Step {step}, 獎勵: {episode_data['episode_reward']:.3f}")
            
            if terminated or truncated:
                print(f"   Episode在第{step}步結束")
                break
            
            obs = next_obs
            info = next_info
            time.sleep(0.05)
        
        # Episode結束
        duration = time.time() - episode_data['start_time']
        reward = episode_data['episode_reward']
        
        print(f"\n📊 Episode {self.episode_count} 結果:")
        print(f"   獎勵: {reward:.3f}")
        print(f"   步數: {len(episode_data['actions'])}")
        print(f"   時長: {duration:.1f}秒")
        
        # 保存決定
        if reward > -10:
            save_decision = input("💾 保存這個episode嗎？[Y/n]: ").strip().lower()
            save_decision = save_decision in ['', 'y', 'yes']
        else:
            save_decision = input("😞 表現較差，仍要保存嗎？[y/N]: ").strip().lower() 
            save_decision = save_decision in ['y', 'yes']
        
        if save_decision:
            self._save_episode(episode_data)
        
        return episode_data if save_decision else None
    
    def _save_episode(self, episode_data):
        """保存episode"""
        
        trajectory = {
            'observations': np.array(episode_data['observations'][:-1]),
            'actions': np.array(episode_data['actions']),
            'rewards': np.array(episode_data['rewards']),
            'episode_reward': episode_data['episode_reward'],
            'collection_time': datetime.now().isoformat(),
            'episode_id': self.episode_count
        }
        
        self.all_trajectories.append(trajectory)
        
        # 保存
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"proper_expert_trajectories_{timestamp}.pkl"
        filepath = os.path.join(self.save_dir, filename)
        
        with open(filepath, 'wb') as f:
            pickle.dump(self.all_trajectories, f)
        
        # BC標準格式
        standard_path = os.path.join(self.save_dir, "expert_trajectories.pkl")
        with open(standard_path, 'wb') as f:
            pickle.dump(self.all_trajectories, f)
        
        print(f"✅ Episode已保存！總軌跡數: {len(self.all_trajectories)}")
    
    def run_collection(self):
        """運行收集"""
        
        print("🚀 開始正確的專家軌跡收集")
        print("="*60)
        
        self.keyboard_controller.start_keyboard_listener()
        
        try:
            while self.keyboard_controller.running:
                episode = self.collect_episode()
                
                if not self.keyboard_controller.running:
                    break
                
                continue_choice = input("\n🔄 收集下一個episode嗎？[Y/n]: ").strip().lower()
                if continue_choice in ['n', 'no']:
                    break
        
        except KeyboardInterrupt:
            print("\n⚠️ 收集中斷")
        
        finally:
            keyboard.unhook_all()
            self.env.close()
            
            if self.all_trajectories:
                rewards = [t['episode_reward'] for t in self.all_trajectories]
                print(f"\n📊 收集統計:")
                print(f"   總軌跡數: {len(self.all_trajectories)}")
                print(f"   平均獎勵: {np.mean(rewards):.3f}")
                print(f"   最佳獎勵: {max(rewards):.3f}")
                print("\n💡 現在可以運行 behavioral_cloning.py")


if __name__ == "__main__":
    print("🤖 正確的機器人控制專家軌跡收集器")
    print("💡 使用真正的機器人運動學控制系統")
    
    if not ROBOT_CONTROL_AVAILABLE:
        print("⚠️ 警告: 無法使用機器人控制系統，功能受限")
        choice = input("是否繼續？[y/N]: ").strip().lower()
        if choice not in ['y', 'yes']:
            exit(1)
    
    try:
        collector = ProperExpertCollector()
        collector.run_collection()
    except Exception as e:
        print(f"❌ 錯誤: {e}")
        import traceback
        traceback.print_exc()