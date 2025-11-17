"""
專家軌跡收集器 - 整合手動遙控和數據記錄
基於 booster_control/teleoperate.py 修改，添加軌跡記錄功能
"""

import argparse
import numpy as np
import pickle
import os
from datetime import datetime
from pathlib import Path
import sai_mujoco  # noqa: F401
import gymnasium as gym
import sys

# 添加 booster_control 到路徑
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
booster_control_path = os.path.join(os.path.dirname(current_dir), 'booster_control')
sys.path.insert(0, booster_control_path)

try:
    from se3_keyboard import Se3Keyboard, Se3Keyboard_Pynput
    from t1_utils import LowerT1JoyStick
    print(f"✅ 成功導入 booster_control 模組")
except ImportError as e:
    print(f"❌ 導入模組失敗: {e}")
    print(f"🔍 嘗試的路徑: {booster_control_path}")
    print(f"📁 當前目錄內容: {os.listdir('.')}")
    if os.path.exists('booster_control'):
        print(f"📁 booster_control 內容: {os.listdir('booster_control')}")
    raise


class ExpertTrajectoryCollector:
    """專家軌跡收集器"""
    
    def __init__(self, save_dir="expert_data"):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # 當前episode的數據
        self.current_episode = {
            'observations': [],
            'actions': [],
            'rewards': [],
            'info_history': [],
            'episode_reward': 0
        }
        
        # 所有收集的軌跡
        self.all_trajectories = []
        self.episode_count = 0
        
        print(f"🎯 專家軌跡收集器初始化")
        print(f"📁 數據保存目錄: {save_dir}")
    
    def start_episode(self, observation, info):
        """開始新的episode"""
        self.current_episode = {
            'observations': [observation.copy()],
            'actions': [],
            'rewards': [],
            'info_history': [info.copy()],
            'episode_reward': 0,
            'start_time': datetime.now()
        }
        self.episode_count += 1
        print(f"\n🎮 Episode {self.episode_count} 開始記錄...")
    
    def record_step(self, observation, action, reward, info):
        """記錄單步數據"""
        self.current_episode['observations'].append(observation.copy())
        self.current_episode['actions'].append(action.copy())
        self.current_episode['rewards'].append(reward)
        self.current_episode['info_history'].append(info.copy())
        self.current_episode['episode_reward'] += reward
    
    def end_episode(self, success=False):
        """結束episode並決定是否保存"""
        duration = datetime.now() - self.current_episode['start_time']
        reward = self.current_episode['episode_reward']
        steps = len(self.current_episode['actions'])
        
        print(f"📊 Episode {self.episode_count} 完成:")
        print(f"   獎勵: {reward:.3f}")
        print(f"   步數: {steps}")
        print(f"   時長: {duration.total_seconds():.1f}秒")
        print(f"   成功: {'✅' if success else '❌'}")
        
        # 詢問是否保存這個episode
        if success or reward > -10:  # 自動保存成功或還不錯的episode
            save_choice = input(f"💾 保存這個episode嗎？[Y/n]: ").strip().lower()
            if save_choice in ['', 'y', 'yes']:
                self._save_episode(quality="good")
        elif reward > -50:  # 中等表現
            save_choice = input(f"📝 這個episode表現一般，保存嗎？[y/N]: ").strip().lower()
            if save_choice in ['y', 'yes']:
                self._save_episode(quality="medium")
        else:  # 表現很差
            save_choice = input(f"🗑️  這個episode表現較差，仍要保存嗎？[y/N]: ").strip().lower()
            if save_choice in ['y', 'yes']:
                self._save_episode(quality="poor")
        
        print(f"📈 目前已保存 {len(self.all_trajectories)} 條軌跡")
    
    def _save_episode(self, quality="good"):
        """保存episode"""
        # 添加質量標籤
        self.current_episode['quality'] = quality
        self.current_episode['collection_time'] = datetime.now().isoformat()
        
        # 轉換為numpy arrays
        episode_data = {
            'observations': np.array(self.current_episode['observations'][:-1]),  # 移除最後一個obs
            'actions': np.array(self.current_episode['actions']),
            'rewards': np.array(self.current_episode['rewards']),
            'episode_reward': self.current_episode['episode_reward'],
            'quality': quality,
            'collection_time': self.current_episode['collection_time'],
            'episode_id': self.episode_count
        }
        
        self.all_trajectories.append(episode_data)
        print(f"✅ Episode {self.episode_count} 已保存 ({quality} quality)")
        
        # 自動保存到文件
        self.save_to_file()
    
    def save_to_file(self, filename=None):
        """保存所有軌跡到文件"""
        if not self.all_trajectories:
            return
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"expert_trajectories_{timestamp}.pkl"
        
        filepath = os.path.join(self.save_dir, filename)
        
        with open(filepath, 'wb') as f:
            pickle.dump(self.all_trajectories, f)
        
        print(f"💾 已保存 {len(self.all_trajectories)} 條軌跡到 {filepath}")
    
    def print_statistics(self):
        """打印收集統計"""
        if not self.all_trajectories:
            print("📊 尚未收集任何軌跡")
            return
        
        rewards = [traj['episode_reward'] for traj in self.all_trajectories]
        lengths = [len(traj['actions']) for traj in self.all_trajectories]
        qualities = [traj['quality'] for traj in self.all_trajectories]
        
        print(f"\n📊 收集統計:")
        print(f"   總軌跡數: {len(self.all_trajectories)}")
        print(f"   平均獎勵: {np.mean(rewards):.3f}")
        print(f"   最佳獎勵: {max(rewards):.3f}")
        print(f"   最差獎勵: {min(rewards):.3f}")
        print(f"   平均長度: {np.mean(lengths):.1f} 步")
        print(f"   質量分布: Good={qualities.count('good')}, Medium={qualities.count('medium')}, Poor={qualities.count('poor')}")


def expert_teleoperate(
    env_name: str = "LowerT1GoaliePenaltyKick-v0",
    pos_sensitivity: float = 0.1,
    rot_sensitivity: float = 1.5,
    renderer="mjviewer"
):
    """專家遙控with軌跡記錄"""
    
    print("🎯 專家軌跡收集模式")
    print("="*60)
    print("🎮 你將手動操作機器人踢足球")
    print("📝 系統會自動記錄你的操作作為專家演示")
    print("💡 建議:")
    print("   - 嘗試讓機器人保持穩定")
    print("   - 慢慢接近球")
    print("   - 如果成功請保存軌跡")
    print("   - ESC鍵退出")
    print("="*60)
    
    # 創建環境
    env = gym.make(env_name, render_mode="human", renderer=renderer)
    lower_t1_robot = LowerT1JoyStick(env.unwrapped)
    
    # 創建軌跡收集器
    collector = ExpertTrajectoryCollector()
    
    # 初始化鍵盤控制器
    if renderer == "mjviewer":
        keyboard_controller = Se3Keyboard_Pynput(
            renderer=env.unwrapped.mujoco_renderer,
            pos_sensitivity=pos_sensitivity,
            rot_sensitivity=rot_sensitivity,
        )
    else:
        keyboard_controller = Se3Keyboard(
            renderer=env.unwrapped.mujoco_renderer,
            pos_sensitivity=pos_sensitivity,
            rot_sensitivity=rot_sensitivity,
        )
    
    # 設置重置回調
    keyboard_controller.set_reset_env_callback(env.reset)
    
    # 打印控制說明
    print("\n🎮 鍵盤控制:")
    print(keyboard_controller)
    print("\n📝 數據收集:")
    print("   - 每個episode結束後會詢問是否保存")
    print("   - 成功的episode會自動提示保存")
    print("   - 按 Ctrl+C 查看統計信息")
    
    try:
        # 主要遙控循環
        while True:
            # 重置環境
            observation, info = env.reset()
            collector.start_episode(observation, info)
            
            # Episode循環
            terminated = truncated = False
            while not (terminated or truncated):
                # 檢查退出
                if keyboard_controller.should_quit():
                    print("\n[INFO] ESC pressed — 退出遙控")
                    collector.print_statistics()
                    env.close()
                    return
                
                # 獲取鍵盤輸入
                command = keyboard_controller.advance()
                ctrl, _ = lower_t1_robot.get_actions(command, observation, info)
                
                # 執行動作
                next_observation, reward, terminated, truncated, next_info = env.step(ctrl)
                
                # 記錄數據
                collector.record_step(next_observation, ctrl, reward, next_info)
                
                observation = next_observation
                info = next_info
            
            # Episode結束
            success = info.get("success", False)
            collector.end_episode(success)
            
            # 詢問是否繼續
            continue_choice = input("\n🔄 繼續下一個episode嗎？[Y/n]: ").strip().lower()
            if continue_choice in ['n', 'no']:
                collector.print_statistics()
                break
    
    except KeyboardInterrupt:
        print("\n\n⚠️ 收到中斷信號...")
        collector.print_statistics()
        
        # 詢問是否保存
        if collector.all_trajectories:
            save_choice = input("💾 保存已收集的軌跡嗎？[Y/n]: ").strip().lower()
            if save_choice in ['', 'y', 'yes']:
                collector.save_to_file()
    
    finally:
        env.close()
        print("🎉 專家軌跡收集完成！")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("收集專家軌跡 - 手動遙控機器人")
    parser.add_argument("--env", type=str, default="LowerT1GoaliePenaltyKick-v0", help="環境名稱")
    parser.add_argument("--pos_sensitivity", type=float, default=0.1, help="位置敏感度")
    parser.add_argument("--rot_sensitivity", type=float, default=0.5, help="旋轉敏感度")
    parser.add_argument("--renderer", type=str, default="mujoco", help="渲染器")
    
    args = parser.parse_args()
    
    expert_teleoperate(args.env, args.pos_sensitivity, args.rot_sensitivity, args.renderer)