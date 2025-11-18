"""
轉換專家數據格式
將手動收集的軌跡轉換為BC/DAgger可用的格式
"""

import pickle
import numpy as np
import os
from pathlib import Path
from sai_rl import SAIClient


class ExpertDataConverter:
    """專家數據格式轉換器"""
    
    def __init__(self):
        self.sai = SAIClient(comp_id="booster-soccer-showdown", api_key="sai_LFcuaCZiqEkUbNVolQ3wbk5yU7H11jfv")
        self.env = self.sai.make_env()
        
        # 導入preprocessor
        import sys
        sys.path.append('training_scripts')
        from main_improved_dreamerv3 import Preprocessor
        self.preprocessor = Preprocessor()
        
        print("🔄 專家數據轉換器初始化完成")
    
    def load_expert_trajectories(self, filepath):
        """載入專家軌跡"""
        with open(filepath, 'rb') as f:
            trajectories = pickle.load(f)
        
        print(f"✅ 載入 {len(trajectories)} 條軌跡 from {filepath}")
        return trajectories
    
    def convert_trajectory(self, traj):
        """轉換單條軌跡格式"""
        
        # 原始數據
        observations = traj['observations']  # 原始環境obs
        actions = traj['actions']           # 關節控制指令
        rewards = traj['rewards']
        episode_reward = traj['episode_reward']
        
        # 轉換觀察值格式
        converted_observations = []
        
        print(f"🔄 轉換軌跡 (原始長度: {len(observations)})...")
        
        for i, raw_obs in enumerate(observations):
            try:
                # 模擬環境重置以獲得正確的info格式
                if i == 0:
                    _, info = self.env.reset()
                    # 這裡需要從raw_obs中提取info，這是個挑戰
                    # 簡化處理：使用reset的info作為模板
                    mock_info = info
                else:
                    mock_info = info  # 重用
                
                # 使用preprocessor轉換
                processed_obs = self.preprocessor.modify_state(raw_obs, mock_info)
                converted_observations.append(processed_obs.squeeze())
                
            except Exception as e:
                print(f"⚠️ 觀察值轉換失敗 at step {i}: {e}")
                continue
        
        # 轉換動作格式 (從關節控制到歸一化動作)
        converted_actions = []
        
        for action in actions:
            try:
                # 假設action已經是關節空間的控制指令
                # 需要反向轉換到 [-1, 1] 的歸一化空間
                normalized_action = self._convert_joint_to_normalized(action)
                converted_actions.append(normalized_action)
                
            except Exception as e:
                print(f"⚠️ 動作轉換失敗: {e}")
                continue
        
        # 確保長度一致
        min_length = min(len(converted_observations), len(converted_actions))
        if min_length == 0:
            return None
        
        converted_traj = {
            'observations': converted_observations[:min_length],
            'actions': converted_actions[:min_length],
            'rewards': rewards[:min_length],
            'episode_reward': episode_reward,
            'quality': traj.get('quality', 'unknown'),
            'original_length': len(observations),
            'converted_length': min_length
        }
        
        print(f"✅ 轉換完成: {len(observations)} → {min_length} steps")
        return converted_traj
    
    def _convert_joint_to_normalized(self, joint_action):
        """將關節控制轉換為歸一化動作"""
        # 這是一個簡化版本，可能需要根據實際情況調整
        
        # 假設joint_action的範圍和env.action_space的範圍相同
        action_low = self.env.action_space.low
        action_high = self.env.action_space.high
        
        # 歸一化到 [-1, 1]
        normalized = 2 * (joint_action - action_low) / (action_high - action_low) - 1
        normalized = np.clip(normalized, -1, 1)
        
        return normalized
    
    def convert_all_trajectories(self, input_filepath, output_filepath=None):
        """轉換所有軌跡"""
        
        # 載入原始數據
        original_trajectories = self.load_expert_trajectories(input_filepath)
        
        # 轉換
        converted_trajectories = []
        
        for i, traj in enumerate(original_trajectories):
            print(f"\n📝 轉換軌跡 {i+1}/{len(original_trajectories)}...")
            converted = self.convert_trajectory(traj)
            
            if converted is not None:
                converted_trajectories.append(converted)
                print(f"✅ 軌跡 {i+1} 轉換成功")
            else:
                print(f"❌ 軌跡 {i+1} 轉換失敗")
        
        # 保存轉換後的數據
        if output_filepath is None:
            output_filepath = input_filepath.replace('.pkl', '_converted.pkl')
        
        os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
        
        with open(output_filepath, 'wb') as f:
            pickle.dump(converted_trajectories, f)
        
        print(f"\n💾 已保存 {len(converted_trajectories)} 條轉換後的軌跡")
        print(f"📁 輸出文件: {output_filepath}")
        
        # 統計信息
        self.print_conversion_stats(original_trajectories, converted_trajectories)
        
        return converted_trajectories
    
    def print_conversion_stats(self, original, converted):
        """打印轉換統計"""
        
        print(f"\n📊 轉換統計:")
        print(f"   原始軌跡數: {len(original)}")
        print(f"   轉換軌跡數: {len(converted)}")
        print(f"   成功率: {len(converted)/len(original)*100:.1f}%")
        
        if converted:
            rewards = [t['episode_reward'] for t in converted]
            lengths = [t['converted_length'] for t in converted]
            qualities = [t['quality'] for t in converted]
            
            print(f"   平均獎勵: {np.mean(rewards):.3f}")
            print(f"   平均長度: {np.mean(lengths):.1f}")
            print(f"   質量分布: {dict(zip(*np.unique(qualities, return_counts=True)))}")


def find_latest_expert_data():
    """找到最新的專家數據文件"""
    
    expert_dir = Path("expert_data")
    if not expert_dir.exists():
        return None
    
    pkl_files = list(expert_dir.glob("expert_trajectories_*.pkl"))
    if not pkl_files:
        return None
    
    # 找到最新的文件
    latest_file = max(pkl_files, key=lambda p: p.stat().st_mtime)
    return str(latest_file)


if __name__ == "__main__":
    print("🔄 專家數據格式轉換器")
    print("="*50)
    
    # 查找最新的專家數據
    latest_file = find_latest_expert_data()
    
    if latest_file is None:
        print("❌ 沒有找到專家軌跡數據！")
        print("請先運行 expert_data_collector.py 收集專家軌跡")
        exit(1)
    
    print(f"📁 找到專家數據: {latest_file}")
    
    # 創建轉換器
    converter = ExpertDataConverter()
    
    # 轉換數據
    output_file = "expert_data/expert_trajectories.pkl"  # BC期望的格式
    converted_trajectories = converter.convert_all_trajectories(latest_file, output_file)
    
    print(f"\n🎉 轉換完成！")
    print(f"💡 現在可以運行 behavioral_cloning.py 開始訓練BC模型")