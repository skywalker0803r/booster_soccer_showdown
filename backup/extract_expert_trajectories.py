"""
Extract Expert Trajectories from Training Data
從你的歷史訓練中提取成功的軌跡作為專家演示
"""

import numpy as np
import pickle
import os
from pathlib import Path
import torch

def extract_successful_episodes():
    """從訓練歷史中提取成功軌跡"""
    
    print("🔍 尋找專家軌跡數據...")
    
    # 檢查可能的數據來源
    possible_sources = [
        "saved_models/training_logs/",
        "runs/",
        "training_data/",
        "episodes_data/",
        "./"  # 當前目錄
    ]
    
    expert_trajectories = []
    
    # 方法1: 從TensorBoard logs提取
    print("\n📊 方法1: 檢查TensorBoard日誌...")
    runs_dir = Path("runs")
    if runs_dir.exists():
        for run_dir in runs_dir.iterdir():
            if run_dir.is_dir():
                print(f"   Found run: {run_dir.name}")
                # TODO: 解析TensorBoard events文件
    
    # 方法2: 從checkpoint重現成功episode
    print("\n🎯 方法2: 從最佳checkpoint重現軌跡...")
    checkpoints = list(Path("saved_models/checkpoints/").glob("*.pth")) if Path("saved_models/checkpoints/").exists() else []
    best_models = list(Path("saved_models/best_models/").glob("*.pth")) if Path("saved_models/best_models/").exists() else []
    
    if checkpoints or best_models:
        print(f"   Found {len(checkpoints)} checkpoints, {len(best_models)} best models")
    
    # 方法3: 手動收集專家軌跡
    print("\n🎮 方法3: 收集新的專家軌跡...")
    print("   可以從以下來源收集:")
    print("   - 人類操控機器人")
    print("   - 使用成功的checkpoint")
    print("   - 從其他成功的訓練運行")
    
    return expert_trajectories

def collect_expert_trajectory_from_model(model_path, num_episodes=10):
    """從訓練好的模型收集專家軌跡"""
    
    print(f"🤖 從模型收集專家軌跡: {model_path}")
    
    from sai_rl import SAIClient
    from improved_dreamerv3 import ImprovedDreamerV3
    from sai_compatible_dreamerv3 import SAICompatibleDreamerV3
    
    # 加載模型
    try:
        model = ImprovedDreamerV3(obs_dim=89, action_dim=12)
        model.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=False))
        model.eval()
        print("✅ 模型加載成功")
    except Exception as e:
        print(f"❌ 模型加載失敗: {e}")
        return []
    
    # 初始化環境
    sai = SAIClient(comp_id="booster-soccer-showdown", api_key="sai_LFcuaCZiqEkUbNVolQ3wbk5yU7H11jfv")
    env = sai.make_env()
    
    from main_improved_dreamerv3 import Preprocessor
    preprocessor = Preprocessor()
    
    expert_trajectories = []
    successful_count = 0
    
    for episode in range(num_episodes):
        print(f"   收集軌跡 {episode+1}/{num_episodes}...")
        
        obs, info = env.reset()
        obs = preprocessor.modify_state(obs, info).squeeze()
        
        trajectory = {
            'observations': [],
            'actions': [],
            'rewards': [],
            'episode_reward': 0
        }
        
        agent_state = None
        
        for step in range(800):  # 最大步數
            # 記錄狀態
            trajectory['observations'].append(obs.copy())
            
            # 獲取動作
            action, agent_state = model.select_action(obs, agent_state, deterministic=True)
            trajectory['actions'].append(action.copy())
            
            # 執行動作
            env_action = env.action_space.low + (env.action_space.high - env.action_space.low) * (action + 1) / 2
            next_obs, reward, terminated, truncated, next_info = env.step(env_action)
            
            trajectory['rewards'].append(reward)
            trajectory['episode_reward'] += reward
            
            obs = preprocessor.modify_state(next_obs, next_info).squeeze()
            
            if terminated or truncated:
                break
        
        print(f"     Episode獎勵: {trajectory['episode_reward']:.3f}")
        
        # 只保留成功的軌跡
        if trajectory['episode_reward'] > -1.0:  # 比純step penalty好
            expert_trajectories.append(trajectory)
            successful_count += 1
            print(f"     ✅ 成功軌跡 #{successful_count}")
        
    print(f"🎯 收集到 {successful_count} 條成功軌跡")
    return expert_trajectories

def save_expert_trajectories(trajectories, filename="expert_trajectories.pkl"):
    """保存專家軌跡"""
    
    if not trajectories:
        print("❌ 沒有軌跡可保存")
        return
    
    os.makedirs("expert_data", exist_ok=True)
    filepath = f"expert_data/{filename}"
    
    with open(filepath, 'wb') as f:
        pickle.dump(trajectories, f)
    
    print(f"💾 已保存 {len(trajectories)} 條軌跡到 {filepath}")
    
    # 統計信息
    rewards = [traj['episode_reward'] for traj in trajectories]
    lengths = [len(traj['observations']) for traj in trajectories]
    
    print(f"📊 軌跡統計:")
    print(f"   平均獎勵: {np.mean(rewards):.3f}")
    print(f"   最佳獎勵: {max(rewards):.3f}")
    print(f"   平均長度: {np.mean(lengths):.1f} 步")
    print(f"   最長軌跡: {max(lengths)} 步")

if __name__ == "__main__":
    print("🎯 專家軌跡提取工具")
    print("="*50)
    
    # 1. 檢查現有數據
    extract_successful_episodes()
    
    # 2. 從最佳模型收集軌跡
    best_models = list(Path("saved_models/best_models/").glob("*.pth")) if Path("saved_models/best_models/").exists() else []
    
    if best_models:
        print(f"\n🚀 發現 {len(best_models)} 個最佳模型:")
        for model_path in best_models:
            print(f"   {model_path}")
        
        # 使用最新的模型
        latest_model = max(best_models, key=lambda p: p.stat().st_mtime)
        print(f"\n🎯 使用最新模型: {latest_model}")
        
        trajectories = collect_expert_trajectory_from_model(latest_model, num_episodes=20)
        
        if trajectories:
            save_expert_trajectories(trajectories)
        else:
            print("❌ 沒有收集到成功軌跡")
    else:
        print("\n⚠️ 沒有找到訓練好的模型")
        print("建議:")
        print("1. 先訓練一個基本可用的模型")
        print("2. 或手動收集專家軌跡")
        print("3. 或使用online imitation learning")