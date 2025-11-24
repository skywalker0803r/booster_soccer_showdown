# -*- coding: utf-8 -*-
# eval_and_submit_sb3.py
# 專為Stable Baselines3 PPO模型設計的評估和提交腳本

import torch
import numpy as np
import os
import glob
from sai_rl import SAIClient
from stable_baselines3 import PPO
from utils import Preprocessor

# =================================================================
# 1. Configuration
# =================================================================
# 自動尋找最新的SB3模型
def find_latest_sb3_model():
    """自動找到最新的SB3模型檔案"""
    patterns = [
        'BC-SB3-PPO_*.zip',
        'best_*.zip', 
        'final_*.zip',
        'checkpoint_*.zip'
    ]
    
    all_models = []
    for pattern in patterns:
        models = glob.glob(pattern)
        all_models.extend(models)
    
    if all_models:
        # 按修改時間排序，取最新的
        latest_model = max(all_models, key=os.path.getmtime)
        return latest_model
    else:
        return None

# 尋找模型檔案
MODEL_PATH = find_latest_sb3_model()
if MODEL_PATH:
    print(f"✅ 自動找到最新的SB3模型: {MODEL_PATH}")
else:
    MODEL_PATH = "BC-SB3-PPO_100000_steps.zip"  # 您的模型
    print(f"⚠️ 使用指定模型: {MODEL_PATH}")

# 初始化環境獲取動作空間信息
sai = SAIClient(
    comp_id="booster-soccer-showdown",
    api_key="sai_LFcuaCZiqEkUbNVolQ3wbk5yU7H11jfv",
)

# 動作轉換函數 (根據docs/About.md的動作空間定義)
def action_function(policy):
    """
    根據docs/About.md和Action Functions.md的規範：
    - 動作空間: Box(shape=(12,), low=[-45,-45,-30,-65,-24,-15,-45,-45,-30,-65,-24,-15], high=[45,45,30,65,24,15,45,45,30,65,24,15])
    - 12個關節的力矩控制（左腿6個關節 + 右腿6個關節）
    - 只能訪問numpy(np)和環境(env)
    """
    
    # 動作空間的上下界（從docs/About.md第55行）
    action_low = np.array([-45,-45,-30,-65,-24,-15,-45,-45,-30,-65,-24,-15], dtype=np.float32)
    action_high = np.array([45,45,30,65,24,15,45,45,30,65,24,15], dtype=np.float32)
    
    # 假設模型輸出在[-1, 1]範圍，縮放到實際動作空間
    # policy: [-1, 1] -> action_space: [action_low, action_high]
    scaled_actions = action_low + (policy + 1.0) * (action_high - action_low) / 2.0
    
    # 確保動作在有效範圍內
    clipped_actions = np.clip(scaled_actions, action_low, action_high)
    
    return clipped_actions

# =================================================================
# 2. SB3模型包裝器
# =================================================================
class SB3ModelWrapper(torch.nn.Module):
    """
    將SB3 PPO模型包裝成符合SAI評估工具期望的格式
    """
    def __init__(self, sb3_model):
        super().__init__()
        self.sb3_model = sb3_model
        self.preprocessor = Preprocessor()
        
        # 獲取環境信息
        self.env = sai.make_env()
        
    def forward(self, state):
        """
        符合SAI評估工具的forward接口
        輸入: state tensor [batch_size, state_dim]
        輸出: action tensor [batch_size, action_dim]
        """
        # 將tensor轉為numpy (SB3期望numpy輸入)
        if isinstance(state, torch.Tensor):
            state_np = state.detach().cpu().numpy()
        else:
            state_np = state
        
        # 處理批次維度
        if state_np.ndim == 1:
            state_np = state_np.reshape(1, -1)
            single_sample = True
        else:
            single_sample = False
        
        # 使用SB3模型預測
        actions, _ = self.sb3_model.predict(state_np, deterministic=True)
        
        # 處理返回維度
        if single_sample and actions.ndim > 1:
            actions = actions.squeeze(0)
        
        # 轉回tensor格式 (如果原本是tensor)
        if isinstance(state, torch.Tensor):
            actions = torch.tensor(actions, dtype=state.dtype, device=state.device)
        
        return actions
    
    def __del__(self):
        """清理環境資源"""
        if hasattr(self, 'env'):
            self.env.close()

def load_sb3_model(model_path):
    """載入SB3模型"""
    if not os.path.exists(model_path):
        print(f"❌ 錯誤: 找不到模型檔案 '{model_path}'")
        print("📁 當前目錄中的.zip檔案:")
        for f in glob.glob("*.zip"):
            print(f"   - {f}")
        return None
    
    try:
        print(f"📂 載入SB3模型: {model_path}")
        
        # 載入SB3模型 (不需要環境，稍後會設置)
        sb3_model = PPO.load(model_path)
        print(f"✅ 成功載入SB3模型")
        
        # 設置為評估模式
        sb3_model.policy.set_training_mode(False)
        
        # 包裝模型
        wrapped_model = SB3ModelWrapper(sb3_model)
        
        return wrapped_model
        
    except Exception as e:
        print(f"❌ 載入模型時發生錯誤: {e}")
        print(f"💡 提示: 確保 {model_path} 是有效的SB3模型檔案")
        return None

# =================================================================
# 3. 詳細評估函數
# =================================================================
def evaluate_model_detailed(model, num_episodes=10):
    """
    執行詳細的模型評估，收集episode統計數據
    類似訓練時的ep_length_mean和ep_reward_mean
    """
    print(f"🔍 開始詳細評估 ({num_episodes} episodes)")
    
    # 創建評估環境
    eval_env = sai.make_env()
    
    episode_rewards = []
    episode_lengths = []
    success_count = 0
    
    try:
        for episode in range(num_episodes):
            obs, info = eval_env.reset()
            episode_reward = 0
            episode_length = 0
            done = False
            
            print(f"  Episode {episode + 1}/{num_episodes}", end=" ")
            
            while not done:
                # 使用模型預測動作
                with torch.no_grad():
                    if hasattr(model, 'forward'):
                        # 使用包裝器的forward方法
                        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
                        action_tensor = model.forward(obs_tensor)
                        action = action_tensor.squeeze(0).cpu().numpy()
                    else:
                        # 直接使用SB3模型
                        action, _ = model.predict(obs, deterministic=True)
                
                # 應用動作函數轉換
                final_action = action_function(action)
                
                # 執行動作
                obs, reward, terminated, truncated, info = eval_env.step(final_action)
                done = terminated or truncated
                
                episode_reward += reward
                episode_length += 1
                
                # 防止無限長的episode
                if episode_length > 1000:
                    break
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            
            # 判斷成功 (這裡可以根據具體任務調整成功條件)
            if episode_reward > 0:  # 簡單的成功標準
                success_count += 1
                print(f"✅ 獎勵: {episode_reward:.3f}, 長度: {episode_length}")
            else:
                print(f"❌ 獎勵: {episode_reward:.3f}, 長度: {episode_length}")
    
    finally:
        eval_env.close()
    
    # 計算統計數據
    if episode_rewards:
        stats = {
            'ep_reward_mean': np.mean(episode_rewards),
            'ep_reward_std': np.std(episode_rewards),
            'ep_length_mean': np.mean(episode_lengths),
            'ep_length_std': np.std(episode_lengths),
            'total_episodes': len(episode_rewards),
            'success_rate': success_count / len(episode_rewards),
            'min_reward': np.min(episode_rewards),
            'max_reward': np.max(episode_rewards),
            'min_length': np.min(episode_lengths),
            'max_length': np.max(episode_lengths)
        }
        return stats
    else:
        return None

# =================================================================
# 4. 主要執行流程
# =================================================================
def main_flow():
    """主要執行流程"""
    
    # 載入SB3模型
    print("🔧 載入SB3模型...")
    loaded_model = load_sb3_model(MODEL_PATH)
    if loaded_model is None:
        return
    
    print(f"✅ 模型載入成功！")
    
    # --- 觀看模型表現 (Watch) ---
    print("\n" + "="*50)
    print("👁️ 觀看模型表現 (sai.watch)")
    print("="*50)
    print("💡 提示: 在控制台按 Ctrl+C 停止觀看")
    
    try:
        sai.watch(
            model=loaded_model,
            action_function=action_function,
            preprocessor_class=Preprocessor,
        )
        print("觀看結束")
    except KeyboardInterrupt:
        print("\n⏹️ 觀看被用戶中斷")
    except Exception as e:
        print(f"❌ sai.watch 執行失敗: {e}")
    
    # --- 評估模型性能 (Benchmark) ---
    print("\n" + "="*50)
    print("📊 評估模型性能 (sai.benchmark)")
    print("="*50)
    
    try:
        results = sai.benchmark(
            model=loaded_model,
            action_function=action_function,
            preprocessor_class=Preprocessor,
        )
        print("\n🏆 === 基準測試結果 ===")
        print(results)
        print("=" * 30)
    except Exception as e:
        print(f"❌ sai.benchmark 執行失敗: {e}")
    
    # --- 詳細評估 (收集episode統計) ---
    print("\n" + "="*50)
    print("📈 詳細episode統計分析")
    print("="*50)
    
    try:
        episode_stats = evaluate_model_detailed(loaded_model, num_episodes=10)
        if episode_stats:
            print("\n📊 === Episode統計結果 ===")
            print(f"ep_length_mean: {episode_stats['ep_length_mean']:.2f}")
            print(f"ep_reward_mean: {episode_stats['ep_reward_mean']:.4f}")
            print(f"ep_length_std: {episode_stats['ep_length_std']:.2f}")
            print(f"ep_reward_std: {episode_stats['ep_reward_std']:.4f}")
            print(f"total_episodes: {episode_stats['total_episodes']}")
            print(f"success_rate: {episode_stats['success_rate']:.2%}")
            print("=" * 30)
    except Exception as e:
        print(f"❌ 詳細評估執行失敗: {e}")
    
    # --- 提交模型 (Submit) ---
    print("\n" + "="*50)
    print("🚀 模型提交")
    print("="*50)
    
    submit_prompt = input("是否要將此模型提交到比賽？ (y/n): ").strip().lower()
    
    if submit_prompt in ['y', 'yes', '是']:
        submission_name = input("請輸入提交名稱 (例如: 'BC_SB3_PPO_100k'): ").strip()
        if not submission_name:
            submission_name = f"BC_SB3_PPO_{os.path.basename(MODEL_PATH).replace('.zip', '')}"
        
        print(f"🚀 正在提交模型: {submission_name}")
        try:
            submission = sai.submit(
                name=submission_name,
                model=loaded_model,
                action_function=action_function,
                preprocessor_class=Preprocessor,
            )
            print("\n🎉 === 提交結果 ===")
            print(submission)
            print("=" * 20)
        except Exception as e:
            print(f"❌ sai.submit 執行失敗: {e}")
    else:
        print("❌ 取消模型提交")

# =================================================================
# 5. 輔助功能
# =================================================================
def quick_test():
    """快速測試模型載入和基本功能"""
    print("🧪 快速測試模式")
    
    model = load_sb3_model(MODEL_PATH)
    if model is None:
        return
        
    # 測試forward方法
    try:
        test_input = torch.randn(1, 45)  # 假設45維狀態
        output = model.forward(test_input)
        print(f"✅ Forward測試成功:")
        print(f"   輸入形狀: {test_input.shape}")
        print(f"   輸出形狀: {output.shape}")
        print(f"   輸出範圍: [{output.min():.3f}, {output.max():.3f}]")
    except Exception as e:
        print(f"❌ Forward測試失敗: {e}")

if __name__ == "__main__":
    import sys
    
    # 檢查命令行參數
    if len(sys.argv) > 1:
        if sys.argv[1] == "test":
            quick_test()
        elif sys.argv[1] == "eval":
            # 允許指定評估episode數量
            num_eps = int(sys.argv[2]) if len(sys.argv) > 2 else 10
            print(f"🎯 評估模式: {num_eps} episodes")
            main_flow()
        else:
            main_flow()
    else:
        main_flow()