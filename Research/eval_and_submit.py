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

# 動作轉換函數 (修正版)
def action_function(policy):
    """
    根據docs/Action Functions.md的規範：
    - 只能訪問numpy(np)和環境(env)
    - 不能訪問外部變量或模組
    - 預設連續動作會用tanh然後重新縮放
    """
    # 簡化版本：假設模型輸出已經在正確範圍內
    return policy

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
# 3. 主要執行流程
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
# 4. 輔助功能
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
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        quick_test()
    else:
        main_flow()