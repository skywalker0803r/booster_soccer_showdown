# eval_and_submit.py

import torch
import numpy as np
import os
from sai_rl import SAIClient

# 從 main.py, ddpg_model.py, utils.py 匯入必要的類別和函數
from ddpg_model import DDPG_FF
from utils import Preprocessor

# =================================================================
# 1. 配置與輔助函數 (從 main.py 複製)
# =================================================================
MODEL_NAME = "Booster-DDPG-FF-v1" 
MODEL_PATH = f"best_{MODEL_NAME}.pth" # 預設載入最佳模型
N_FEATURES = 45 # Preprocessor 輸出的狀態維度
NEURONS = [256, 256] 
LEARNING_RATE = 3e-4 # DDPG_FF 初始化需要此參數

# 初始化 SAIClient
sai = SAIClient(
    comp_id="booster-soccer-showdown" , 
    api_key="sai_LFcuaCZiqEkUbNVolQ3wbk5yU7H11jfv",
)
# 創建一個環境實例以獲取動作空間維度
env = sai.make_env() 
N_ACTIONS = env.action_space.shape[0]

# 將原始 policy 輸出 [-1, 1] 映射到環境動作空間
def action_function(policy):
    expected_bounds = [-1, 1]
    action_percent = (policy - expected_bounds[0]) / (
        expected_bounds[1] - expected_bounds[0]
    )
    bounded_percent = np.minimum(np.maximum(action_percent, 0), 1)
    return (
        env.action_space.low
        + (env.action_space.high - env.action_space.low) * bounded_percent
    )

# =================================================================
# 2. 載入模型
# =================================================================

def load_ddpg_model(model_path):
    """載入 DDPG 模型權重"""
    if not os.path.exists(model_path):
        print(f"錯誤: 找不到模型檔案 {model_path}。請確認路徑或檔案名稱是否正確。")
        return None

    # 必須先初始化模型架構
    ddpg_agent = DDPG_FF(
        N_FEATURES, 
        env.action_space, 
        NEURONS, 
        torch.nn.functional.relu, 
        LEARNING_RATE
    )
    
    # 載入權重
    ddpg_agent.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    ddpg_agent.eval() # 設置為評估模式
    print(f"成功載入模型權重: {model_path}")
    
    # 關閉初始化的環境實例
    env.close() 
    return ddpg_agent


# =================================================================
# 3. 執行操作
# =================================================================

def main_flow():
    """主執行流程"""
    
    loaded_model = load_ddpg_model(MODEL_PATH)
    if loaded_model is None:
        return

    # --- 觀看模型表現 (Watch) ---
    print("\n--- 正在觀看模型表現 (sai.watch) ---")
    print("這將在您本地播放模型在環境中的運行情況...")
    try:
        sai.watch(
            model=loaded_model,
            action_function=action_function,
            preprocessor_class=Preprocessor,
        )
        print("觀看結束。")
    except Exception as e:
        print(f"執行 sai.watch 失敗: {e}")
    
    # --- 評估模型效能 (Benchmark) ---
    print("\n--- 正在評估模型效能 (sai.benchmark) ---")
    try:
        results = sai.benchmark(
            model=loaded_model.actor,
            action_function=action_function,
            preprocessor_class=Preprocessor,
        )
        print("\n=== Benchmark 結果 ===")
        print(results)
        print("========================")
    except Exception as e:
        print(f"執行 sai.benchmark 失敗: {e}")


    # --- 提交模型 (Submit) ---
    submit_prompt = input("\n您是否要提交模型到競賽中？(輸入 y 進行提交): ")
    
    if submit_prompt.lower() == 'y':
        submission_name = input("請輸入本次提交的名稱 (例如 'DDPG_Final_Tuning'): ")
        print(f"--- 正在提交模型: {submission_name} ---")
        try:
            submission = sai.submit(
                name=submission_name,
                model=loaded_model,
                action_function=action_function,
                preprocessor_class=Preprocessor,
            )
            print("\n=== 提交結果 ===")
            print(submission)
            print("=================")
        except Exception as e:
            print(f"執行 sai.submit 失敗: {e}")
    else:
        print("已取消模型提交。")

if __name__ == "__main__":
    main_flow()