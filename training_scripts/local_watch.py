"""
本地模型總結 - TensorBoard 監控 + 觀看 + 提交
從 Colab 下載模型檔案後，在本地完成：
1. TensorBoard 監控訓練過程
2. 觀看模型實際表現
3. 決定是否提交到排行榜
"""

from sai_rl import SAIClient
from stable_baselines3 import PPO
import numpy as np
import os
import subprocess
import threading
import time
import webbrowser

# 你的模型檔案路徑 (需要修改為實際下載的模型路徑)
MODEL_PATH = "./saved_models/simple_ppo_20251117_062217.zip"  # 修改這裡！

class Preprocessor():
    def get_task_onehot(self, info):
        if 'task_index' in info:
            return info['task_index']
        else:
            return np.array([])

    def quat_rotate_inverse(self, q: np.ndarray, v: np.ndarray):
        q_w = q[:,[-1]]
        q_vec = q[:,:3]
        a = v * (2.0 * q_w**2 - 1.0)
        b = np.cross(q_vec, v) * (q_w * 2.0)
        c = q_vec * (np.dot(q_vec, v).reshape(-1,1) * 2.0)    
        return a - b + c 

    def modify_state(self, obs, info):
        if len(obs.shape) == 1:
            obs = np.expand_dims(obs, axis=0)

        task_onehot = self.get_task_onehot(info)
        if len(task_onehot.shape) == 1:
            task_onehot = np.expand_dims(task_onehot, axis=0)
        
        if len(info["robot_quat"].shape) == 1:
            info["robot_quat"] = np.expand_dims(info["robot_quat"], axis = 0)
            info["robot_gyro"] = np.expand_dims(info["robot_gyro"], axis = 0)
            info["robot_accelerometer"] = np.expand_dims(info["robot_accelerometer"], axis = 0)
            info["robot_velocimeter"] = np.expand_dims(info["robot_velocimeter"], axis = 0)
            info["goal_team_0_rel_robot"] = np.expand_dims(info["goal_team_0_rel_robot"], axis = 0)
            info["goal_team_1_rel_robot"] = np.expand_dims(info["goal_team_1_rel_robot"], axis = 0)
            info["goal_team_0_rel_ball"] = np.expand_dims(info["goal_team_0_rel_ball"], axis = 0)
            info["goal_team_1_rel_ball"] = np.expand_dims(info["goal_team_1_rel_ball"], axis = 0)
            info["ball_xpos_rel_robot"] = np.expand_dims(info["ball_xpos_rel_robot"], axis = 0) 
            info["ball_velp_rel_robot"] = np.expand_dims(info["ball_velp_rel_robot"], axis = 0) 
            info["ball_velr_rel_robot"] = np.expand_dims(info["ball_velr_rel_robot"], axis = 0) 
            info["player_team"] = np.expand_dims(info["player_team"], axis = 0)
            info["goalkeeper_team_0_xpos_rel_robot"] = np.expand_dims(info["goalkeeper_team_0_xpos_rel_robot"], axis = 0)
            info["goalkeeper_team_0_velp_rel_robot"] = np.expand_dims(info["goalkeeper_team_0_velp_rel_robot"], axis = 0)
            info["goalkeeper_team_1_xpos_rel_robot"] = np.expand_dims(info["goalkeeper_team_1_xpos_rel_robot"], axis = 0)
            info["goalkeeper_team_1_velp_rel_robot"] = np.expand_dims(info["goalkeeper_team_1_velp_rel_robot"], axis = 0)
            info["target_xpos_rel_robot"] = np.expand_dims(info["target_xpos_rel_robot"], axis = 0)
            info["target_velp_rel_robot"] = np.expand_dims(info["target_velp_rel_robot"], axis = 0)
            info["defender_xpos"] = np.expand_dims(info["defender_xpos"], axis = 0)
        
        robot_qpos = obs[:,:12]
        robot_qvel = obs[:,12:24]
        quat = info["robot_quat"]
        base_ang_vel = info["robot_gyro"]
        project_gravity = self.quat_rotate_inverse(quat, np.array([0.0, 0.0, -1.0]))
        
        obs = np.hstack((robot_qpos, 
                         robot_qvel,
                         project_gravity,
                         base_ang_vel,
                         info["robot_accelerometer"],
                         info["robot_velocimeter"],
                         info["goal_team_0_rel_robot"], 
                         info["goal_team_1_rel_robot"], 
                         info["goal_team_0_rel_ball"], 
                         info["goal_team_1_rel_ball"], 
                         info["ball_xpos_rel_robot"], 
                         info["ball_velp_rel_robot"], 
                         info["ball_velr_rel_robot"], 
                         info["player_team"], 
                         info["goalkeeper_team_0_xpos_rel_robot"], 
                         info["goalkeeper_team_0_velp_rel_robot"], 
                         info["goalkeeper_team_1_xpos_rel_robot"], 
                         info["goalkeeper_team_1_velp_rel_robot"], 
                         info["target_xpos_rel_robot"], 
                         info["target_velp_rel_robot"], 
                         info["defender_xpos"],
                         task_onehot))

        return obs

def action_function(policy):
    """動作函數，將策略輸出轉換為環境動作"""
    expected_bounds = [-1, 1]
    action_percent = (policy - expected_bounds[0]) / (
        expected_bounds[1] - expected_bounds[0]
    )
    bounded_percent = np.minimum(np.maximum(action_percent, 0), 1)
    return (
        env.action_space.low
        + (env.action_space.high - env.action_space.low) * bounded_percent
    )

def start_tensorboard():
    """啟動 TensorBoard 並開啟瀏覽器"""
    if not os.path.exists("./runs"):
        print("⚠️  找不到 runs/ 資料夾，跳過 TensorBoard")
        return None
    
    try:
        print("🚀 啟動 TensorBoard...")
        # 啟動 TensorBoard 在背景執行
        proc = subprocess.Popen(
            ["tensorboard", "--logdir=./runs", "--port=6006"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        # 等待一下讓 TensorBoard 啟動
        time.sleep(3)
        
        # 開啟瀏覽器
        print("🌐 開啟瀏覽器: http://localhost:6006")
        webbrowser.open("http://localhost:6006")
        
        return proc
        
    except FileNotFoundError:
        print("❌ 找不到 tensorboard 指令，請先安裝: pip install tensorboard")
        return None
    except Exception as e:
        print(f"⚠️  TensorBoard 啟動失敗: {e}")
        return None

def ask_user_submission(model, sai):
    """詢問用戶是否要提交模型"""
    while True:
        print("\n" + "="*50)
        print("🤔 觀看完模型表現，你想提交到排行榜嗎？")
        print("   y - 是的，提交模型")
        print("   n - 不提交")
        print("   r - 重新觀看一次")
        
        choice = input("請選擇 (y/n/r): ").lower().strip()
        
        if choice == 'y':
            print("\n🏆 正在提交模型到排行榜...")
            try:
                # 詢問提交的模型名稱
                model_name = input("請輸入模型名稱 (預設: My PPO Model): ").strip()
                if not model_name:
                    model_name = "My PPO Model"
                
                sai.submit(model_name, model, action_function, Preprocessor)
                print("✅ 模型提交成功！")
                return True
            except Exception as e:
                print(f"❌ 提交失敗: {e}")
                return False
                
        elif choice == 'n':
            print("👍 好的，不提交模型")
            return False
            
        elif choice == 'r':
            print("\n🎬 重新開始觀看...")
            try:
                sai.watch(model, action_function, Preprocessor)
            except KeyboardInterrupt:
                print("\n⏹️  觀看已停止")
            continue
            
        else:
            print("❌ 無效選擇，請輸入 y、n 或 r")

def main():
    global env
    
    print("📊 本地模型總結 - TensorBoard + 觀看 + 提交")
    print("=" * 60)
    
    # 檢查模型檔案是否存在
    if not os.path.exists(MODEL_PATH):
        print(f"❌ 找不到模型檔案: {MODEL_PATH}")
        print("\n📝 請執行以下步驟:")
        print("1. 從 Colab 下載 saved_models/ 資料夾")
        print("2. 修改此腳本中的 MODEL_PATH 變數")
        print("3. 確保模型檔案路徑正確")
        print(f"\n💡 範例檔案名稱: simple_ppo_20241117_123456.zip")
        
        # 顯示當前目錄下的模型檔案
        if os.path.exists("./saved_models"):
            print(f"\n📁 找到的模型檔案:")
            for file in os.listdir("./saved_models"):
                if file.endswith(".zip"):
                    print(f"   - {os.path.join('./saved_models', file)}")
        return
    
    # 啟動 TensorBoard
    tensorboard_proc = start_tensorboard()
    
    try:
        # 初始化 SAI 客戶端
        sai = SAIClient(comp_id="booster-soccer-showdown", api_key="sai_LFcuaCZiqEkUbNVolQ3wbk5yU7H11jfv")
        print("✅ SAI 客戶端初始化成功")
        
        # 創建環境
        env = sai.make_env(use_custom_eval=False)
        print("✅ 環境創建成功")
        
        # 載入訓練好的模型
        print(f"📥 載入模型: {MODEL_PATH}")
        model = PPO.load(MODEL_PATH)
        print("✅ 模型載入成功")
        
        print("\n" + "="*60)
        print("🎯 現在你可以：")
        print("1. 📊 查看 TensorBoard (已自動開啟瀏覽器)")
        print("2. 🎬 觀看模型實際表現")
        print("3. 🏆 決定是否提交到排行榜")
        print("="*60)
        
        # 詢問是否要開始觀看
        input("\n按 Enter 開始觀看模型表現...")
        
        # 開始觀看模型
        print("\n🎬 開始觀看模型表現...")
        print("   按 Ctrl+C 可以停止觀看")
        
        try:
            sai.watch(model, action_function, Preprocessor)
        except KeyboardInterrupt:
            print("\n⏹️  觀看已停止")
        
        # 詢問是否提交
        ask_user_submission(model, sai)
        
    except Exception as e:
        print(f"\n❌ 發生錯誤: {e}")
        print("\n🛠️  可能的解決方案:")
        print("1. 確保已安裝所有必要的依賴")
        print("2. 檢查網路連接")
        print("3. 確認 API 金鑰是否正確")
        print("4. 確認模型檔案是否完整")
    
    finally:
        # 關閉 TensorBoard
        if tensorboard_proc:
            print("\n🔄 關閉 TensorBoard...")
            tensorboard_proc.terminate()
        
        # 關閉環境
        if 'env' in globals():
            env.close()
            print("✅ 環境已關閉")
        
        print("\n🎉 本地總結完成！感謝使用！")

if __name__ == "__main__":
    main()