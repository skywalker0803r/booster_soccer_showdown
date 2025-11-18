from sai_rl import SAIClient
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
import numpy as np
import os
import torch
import gymnasium as gym
from gymnasium.spaces import Box
from datetime import datetime

## Initialize the SAI client
sai = SAIClient(comp_id="booster-soccer-showdown", api_key="sai_LFcuaCZiqEkUbNVolQ3wbk5yU7H11jfv")

## Make the environment
base_env = sai.make_env()

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

# 創建環境包裝器來正確處理預處理
import gymnasium as gym
from gymnasium.spaces import Box

class SAIPreprocessorWrapper(gym.Wrapper):
    """包裝器，將 SAI 環境與預處理器整合"""
    
    def __init__(self, sai_env, preprocessor_class):
        super().__init__(sai_env)
        self.preprocessor = preprocessor_class()
        
        # 重新定義觀察空間為預處理後的 89 維
        self.observation_space = Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(89,), 
            dtype=np.float32
        )
        
        # 動作空間保持不變
        self.action_space = sai_env.action_space
    
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        
        # 不使用獎勵形塑，無需重置
        
        # 預處理觀察
        processed_obs = self.preprocessor.modify_state(obs, info)
        
        # 確保輸出是一維數組
        if processed_obs.ndim == 2 and processed_obs.shape[0] == 1:
            processed_obs = processed_obs.squeeze(0)
        
        return processed_obs.astype(np.float32), info
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # 預處理觀察
        processed_obs = self.preprocessor.modify_state(obs, info)
        
        # 不使用獎勵形塑，保持原始獎勵
        # reward = reward  # 保持原始獎勵不變
        
        # 確保輸出是一維數組
        if processed_obs.ndim == 2 and processed_obs.shape[0] == 1:
            processed_obs = processed_obs.squeeze(0)
        
        return processed_obs.astype(np.float32), reward, terminated, truncated, info

# 包裝環境（不使用獎勵形塑，只用基本預處理器）
env = SAIPreprocessorWrapper(base_env, Preprocessor)

print(f"✅ 環境已包裝")
print(f"   原始觀察空間: {base_env.observation_space}")
print(f"   處理後觀察空間: {env.observation_space}")
print(f"   動作空間: {env.action_space}")

# TensorBoard callback for logging rewards
class TensorBoardRewardCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_count = 0

    def _on_step(self) -> bool:
        # Log rewards when episodes are done
        if len(self.locals.get('infos', [])) > 0:
            for info in self.locals['infos']:
                if 'episode' in info:
                    episode_reward = info['episode']['r']
                    episode_length = info['episode']['l']
                    self.episode_count += 1
                    
                    # Log to tensorboard
                    self.logger.record('reward/episode_reward', episode_reward)
                    self.logger.record('reward/episode_length', episode_length)
                    self.logger.record('reward/episode_count', self.episode_count)
                    
                    print(f"Episode {self.episode_count}: Reward = {episode_reward:.4f}, Length = {episode_length}")
                    
                    # Keep track for moving average
                    self.episode_rewards.append(episode_reward)
                    if len(self.episode_rewards) > 100:
                        self.episode_rewards.pop(0)
                    
                    # Log moving averages
                    if len(self.episode_rewards) >= 10:
                        avg_10 = np.mean(self.episode_rewards[-10:])
                        self.logger.record('reward/avg_reward_10ep', avg_10)
                    
                    if len(self.episode_rewards) >= 50:
                        avg_50 = np.mean(self.episode_rewards[-50:])
                        self.logger.record('reward/avg_reward_50ep', avg_50)
                        
                    if len(self.episode_rewards) == 100:
                        avg_100 = np.mean(self.episode_rewards)
                        self.logger.record('reward/avg_reward_100ep', avg_100)

        return True

def choose_training_mode():
    """選擇訓練模式：從頭開始或繼續訓練"""
    print("\n" + "="*50)
    print("🤔 請選擇訓練模式：")
    print("   1 - 從頭開始新訓練")
    print("   2 - 載入現有模型繼續訓練")
    print("="*50)
    
    while True:
        choice = input("請選擇 (1 或 2): ").strip()
        
        if choice == "1":
            return "new", None
            
        elif choice == "2":
            # 顯示可用的模型
            if os.path.exists("./saved_models"):
                print("\n📁 找到的模型檔案:")
                model_files = [f for f in os.listdir("./saved_models") if f.endswith(".zip")]
                if model_files:
                    for i, file in enumerate(model_files, 1):
                        print(f"   {i}. {file}")
                    print("   0. 手動輸入路徑")
                else:
                    print("   (沒有找到模型檔案)")
            
            while True:
                model_path = input("\n請輸入模型檔案路徑 (或輸入數字選擇): ").strip()
                
                # 如果輸入數字，選擇對應的模型
                if model_path.isdigit():
                    idx = int(model_path)
                    if idx == 0:
                        model_path = input("請輸入完整路徑: ").strip()
                    elif 1 <= idx <= len(model_files):
                        model_path = f"./saved_models/{model_files[idx-1]}"
                    else:
                        print("❌ 無效的選擇")
                        continue
                
                # 檢查檔案是否存在
                if os.path.exists(model_path):
                    return "continue", model_path
                else:
                    print(f"❌ 找不到檔案: {model_path}")
                    retry = input("重新輸入? (y/n): ").lower()
                    if retry != 'y':
                        return "new", None
        else:
            print("❌ 請輸入 1 或 2")

# 選擇訓練模式
training_mode, model_path = choose_training_mode()

# 設定 TensorBoard 日誌目錄
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
if training_mode == "new":
    tensorboard_log = f"./runs/SimplePPO_{timestamp}"
    print(f"\n🆕 從頭開始新訓練")
else:
    tensorboard_log = f"./runs/SimplePPO_Continue_{timestamp}"
    print(f"\n🔄 繼續訓練模型: {model_path}")

os.makedirs("./runs", exist_ok=True)

print(f"📊 TensorBoard 日誌將保存到: {tensorboard_log}")
print(f"🖥️  啟動 TensorBoard 指令: tensorboard --logdir=./runs")

## Create or load the model
# 配置 PPO 策略，指定正確的觀察空間維度
policy_kwargs = dict(
    net_arch=[256, 128, 64],  # 與 DDPG 版本相同的網路架構
)

if training_mode == "new":
    print("\n🆕 創建新的 PPO 模型...")
    model = PPO(
        "MlpPolicy", 
        env, 
        verbose=1, 
        tensorboard_log=tensorboard_log,
        policy_kwargs=policy_kwargs,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
else:
    print("\n📥 載入現有模型...")
    try:
        model = PPO.load(model_path, env=env)
        # 更新 tensorboard 日誌路徑
        model.tensorboard_log = tensorboard_log
        print("✅ 模型載入成功")
    except Exception as e:
        print(f"❌ 模型載入失敗: {e}")
        print("🔄 改為創建新模型...")
        model = PPO(
            "MlpPolicy", 
            env, 
            verbose=1, 
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )

## Define an action function
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

## Train the model
# 創建回調函數
callback = TensorBoardRewardCallback()

# 詢問訓練步數
default_steps = 100000
while True:
    try:
        steps_input = input(f"\n請輸入訓練步數 (預設 {default_steps}): ").strip()
        if not steps_input:
            total_steps = default_steps
            break
        total_steps = int(steps_input)
        if total_steps > 0:
            break
        else:
            print("❌ 請輸入正整數")
    except ValueError:
        print("❌ 請輸入有效數字")

print(f"\n🚀 開始訓練 PPO 模型...")
print(f"   模式: {'新訓練' if training_mode == 'new' else '繼續訓練'}")
print(f"   步數: {total_steps:,}")

model.learn(total_timesteps=total_steps, callback=callback)

# 保存模型
os.makedirs("./saved_models", exist_ok=True)
if training_mode == "new":
    save_model_path = f"./saved_models/simple_ppo_{timestamp}"
else:
    save_model_path = f"./saved_models/simple_ppo_continued_{timestamp}"

model.save(save_model_path)
print(f"💾 模型已保存到: {save_model_path}")

## Watch (註解掉，因為在 Colab 上無法使用)
#sai.watch(model, action_function, Preprocessor)
print("ℹ️  sai.watch 功能已註解掉 (Colab 環境不支援)")

## Benchmark the model locally
print("📈 進行本地評估...")
sai.benchmark(model, action_function, Preprocessor)

env.close()

print(f"""
🎉 Colab 訓練完成！

📦 下載以下檔案到本地:
   1. saved_models/ 資料夾 - 包含訓練好的模型
   2. runs/ 資料夾 - 包含 TensorBoard 日誌

🖥️  本地操作:
   1. 執行 local_watch.py 觀看模型並決定是否提交
   2. 執行 tensorboard --logdir=./runs 查看訓練曲線

💾 模型檔案: {save_model_path}.zip
""")