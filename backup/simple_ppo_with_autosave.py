from sai_rl import SAIClient
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
import numpy as np
import os
import torch
import gymnasium as gym
from gymnasium.spaces import Box
from datetime import datetime
from aligned_reward_shaping_fixed import aligned_enhanced_preprocessor

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

class TensorBoardRewardCallback(BaseCallback):
    def __init__(self, save_path="./saved_models", save_prefix="best_model", verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_count = 0
        self.save_path = save_path
        self.save_prefix = save_prefix
        self.best_mean_reward = float('-inf')
        self.best_single_reward = float('-inf')
        os.makedirs(save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if len(self.locals.get('infos', [])) > 0:
            for info in self.locals['infos']:
                if 'episode' in info:
                    episode_reward = info['episode']['r']
                    episode_length = info['episode']['l']
                    self.episode_count += 1
                    
                    self.logger.record('reward/episode_reward', episode_reward)
                    self.logger.record('reward/episode_length', episode_length)
                    self.logger.record('reward/episode_count', self.episode_count)
                    
                    print(f"Episode {self.episode_count}: Reward = {episode_reward:.4f}, Length = {episode_length}")
                    
                    if episode_reward > self.best_single_reward:
                        self.best_single_reward = episode_reward
                        single_best_path = os.path.join(self.save_path, f"{self.save_prefix}_single_best.zip")
                        self.model.save(single_best_path)
                        print(f"🏆 NEW SINGLE BEST! Reward: {episode_reward:.4f}")
                    
                    self.episode_rewards.append(episode_reward)
                    if len(self.episode_rewards) > 100:
                        self.episode_rewards.pop(0)
                    
                    if len(self.episode_rewards) >= 10:
                        avg_10 = np.mean(self.episode_rewards[-10:])
                        self.logger.record('reward/avg_reward_10ep', avg_10)
                    
                    if len(self.episode_rewards) >= 50:
                        avg_50 = np.mean(self.episode_rewards[-50:])
                        self.logger.record('reward/avg_reward_50ep', avg_50)
                        
                    if len(self.episode_rewards) >= 100:
                        avg_100 = np.mean(self.episode_rewards)
                        self.logger.record('reward/avg_reward_100ep', avg_100)
                        
                        if avg_100 > self.best_mean_reward:
                            self.best_mean_reward = avg_100
                            mean_best_path = os.path.join(self.save_path, f"{self.save_prefix}_mean_best.zip")
                            self.model.save(mean_best_path)
                            print(f"📈 NEW MEAN BEST! Avg reward (100 ep): {avg_100:.4f}")
        return True

class SAIPreprocessorWrapper(gym.Wrapper):
    def __init__(self, sai_env, preprocessor_class):
        super().__init__(sai_env)
        self.preprocessor = preprocessor_class()
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(89,), dtype=np.float32)
        self.action_space = sai_env.action_space
        self.episode_count = 0
    
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        
        # 重置獎勵形塑器
        if hasattr(self.preprocessor, 'reset_episode'):
            self.preprocessor.reset_episode()
        self.episode_count += 1
        
        processed_obs = self.preprocessor.modify_state(obs, info)
        if processed_obs.ndim == 2 and processed_obs.shape[0] == 1:
            processed_obs = processed_obs.squeeze(0)
        return processed_obs.astype(np.float32), info
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        original_reward = reward
        
        processed_obs = self.preprocessor.modify_state(obs, info)
        
        # 應用獎勵形塑
        if hasattr(self.preprocessor, 'shape_reward') and not (terminated or truncated):
            reward = self.preprocessor.shape_reward(processed_obs.squeeze(), info, reward, terminated or truncated)
        
        if processed_obs.ndim == 2 and processed_obs.shape[0] == 1:
            processed_obs = processed_obs.squeeze(0)
        
        # 偶爾顯示獎勵比較
        if self.episode_count % 100 == 0 and not (terminated or truncated):
            print(f"Step - Original: {original_reward:.4f}, Shaped: {reward:.4f}")
        
        return processed_obs.astype(np.float32), reward, terminated, truncated, info

def choose_training_mode():
    print("\n請選擇訓練模式：")
    print("1 - 從頭開始新訓練")
    print("2 - 載入現有模型繼續訓練")
    
    while True:
        choice = input("請選擇 (1 或 2): ").strip()
        if choice == "1":
            return "new", None
        elif choice == "2":
            if os.path.exists("./saved_models"):
                model_files = [f for f in os.listdir("./saved_models") if f.endswith(".zip")]
                if model_files:
                    print("\n找到的模型檔案:")
                    for i, file in enumerate(model_files, 1):
                        print(f"{i}. {file}")
                    while True:
                        try:
                            idx = int(input("請選擇檔案編號: "))
                            if 1 <= idx <= len(model_files):
                                return "continue", f"./saved_models/{model_files[idx-1]}"
                        except ValueError:
                            pass
                        print("無效選擇")
            return "new", None
        else:
            print("請輸入 1 或 2")

def action_function(policy):
    expected_bounds = [-1, 1]
    action_percent = (policy - expected_bounds[0]) / (expected_bounds[1] - expected_bounds[0])
    bounded_percent = np.minimum(np.maximum(action_percent, 0), 1)
    return base_env.action_space.low + (base_env.action_space.high - base_env.action_space.low) * bounded_percent

## Initialize the SAI client
sai = SAIClient(comp_id="booster-soccer-showdown", api_key="sai_LFcuaCZiqEkUbNVolQ3wbk5yU7H11jfv")

## Make the environment
base_env = sai.make_env()

# 創建增強預處理器（含獎勵形塑）
EnhancedPreprocessor = aligned_enhanced_preprocessor(Preprocessor)
env = SAIPreprocessorWrapper(base_env, EnhancedPreprocessor)

print(f"環境已包裝 (含獎勵形塑)")
print(f"原始觀察空間: {base_env.observation_space}")
print(f"處理後觀察空間: {env.observation_space}")
print(f"獎勵形塑: ✅ 啟用")

# 選擇訓練模式
training_mode, model_path = choose_training_mode()

# 設定 TensorBoard 日誌目錄
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
if training_mode == "new":
    tensorboard_log = f"./runs/SimplePPO_{timestamp}"
    print(f"從頭開始新訓練")
else:
    tensorboard_log = f"./runs/SimplePPO_Continue_{timestamp}"
    print(f"繼續訓練模型: {model_path}")

os.makedirs("./runs", exist_ok=True)
print(f"TensorBoard 日誌: {tensorboard_log}")

## Create or load the model
policy_kwargs = dict(net_arch=[256, 128, 64])

if training_mode == "new":
    print("創建新的 PPO 模型...")
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=tensorboard_log, policy_kwargs=policy_kwargs, learning_rate=3e-4, n_steps=2048, batch_size=64, device='cuda' if torch.cuda.is_available() else 'cpu')
else:
    print("載入現有模型...")
    try:
        model = PPO.load(model_path, env=env)
        model.tensorboard_log = tensorboard_log
        print("模型載入成功")
    except Exception as e:
        print(f"模型載入失敗: {e}")
        model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=tensorboard_log, policy_kwargs=policy_kwargs, learning_rate=3e-4, n_steps=2048, batch_size=64, device='cuda' if torch.cuda.is_available() else 'cpu')

# 訓練步數
while True:
    try:
        steps_input = input(f"請輸入訓練步數 (預設 100000): ").strip()
        if not steps_input:
            total_steps = 100000
            break
        total_steps = int(steps_input)
        if total_steps > 0:
            break
        else:
            print("請輸入正整數")
    except ValueError:
        print("請輸入有效數字")

print(f"開始訓練...")
print(f"模式: {'新訓練' if training_mode == 'new' else '繼續訓練'}")
print(f"步數: {total_steps:,}")

## Train the model
callback = TensorBoardRewardCallback(save_path="./saved_models", save_prefix=f"simple_ppo_{timestamp}")

print("模型會自動保存:")
print("🏆 單次最佳: xxx_single_best.zip")
print("📈 平均最佳: xxx_mean_best.zip")

model.learn(total_timesteps=total_steps, callback=callback)

# 保存模型
os.makedirs("./saved_models", exist_ok=True)
if training_mode == "new":
    save_model_path = f"./saved_models/simple_ppo_{timestamp}"
else:
    save_model_path = f"./saved_models/simple_ppo_continued_{timestamp}"

model.save(save_model_path)
print(f"模型已保存到: {save_model_path}")

## Benchmark the model locally
print("進行本地評估...")
sai.benchmark(model, action_function, EnhancedPreprocessor)

env.close()

print(f"""
訓練完成！

📦 下載以下檔案到本地:
   1. saved_models/ 資料夾 - 包含訓練好的模型
   2. runs/ 資料夾 - 包含 TensorBoard 日誌

🖥️  本地操作:
   1. 執行 local_watch.py 觀看模型並決定是否提交
   2. 執行 tensorboard --logdir=./runs 查看訓練曲線

💾 模型檔案: {save_model_path}.zip
""")