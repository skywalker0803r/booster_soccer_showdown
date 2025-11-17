from sai_rl import SAIClient
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
import numpy as np
import os
import torch
import gymnasium as gym
from gymnasium.spaces import Box
from datetime import datetime

class RewardShapingPreprocessor():
    """包含獎勵塑形的預處理器"""
    
    def __init__(self):
        # 初始化前一個時間步的潛力值
        self._prev_potential = None
        
    def get_task_onehot(self, info):
        if 'task_index' in info:
            return info['task_index']
        else:
            return np.array([1, 0, 0])  # 預設為 Task 1

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

    def reward_shaping(self, reward, info, gamma=0.99):
        """基於潛力函數的獎勵塑形 (PBRS)"""
        
        # 獲取任務類型
        task_onehot = self.get_task_onehot(info)
        if len(task_onehot.shape) > 1:
            task_onehot = task_onehot.squeeze()
        
        # 提取關鍵位置資訊
        try:
            ball_pos_rel_robot = info["ball_xpos_rel_robot"].squeeze()
            goal_pos_rel_robot = info["goal_team_0_rel_robot"].squeeze()
            target_pos_rel_robot = info["target_xpos_rel_robot"].squeeze()
            
            # 計算潛力函數
            current_potential = 0.0
            
            # 通用獎勵：鼓勵接近球
            dist_robot_ball = np.linalg.norm(ball_pos_rel_robot)
            robot_to_ball_potential = -0.1 * dist_robot_ball  # 越近越好
            current_potential += robot_to_ball_potential
            
            if task_onehot[0] == 1 or task_onehot[1] == 1:  # Task 1 & 2: 踢球入門
                # 鼓勵球接近球門
                dist_ball_goal = np.linalg.norm(goal_pos_rel_robot - ball_pos_rel_robot)
                ball_to_goal_potential = -0.05 * dist_ball_goal
                current_potential += ball_to_goal_potential
                
            elif task_onehot[2] == 1:  # Task 3: 精準傳球
                # 鼓勵球接近目標點
                dist_ball_target = np.linalg.norm(target_pos_rel_robot - ball_pos_rel_robot)
                ball_to_target_potential = -0.05 * dist_ball_target
                current_potential += ball_to_target_potential
            
            # 額外獎勵：鼓勵球的運動（速度獎勵）
            if "ball_velp_rel_robot" in info:
                ball_velocity = info["ball_velp_rel_robot"].squeeze()
                ball_speed = np.linalg.norm(ball_velocity)
                speed_potential = 0.01 * ball_speed  # 鼓勵球運動
                current_potential += speed_potential
                
        except Exception as e:
            print(f"獎勵塑形計算錯誤: {e}")
            current_potential = 0.0
        
        # 處理第一步
        if self._prev_potential is None:
            self._prev_potential = current_potential
            shaped_reward = reward  # 第一步不加額外獎勵
        else:
            # PBRS 公式
            potential_diff = gamma * current_potential - self._prev_potential
            shaped_reward = reward + potential_diff
            self._prev_potential = current_potential
        
        return float(shaped_reward)

    def reset_episode(self):
        """重置 episode 時調用"""
        self._prev_potential = None

# Enhanced TensorBoard callback with best model saving
class TensorBoardRewardCallback(BaseCallback):
    def __init__(self, save_path="./saved_models", save_prefix="best_model", verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_count = 0
        self.save_path = save_path
        self.save_prefix = save_prefix
        
        # 最佳模型追蹤
        self.best_mean_reward = float('-inf')
        self.best_single_reward = float('-inf')
        self.evaluation_window = 100
        self.check_freq = 10000
        
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
                    
                    # 追蹤最佳單次獎勵
                    if episode_reward > self.best_single_reward:
                        self.best_single_reward = episode_reward
                        single_best_path = os.path.join(self.save_path, f"{self.save_prefix}_single_best.zip")
                        self.model.save(single_best_path)
                        print(f"🏆 NEW SINGLE BEST! Reward: {episode_reward:.4f} - Saved to {single_best_path}")
                    
                    self.episode_rewards.append(episode_reward)
                    if len(self.episode_rewards) > 200:
                        self.episode_rewards.pop(0)
                    
                    # 計算移動平均
                    if len(self.episode_rewards) >= 10:
                        avg_10 = np.mean(self.episode_rewards[-10:])
                        self.logger.record('reward/avg_reward_10ep', avg_10)
                    
                    if len(self.episode_rewards) >= 50:
                        avg_50 = np.mean(self.episode_rewards[-50:])
                        self.logger.record('reward/avg_reward_50ep', avg_50)
                        
                    if len(self.episode_rewards) >= 100:
                        avg_100 = np.mean(self.episode_rewards[-100:])
                        self.logger.record('reward/avg_reward_100ep', avg_100)
                        
                        # 檢查是否為最佳平均獎勵
                        if avg_100 > self.best_mean_reward:
                            self.best_mean_reward = avg_100
                            mean_best_path = os.path.join(self.save_path, f"{self.save_prefix}_mean_best.zip")
                            self.model.save(mean_best_path)
                            print(f"📈 NEW MEAN BEST! Avg reward (100 ep): {avg_100:.4f} - Saved to {mean_best_path}")

        # 定期保存檢查點
        if self.n_calls % self.check_freq == 0:
            checkpoint_path = os.path.join(self.save_path, f"{self.save_prefix}_checkpoint_{self.n_calls}.zip")
            self.model.save(checkpoint_path)
            print(f"💾 Checkpoint saved: {checkpoint_path}")

        return True
    
    def get_best_stats(self):
        """獲取最佳統計資訊"""
        return {
            'best_single_reward': self.best_single_reward,
            'best_mean_reward': self.best_mean_reward,
            'total_episodes': self.episode_count,
            'final_avg_reward': np.mean(self.episode_rewards[-100:]) if len(self.episode_rewards) >= 100 else np.mean(self.episode_rewards)
        }

# 創建環境包裝器
class SAIRewardShapingWrapper(gym.Wrapper):
    """包含獎勵塑形的環境包裝器"""
    
    def __init__(self, sai_env, preprocessor_class):
        super().__init__(sai_env)
        self.preprocessor = preprocessor_class()
        
        self.observation_space = Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(89,), 
            dtype=np.float32
        )
        
        self.action_space = sai_env.action_space
        self.episode_count = 0
    
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        
        # 重置獎勵塑形器
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
        
        # 應用獎勵塑形
        if not (terminated or truncated):
            reward = self.preprocessor.reward_shaping(reward, info, gamma=0.99)
        
        if processed_obs.ndim == 2 and processed_obs.shape[0] == 1:
            processed_obs = processed_obs.squeeze(0)
        
        return processed_obs.astype(np.float32), reward, terminated, truncated, info

# 選擇訓練模式的函數
def choose_training_mode():
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
                
                if model_path.isdigit():
                    idx = int(model_path)
                    if idx == 0:
                        model_path = input("請輸入完整路徑: ").strip()
                    elif 1 <= idx <= len(model_files):
                        model_path = f"./saved_models/{model_files[idx-1]}"
                    else:
                        print("❌ 無效的選擇")
                        continue
                
                if os.path.exists(model_path):
                    return "continue", model_path
                else:
                    print(f"❌ 找不到檔案: {model_path}")
                    retry = input("重新輸入? (y/n): ").lower()
                    if retry != 'y':
                        return "new", None
        else:
            print("❌ 請輸入 1 或 2")

def main():
    print("🎯 PPO + 獎勵塑形訓練 (修復版)")
    print("=" * 50)
    
    # Initialize SAI
    sai = SAIClient(comp_id="booster-soccer-showdown", api_key="sai_LFcuaCZiqEkUbNVolQ3wbk5yU7H11jfv")
    base_env = sai.make_env()
    
    # 包裝環境
    env = SAIRewardShapingWrapper(base_env, RewardShapingPreprocessor)
    
    print(f"✅ 環境已包裝 (含獎勵塑形)")
    print(f"   原始觀察空間: {base_env.observation_space}")
    print(f"   處理後觀察空間: {env.observation_space}")
    
    # 選擇訓練模式
    training_mode, model_path = choose_training_mode()
    
    # 設定 TensorBoard
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if training_mode == "new":
        tensorboard_log = f"./runs/PPO_RewardShaping_{timestamp}"
        print(f"\n🆕 從頭開始新訓練 (含獎勵塑形)")
    else:
        tensorboard_log = f"./runs/PPO_RewardShaping_Continue_{timestamp}"
        print(f"\n🔄 繼續訓練模型: {model_path}")
    
    os.makedirs("./runs", exist_ok=True)
    print(f"📊 TensorBoard: {tensorboard_log}")
    
    # 創建模型
    policy_kwargs = dict(net_arch=[256, 128, 64])
    
    if training_mode == "new":
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
        try:
            model = PPO.load(model_path, env=env)
            model.tensorboard_log = tensorboard_log
            print("✅ 模型載入成功")
        except Exception as e:
            print(f"❌ 模型載入失敗: {e}")
            model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=tensorboard_log)
    
    # 訓練步數
    while True:
        try:
            steps_input = input(f"\n請輸入訓練步數 (建議 200000): ").strip()
            if not steps_input:
                total_steps = 200000
                break
            total_steps = int(steps_input)
            if total_steps > 0:
                break
            else:
                print("❌ 請輸入正整數")
        except ValueError:
            print("❌ 請輸入有效數字")
    
    print(f"\n🚀 開始訓練...")
    print(f"   模式: {'新訓練' if training_mode == 'new' else '繼續訓練'}")
    print(f"   步數: {total_steps:,}")
    print(f"   獎勵塑形: ✅ 啟用")
    
    # 創建增強回調
    callback = TensorBoardRewardCallback(
        save_path="./saved_models",
        save_prefix=f"ppo_reward_shaping_{timestamp}"
    )
    
    print(f"\n🤖 模型會自動保存：")
    print(f"   🏆 單次最佳: xxx_single_best.zip")
    print(f"   📈 平均最佳: xxx_mean_best.zip") 
    print(f"   💾 定期檢查點: xxx_checkpoint_xxxxx.zip")
    
    # 訓練
    model.learn(total_timesteps=total_steps, callback=callback)
    
    # 獲取訓練統計
    stats = callback.get_best_stats()
    print(f"\n📊 訓練統計摘要:")
    print(f"   🏆 最佳單次獎勵: {stats['best_single_reward']:.4f}")
    print(f"   📈 最佳平均獎勵: {stats['best_mean_reward']:.4f}")
    print(f"   🎮 總回合數: {stats['total_episodes']}")
    print(f"   🎯 最終平均獎勵: {stats['final_avg_reward']:.4f}")
    
    # 保存模型
    os.makedirs("./saved_models", exist_ok=True)
    save_model_path = f"./saved_models/ppo_reward_shaping_{timestamp}"
    model.save(save_model_path)
    
    print(f"💾 模型已保存到: {save_model_path}")
    
    # 評估
    print("📈 進行本地評估...")
    
    def action_function(policy):
        expected_bounds = [-1, 1]
        action_percent = (policy - expected_bounds[0]) / (expected_bounds[1] - expected_bounds[0])
        bounded_percent = np.minimum(np.maximum(action_percent, 0), 1)
        return base_env.action_space.low + (base_env.action_space.high - base_env.action_space.low) * bounded_percent

    sai.benchmark(model, action_function, RewardShapingPreprocessor)
    
    env.close()
    
    print(f"""
🎉 訓練完成！

📦 下載以下檔案到本地:
   1. saved_models/ 資料夾 - 包含訓練好的模型
   2. runs/ 資料夾 - 包含 TensorBoard 日誌

🖥️  本地操作:
   1. 執行 local_watch.py 觀看模型並決定是否提交
   2. 執行 tensorboard --logdir=./runs 查看訓練曲線

💾 模型檔案: {save_model_path}.zip
""")

if __name__ == "__main__":
    main()