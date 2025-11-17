from sai_rl import SAIClient
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
import numpy as np
import os
import torch
import gymnasium as gym
from gymnasium.spaces import Box
from datetime import datetime

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

class SimpleRewardShaper:
    """
    極簡獎勵塑形：大道至簡
    只關注：不倒下 + 持續動作 + 球速度
    """
    
    def __init__(self):
        self.step_count = 0
    
    def shape_reward(self, obs, info, original_reward, terminated):
        shaped_reward = 0.0
        self.step_count += 1
        
        # Extract key info
        robot_quat = info.get("robot_quat", np.array([[0, 0, 0, 1]]))
        robot_vel = info.get("robot_velocimeter", np.array([[0, 0, 0]]))
        ball_vel = info.get("ball_velp_rel_robot", np.array([[0, 0, 0]]))
        
        # Handle shape
        if len(robot_quat.shape) > 1:
            robot_quat = robot_quat[0]
        if len(robot_vel.shape) > 1:
            robot_vel = robot_vel[0]
        if len(ball_vel.shape) > 1:
            ball_vel = ball_vel[0]
        
        # 1. STAY UPRIGHT: 不倒下 (最重要)
        robot_upright = 1.0 - abs(robot_quat[2])  # Z rotation = falling
        if robot_upright > 0.85:
            shaped_reward += 0.01  # 持續站立獎勵
        
        # 2. KEEP MOVING: 持續動作，不要靜止
        robot_speed = np.linalg.norm(robot_vel)
        if robot_upright > 0.8:  # 只有站立時才獎勵動作
            if 0.1 < robot_speed < 2.0:  # 適中的移動速度
                shaped_reward += 0.005  # 鼓勵持續動作
            elif robot_speed < 0.05:  # 幾乎靜止
                shaped_reward -= 0.005  # 輕微懲罰靜止
        
        # 3. BALL VELOCITY: 球速度獎勵 (當碰巧踢到球時)
        ball_speed = np.linalg.norm(ball_vel)
        if ball_speed > 0.5:  # 球有明顯移動
            shaped_reward += 0.02 * min(ball_speed, 5.0)  # 球速度獎勵，有上限
        
        # 4. AMPLIFY SUCCESS: 放大任何正向官方獎勵
        if original_reward > 0:
            shaped_reward += 0.01  # 強化成功
            
        # 保守限制
        shaped_reward = np.clip(shaped_reward, -0.02, 0.15)
        
        return original_reward + shaped_reward

    def reset(self):
        self.step_count = 0

class MinimalEnhancedPreprocessor(Preprocessor):
    def __init__(self):
        super().__init__()
        self.reward_shaper = SimpleRewardShaper()
    
    def shape_reward(self, obs, info, reward, terminated):
        return self.reward_shaper.shape_reward(obs, info, reward, terminated)
    
    def reset_episode(self):
        self.reward_shaper.reset()

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
        
        if hasattr(self.preprocessor, 'shape_reward') and not (terminated or truncated):
            reward = self.preprocessor.shape_reward(processed_obs.squeeze(), info, reward, terminated or truncated)
        
        if processed_obs.ndim == 2 and processed_obs.shape[0] == 1:
            processed_obs = processed_obs.squeeze(0)
        
        if self.episode_count % 100 == 0 and not (terminated or truncated):
            print(f"Step - Original: {original_reward:.4f}, Shaped: {reward:.4f}")
        
        return processed_obs.astype(np.float32), reward, terminated, truncated, info

def choose_training_mode():
    print("\nChoose training mode:")
    print("1 - Start new training")
    print("2 - Continue from existing model")
    
    while True:
        choice = input("Please choose (1 or 2): ").strip()
        if choice == "1":
            return "new", None
        elif choice == "2":
            if os.path.exists("./saved_models"):
                model_files = [f for f in os.listdir("./saved_models") if f.endswith(".zip")]
                if model_files:
                    print("\nFound model files:")
                    for i, file in enumerate(model_files, 1):
                        print(f"{i}. {file}")
                    while True:
                        try:
                            idx = int(input("Select file number: "))
                            if 1 <= idx <= len(model_files):
                                return "continue", f"./saved_models/{model_files[idx-1]}"
                        except ValueError:
                            pass
                        print("Invalid choice")
            return "new", None
        else:
            print("Please enter 1 or 2")

def action_function(policy):
    expected_bounds = [-1, 1]
    action_percent = (policy - expected_bounds[0]) / (expected_bounds[1] - expected_bounds[0])
    bounded_percent = np.minimum(np.maximum(action_percent, 0), 1)
    # Use official action space bounds from docs/About.md
    action_low = np.array([-45,-45,-30,-65,-24,-15,-45,-45,-30,-65,-24,-15])
    action_high = np.array([45,45,30,65,24,15,45,45,30,65,24,15])
    return action_low + (action_high - action_low) * bounded_percent

## Initialize the SAI client
sai = SAIClient(comp_id="booster-soccer-showdown", api_key="sai_LFcuaCZiqEkUbNVolQ3wbk5yU7H11jfv")

## Make the environment
base_env = sai.make_env()
env = SAIPreprocessorWrapper(base_env, MinimalEnhancedPreprocessor)

print(f"Environment wrapped (minimal reward shaping)")
print(f"Original obs space: {base_env.observation_space}")
print(f"Processed obs space: {env.observation_space}")
print(f"Reward shaping: ✅ MINIMAL (Stay upright + Keep moving + Ball velocity)")

# Choose training mode
training_mode, model_path = choose_training_mode()

# Setup TensorBoard log
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
if training_mode == "new":
    tensorboard_log = f"./runs/PPO_Minimal_{timestamp}"
    print(f"Starting new training")
else:
    tensorboard_log = f"./runs/PPO_Minimal_Continue_{timestamp}"
    print(f"Continuing training from: {model_path}")

os.makedirs("./runs", exist_ok=True)
print(f"TensorBoard log: {tensorboard_log}")

## Create or load the model
policy_kwargs = dict(net_arch=[256, 128, 64])

if training_mode == "new":
    print("Creating new PPO model...")
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=tensorboard_log, policy_kwargs=policy_kwargs, learning_rate=3e-4, n_steps=2048, batch_size=64, device='cuda' if torch.cuda.is_available() else 'cpu')
else:
    print("Loading existing model...")
    try:
        model = PPO.load(model_path, env=env)
        model.tensorboard_log = tensorboard_log
        print("Model loaded successfully")
    except Exception as e:
        print(f"Model loading failed: {e}")
        model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=tensorboard_log, policy_kwargs=policy_kwargs, learning_rate=3e-4, n_steps=2048, batch_size=64, device='cuda' if torch.cuda.is_available() else 'cpu')

# Training steps
while True:
    try:
        steps_input = input(f"Enter training steps (default 100000): ").strip()
        if not steps_input:
            total_steps = 100000
            break
        total_steps = int(steps_input)
        if total_steps > 0:
            break
        else:
            print("Please enter positive integer")
    except ValueError:
        print("Please enter valid number")

print(f"Starting training...")
print(f"Mode: {'New training' if training_mode == 'new' else 'Continue training'}")
print(f"Steps: {total_steps:,}")
print(f"Philosophy: 大道至簡 - Keep it simple!")

## Train the model
callback = TensorBoardRewardCallback(save_path="./saved_models", save_prefix=f"ppo_minimal_{timestamp}")

print("Model auto-save enabled:")
print("🏆 Single best: xxx_single_best.zip")
print("📈 Mean best: xxx_mean_best.zip")

model.learn(total_timesteps=total_steps, callback=callback)

# Save model
os.makedirs("./saved_models", exist_ok=True)
if training_mode == "new":
    save_model_path = f"./saved_models/ppo_minimal_{timestamp}"
else:
    save_model_path = f"./saved_models/ppo_minimal_continued_{timestamp}"

model.save(save_model_path)
print(f"Model saved to: {save_model_path}")

## Benchmark the model locally
print("Running local evaluation...")
sai.benchmark(model, action_function, Preprocessor)

env.close()

print(f"""
Training complete!

📦 Download these files locally:
   1. saved_models/ folder - Contains trained models
   2. runs/ folder - Contains TensorBoard logs

🖥️  Local operations:
   1. Run local_watch_english.py to watch model and decide submission
   2. Run tensorboard --logdir=./runs to view training curves

💾 Model file: {save_model_path}.zip
🎯 Philosophy: 大道至簡 - Less is more!
""")