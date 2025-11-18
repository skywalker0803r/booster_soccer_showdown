import argparse
import os
from datetime import datetime
import numpy as np
import gymnasium as gym
from gymnasium.spaces import Box
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback
from collections import deque
from stable_baselines3.common.logger import configure_logger
import torch

from sai_rl import SAIClient

# --- Preprocessor ---
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

# --- Potential-Based Reward Shaping Wrapper ---
class PotentialBasedRewardWrapper(gym.Wrapper):
    def __init__(self, env, gamma=0.99, k1=1.0, k2=1.0):
        super().__init__(env)
        self.gamma = gamma
        self.k1 = k1
        self.k2 = k2
        self.previous_potential = 0.0
        self.preprocessor = Preprocessor()
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(89,), dtype=np.float32)

    def _calculate_potential(self, info):
        agent_pos = np.array([0.0, 0.0, 0.0])
        ball_pos_rel_agent = info.get("ball_xpos_rel_robot", np.array([0,0,0]))
        goal_pos_rel_agent = info.get("goal_team_1_rel_robot", np.array([0,0,0]))
        ball_pos_rel_goal = goal_pos_rel_agent - ball_pos_rel_agent
        dist_agent_to_ball = np.linalg.norm(agent_pos - ball_pos_rel_agent)
        dist_ball_to_goal = np.linalg.norm(ball_pos_rel_goal)
        potential = -self.k1 * dist_agent_to_ball - self.k2 * dist_ball_to_goal
        return potential, dist_agent_to_ball, dist_ball_to_goal

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.previous_potential, _, _ = self._calculate_potential(info)
        processed_obs = self.preprocessor.modify_state(obs, info).squeeze(0)
        return processed_obs.astype(np.float32), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        current_potential, dist_agent_ball, dist_ball_goal = self._calculate_potential(info)
        shaped_reward = self.gamma * current_potential - self.previous_potential
        total_reward = reward + shaped_reward
        
        # Add detailed info for logging
        info['reward_components'] = {
            'original': reward,
            'shaped': shaped_reward,
        }
        info['potential'] = {
            'current': current_potential,
            'previous': self.previous_potential,
        }
        info['distances'] = {
            'agent_to_ball': dist_agent_ball,
            'ball_to_goal': dist_ball_goal,
        }
        
        self.previous_potential = current_potential
        processed_obs = self.preprocessor.modify_state(obs, info).squeeze(0)
        return processed_obs.astype(np.float32), total_reward, terminated, truncated, info

# --- Custom Callback for Detailed Logging and Saving Best Models ---
def prompt_for_value(prompt_text, default, value_type=str):
    """Prompts the user for a value with a default and type casting."""
    while True:
        try:
            user_input = input(f"{prompt_text} (預設: {default}): ").strip()
            if not user_input:
                return default
            return value_type(user_input)
        except ValueError:
            print(f"無效輸入，請輸入一個 {value_type.__name__} 類型的值。")

class DetailedLogCallback(BaseCallback):
    def __init__(self, save_path, save_prefix, log_interval=10000, verbose=0):
        super().__init__(verbose)
        self.save_path = save_path
        self.save_prefix = save_prefix
        self.log_interval = log_interval
        self.best_mean_reward = -np.inf
        
        # Deques for storing metrics over the log interval
        self.episode_rewards = deque(maxlen=100) # For rolling mean of finished episodes
        self.step_metrics = {
            'original_reward': deque(maxlen=log_interval),
            'shaped_reward': deque(maxlen=log_interval),
            'potential': deque(maxlen=log_interval),
            'dist_agent_ball': deque(maxlen=log_interval),
            'dist_ball_goal': deque(maxlen=log_interval),
        }
        os.makedirs(save_path, exist_ok=True)

    def _on_step(self) -> bool:
        # Collect metrics from each environment at each step
        for i, info in enumerate(self.locals.get('infos', [])):
            if 'reward_components' in info:
                self.step_metrics['original_reward'].append(info['reward_components']['original'])
                self.step_metrics['shaped_reward'].append(info['reward_components']['shaped'])
            if 'potential' in info:
                self.step_metrics['potential'].append(info['potential']['current'])
            if 'distances' in info:
                self.step_metrics['dist_agent_ball'].append(info['distances']['agent_to_ball'])
                self.step_metrics['dist_ball_goal'].append(info['distances']['ball_to_goal'])

            if 'episode' in info:
                self.episode_rewards.append(info['episode']['r'])
                self.logger.record('episode/reward', info['episode']['r'])
                self.logger.record('episode/length', info['episode']['l'])

        # Log to terminal and TensorBoard at specified interval
        if self.n_calls % self.log_interval == 0 and self.n_calls > 0:
            # Calculate mean of step-wise metrics
            mean_original_reward = np.mean(self.step_metrics['original_reward']) if self.step_metrics['original_reward'] else 0
            mean_shaped_reward = np.mean(self.step_metrics['shaped_reward']) if self.step_metrics['shaped_reward'] else 0
            mean_potential = np.mean(self.step_metrics['potential']) if self.step_metrics['potential'] else 0
            mean_dist_agent_ball = np.mean(self.step_metrics['dist_agent_ball']) if self.step_metrics['dist_agent_ball'] else 0
            mean_dist_ball_goal = np.mean(self.step_metrics['dist_ball_goal']) if self.step_metrics['dist_ball_goal'] else 0
            
            # Log to TensorBoard
            self.logger.record('step_metrics/mean_original_reward', mean_original_reward)
            self.logger.record('step_metrics/mean_shaped_reward', mean_shaped_reward)
            self.logger.record('step_metrics/mean_potential', mean_potential)
            self.logger.record('step_metrics/mean_dist_agent_ball', mean_dist_agent_ball)
            self.logger.record('step_metrics/mean_dist_ball_goal', mean_dist_ball_goal)
            
            # Log to terminal
            if self.verbose > 0:
                print(f"\n--- Step {self.num_timesteps} Log ---")
                print(f"  Avg Original Reward: {mean_original_reward:.4f}")
                print(f"  Avg Shaped Reward:   {mean_shaped_reward:.4f}")
                print(f"  Avg Potential:       {mean_potential:.2f}")
                print(f"  Avg Dist Agent-Ball: {mean_dist_agent_ball:.2f}")
                print(f"  Avg Dist Ball-Goal:  {mean_dist_ball_goal:.2f}")
                if self.episode_rewards:
                    print(f"  Rolling Episode Reward (100ep): {np.mean(self.episode_rewards):.2f}")
                print("-------------------------\n")

        # Save best model based on rolling episode reward
        if self.episode_rewards and len(self.episode_rewards) >= 20: # Start checking after 20 episodes
            current_mean_reward = np.mean(self.episode_rewards)
            if current_mean_reward > self.best_mean_reward:
                self.best_mean_reward = current_mean_reward
                best_model_path = os.path.join(self.save_path, f"{self.save_prefix}_best.zip")
                self.model.save(best_model_path)
                if self.verbose > 0:
                    print(f"📈 New best mean reward: {self.best_mean_reward:.2f} -> Saved model to {best_model_path}")
        return True

# --- Main Execution Block ---
if __name__ == "__main__":
    # --- Interactive Configuration ---
    config = {}
    print("\n--- 請設定訓練參數 ---")
    config['total_timesteps'] = prompt_for_value("總訓練步數", default=1000000, value_type=int)
    config['n_envs'] = prompt_for_value("平行環境數量", default=4, value_type=int)
    config['log_interval'] = prompt_for_value("日誌記錄間隔 (步)", default=10000, value_type=int)
    
    print("\n--- PPO 超參數 ---")
    config['lr'] = prompt_for_value("學習率 (Learning Rate)", default=3e-4, value_type=float)
    config['n_steps'] = prompt_for_value("每次更新的步數 (N_Steps)", default=2048, value_type=int)
    config['batch_size'] = prompt_for_value("批次大小 (Batch Size)", default=64, value_type=int)
    config['gamma'] = prompt_for_value("折扣因子 (Gamma)", default=0.99, value_type=float)
    config['clip_range'] = prompt_for_value("PPO 裁剪範圍 (Clip Range)", default=0.2, value_type=float)

    print("\n--- PBRS 獎勵塑形超參數 ---")
    config['k1'] = prompt_for_value("k1 (智能體到球的距離權重)", default=1.0, value_type=float)
    config['k2'] = prompt_for_value("k2 (球到球門的距離權重)", default=1.0, value_type=float)

    # --- Environment Setup ---
    sai = SAIClient(comp_id="booster-soccer-showdown", api_key="sai_LFcuaCZiqEkUbNVolQ3wbk5yU7H11jfv")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = "./saved_models"
    
    wrapper_kwargs = {'gamma': config['gamma'], 'k1': config['k1'], 'k2': config['k2']}

    env = make_vec_env(
        sai.make_env,
        n_envs=config['n_envs'],
        wrapper_class=PotentialBasedRewardWrapper,
        wrapper_kwargs=wrapper_kwargs
    )
    
    # --- Choose Training Mode ---
    model_path_to_load = None
    print("\n--- 請選擇訓練模式 ---")
    print("1 - 從頭開始新訓練")
    print("2 - 載入現有模型繼續訓練")
    
    while True:
        choice = input("請選擇 (1 或 2): ").strip()
        if choice == '1':
            mode = 'new'
            break
        elif choice == '2':
            mode = 'continue'
            model_files = [f for f in os.listdir(save_path) if f.endswith(".zip")] if os.path.exists(save_path) else []
            if not model_files:
                print(f"在 '{save_path}' 資料夾中找不到任何模型。將開始新訓練。")
                mode = 'new'
                break
            
            print("\n找到的模型檔案:")
            for i, file in enumerate(model_files, 1):
                print(f"{i}. {file}")
            
            while True:
                try:
                    idx = int(input(f"請選擇要載入的模型編號 (1-{len(model_files)}): "))
                    if 1 <= idx <= len(model_files):
                        model_path_to_load = os.path.join(save_path, model_files[idx-1])
                        break
                    else:
                        print("無效的編號。")
                except ValueError:
                    print("請輸入數字。")
            break
        else:
            print("無效輸入，請重新輸入 1 或 2。")

    # --- Model Loading or Creation ---
    if mode == 'continue' and model_path_to_load:
        print(f"載入模型: {model_path_to_load}")
        log_dir = f"./runs/PPO_PBRS_Continue_{timestamp}"
        save_prefix = f"ppo_pbrs_continued_{timestamp}"
        try:
            model = PPO.load(model_path_to_load, env=env)
            new_logger = configure_logger(verbose=0, tensorboard_log=log_dir, reset_num_timesteps=False)
            model.set_logger(new_logger)
            print("模型載入成功，將繼續訓練。\n注意：模型將使用已保存的超參數，剛才輸入的PPO超參數將被忽略。")
        except Exception as e:
            print(f"模型載入失敗: {e}。將創建一個新模型。")
            mode = 'new'
    
    if mode == 'new':
        print("創建新模型...")
        log_dir = f"./runs/PPO_PBRS_{timestamp}"
        save_prefix = f"ppo_pbrs_{timestamp}"
        model = PPO(
            "MlpPolicy",
            env,
            verbose=0,
            tensorboard_log=log_dir,
            policy_kwargs=dict(net_arch=[256,256,128,128,64]),
            learning_rate=config['lr'],
            n_steps=config['n_steps'],
            batch_size=config['batch_size'],
            gamma=config['gamma'],
            clip_range=config['clip_range'],
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )

    print("\n--- 最終訓練配置 ---")
    print(f"模式: {'繼續訓練' if mode == 'continue' else '新訓練'}")
    print(f"平行環境數量: {config['n_envs']}")
    print(f"總訓練步數: {config['total_timesteps']:,}")
    print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print(f"日誌記錄間隔: {config['log_interval']} 步")
    if mode == 'new':
        print("\nPPO Hyperparameters:")
        print(f"  Learning Rate: {model.learning_rate}")
        print(f"  N_Steps: {model.n_steps}")
        print(f"  Batch Size: {model.batch_size}")
        print(f"  Gamma: {model.gamma}")
        print(f"  Clip Range: {model.clip_range}")
    print("\nPBRS Hyperparameters (在環境初始化時生效):")
    print(f"  k1 (agent-ball): {config['k1']}")
    print(f"  k2 (ball-goal): {config['k2']}")
    print(f"\nTensorBoard Log: {log_dir}")
    print(f"Models will be saved in: {save_path}")
    print("-----------------------------\n")

    # --- Model Training ---
    callback = DetailedLogCallback(
        save_path=save_path, 
        save_prefix=save_prefix, 
        log_interval=config['log_interval'],
        verbose=1
    )

    try:
        model.learn(total_timesteps=config['total_timesteps'], callback=callback, reset_num_timesteps=(mode=='new'))
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
    finally:
        final_model_path = os.path.join(save_path, f"{save_prefix}_final.zip")
        model.save(final_model_path)
        print(f"\nFinal model saved to: {final_model_path}")
        env.close()

    print("\nTraining complete.")
    print(f"To view logs, run: tensorboard --logdir={log_dir}")




    # Note: The benchmark function from the original script might need adjustments
    # as it expects a different preprocessor structure. For now, we focus on training.
    # To benchmark, you would load the saved model and run it in an environment.
    
    # Example of how to load and benchmark later:
    # model = PPO.load(f"{save_path}/{save_prefix}_best.zip")
    # sai.benchmark(model, action_function, Preprocessor) # action_function needs to be defined