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
import torch

from sai_rl import SAIClient

# --- Preprocessor ---
class Preprocessor:
    def get_task_onehot(self, info):
        return info.get('task_index', np.array([]))

    def quat_rotate_inverse(self, q: np.ndarray, v: np.ndarray):
        q_w = q[:, [-1]]
        q_vec = q[:, :3]
        a = v * (2.0 * q_w**2 - 1.0)
        b = np.cross(q_vec, v) * (q_w * 2.0)
        c = q_vec * (np.dot(q_vec, v).reshape(-1, 1) * 2.0)
        return a - b + c

    def modify_state(self, obs, info):
        if obs.ndim == 1:
            obs = np.expand_dims(obs, axis=0)
        info_keys = [
            "robot_quat", "robot_gyro", "robot_accelerometer", "robot_velocimeter",
            "goal_team_0_rel_robot", "goal_team_1_rel_robot", "goal_team_0_rel_ball",
            "goal_team_1_rel_ball", "ball_xpos_rel_robot", "ball_velp_rel_robot",
            "ball_velr_rel_robot", "player_team", "goalkeeper_team_0_xpos_rel_robot",
            "goalkeeper_team_0_velp_rel_robot", "goalkeeper_team_1_xpos_rel_robot",
            "goalkeeper_team_1_velp_rel_robot", "target_xpos_rel_robot",
            "target_velp_rel_robot", "defender_xpos"
        ]
        for key in info_keys:
            if key in info:
                val = np.asarray(info[key])
                if val.ndim == 1:
                    info[key] = np.expand_dims(val, axis=0)
        task_onehot = self.get_task_onehot(info)
        if task_onehot.ndim == 1:
            task_onehot = np.expand_dims(task_onehot, axis=0)
        robot_qpos = obs[:, :12]
        robot_qvel = obs[:, 12:24]
        quat = info["robot_quat"]
        base_ang_vel = info["robot_gyro"]
        gravity_vec_2d = np.array([[0.0, 0.0, -1.0]])
        project_gravity = self.quat_rotate_inverse(quat, gravity_vec_2d)
        processed_obs = np.hstack([
            robot_qpos, robot_qvel, project_gravity, base_ang_vel,
            info["robot_accelerometer"], info["robot_velocimeter"],
            info["goal_team_0_rel_robot"], info["goal_team_1_rel_robot"],
            info["goal_team_0_rel_ball"], info["goal_team_1_rel_ball"],
            info["ball_xpos_rel_robot"], info["ball_velp_rel_robot"],
            info["ball_velr_rel_robot"], info["player_team"],
            info["goalkeeper_team_0_xpos_rel_robot"], info["goalkeeper_team_0_velp_rel_robot"],
            info["goalkeeper_team_1_xpos_rel_robot"], info["goalkeeper_team_1_velp_rel_robot"],
            info["target_xpos_rel_robot"], info["target_velp_rel_robot"],
            info["defender_xpos"], task_onehot
        ])
        return processed_obs

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
    parser = argparse.ArgumentParser(description="Train a PPO agent with Potential-Based Reward Shaping.")
    parser.add_argument("--total_timesteps", type=int, default=1_000_000, help="Total number of training steps.")
    parser.add_argument("--n_envs", type=int, default=4, help="Number of parallel environments.")
    parser.add_argument("--log_interval", type=int, default=10000, help="Steps between detailed logs.")
    # PPO Hyperparameters
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate.")
    parser.add_argument("--n_steps", type=int, default=2048, help="Number of steps to run for each environment per update.")
    parser.add_argument("--batch_size", type=int, default=64, help="Mini-batch size.")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor.")
    parser.add_argument("--clip_range", type=float, default=0.2, help="Clipping parameter for PPO.")
    # PBRS Hyperparameters
    parser.add_argument("--k1", type=float, default=1.0, help="Weight for agent-to-ball distance potential.")
    parser.add_argument("--k2", type=float, default=1.0, help="Weight for ball-to-goal distance potential.")
    
    args = parser.parse_args()

    # --- Environment Setup ---
    sai = SAIClient(comp_id="booster-soccer-showdown", api_key="sai_LFcuaCZiqEkUbNVolQ3wbk5yU7H11jfv")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"./runs/PPO_PBRS_{timestamp}"
    save_path = "./saved_models"
    save_prefix = f"ppo_pbrs_{timestamp}"

    wrapper_kwargs = {'gamma': args.gamma, 'k1': args.k1, 'k2': args.k2}

    env = make_vec_env(
        sai.make_env,
        n_envs=args.n_envs,
        wrapper_class=PotentialBasedRewardWrapper,
        wrapper_kwargs=wrapper_kwargs
    )
    
    print("--- Training Configuration ---")
    print(f"Environments: {args.n_envs}")
    print(f"Total Timesteps: {args.total_timesteps:,}")
    print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print(f"Log Interval: {args.log_interval} steps")
    print("\nPPO Hyperparameters:")
    print(f"  Learning Rate: {args.lr}")
    print(f"  N_Steps: {args.n_steps}")
    print(f"  Batch Size: {args.batch_size}")
    print(f"  Gamma: {args.gamma}")
    print(f"  Clip Range: {args.clip_range}")
    print("\nPBRS Hyperparameters:")
    print(f"  k1 (agent-ball): {args.k1}")
    print(f"  k2 (ball-goal): {args.k2}")
    print(f"\nTensorBoard Log: {log_dir}")
    print(f"Models will be saved in: {save_path}")
    print("-----------------------------")

    # --- Model Training ---
    model = PPO(
        "MlpPolicy",
        env,
        verbose=0, # Set to 0 to avoid SB3's default logs and use our custom callback's logs instead
        tensorboard_log=log_dir,
        policy_kwargs=dict(net_arch=[256, 128, 64]),
        learning_rate=args.lr,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        gamma=args.gamma,
        clip_range=args.clip_range,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )

    callback = DetailedLogCallback(
        save_path=save_path, 
        save_prefix=save_prefix, 
        log_interval=args.log_interval,
        verbose=1
    )

    try:
        model.learn(total_timesteps=args.total_timesteps, callback=callback)
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