import argparse
import os
from datetime import datetime
import numpy as np
import gymnasium as gym
from gymnasium.spaces import Box
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback
import torch

from sai_rl import SAIClient

# --- Preprocessor ---
# The original preprocessor for modifying the observation state.
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
        # Ensure obs is 2D
        if obs.ndim == 1:
            obs = np.expand_dims(obs, axis=0)

        # List of all keys we expect to use from the info dict
        info_keys = [
            "robot_quat", "robot_gyro", "robot_accelerometer", "robot_velocimeter",
            "goal_team_0_rel_robot", "goal_team_1_rel_robot", "goal_team_0_rel_ball",
            "goal_team_1_rel_ball", "ball_xpos_rel_robot", "ball_velp_rel_robot",
            "ball_velr_rel_robot", "player_team", "goalkeeper_team_0_xpos_rel_robot",
            "goalkeeper_team_0_velp_rel_robot", "goalkeeper_team_1_xpos_rel_robot",
            "goalkeeper_team_1_velp_rel_robot", "target_xpos_rel_robot",
            "target_velp_rel_robot", "defender_xpos"
        ]

        # Ensure all required info arrays are 2D
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
        # Ensure the vector for quat_rotate_inverse is also 2D
        gravity_vec_2d = np.array([[0.0, 0.0, -1.0]])
        project_gravity = self.quat_rotate_inverse(quat, gravity_vec_2d)

        # Combine all state information
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
    """
    A wrapper that implements Potential-Based Reward Shaping (PBRS).
    The shaped reward is: R' = R + gamma * potential(s') - potential(s)
    """
    def __init__(self, env, gamma=0.99, k1=1.0, k2=1.0):
        super().__init__(env)
        self.gamma = gamma
        self.k1 = k1  # Weight for agent-to-ball distance
        self.k2 = k2  # Weight for ball-to-goal distance
        self.previous_potential = 0.0
        self.preprocessor = Preprocessor()
        # The observation space is modified by the preprocessor
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(89,), dtype=np.float32)

    def _calculate_potential(self, info):
        """
        Calculates the potential Phi(s) based on the current state.
        Potential is higher when the agent is closer to the ball and the ball is closer to the opponent's goal.
        We use negative distance because we want to minimize distance.
        """
        # Get positions from the info dictionary
        agent_pos = np.array([0.0, 0.0, 0.0]) # Agent is at the origin in its own reference frame
        ball_pos_rel_agent = info.get("ball_xpos_rel_robot", np.array([0,0,0]))
        
        # The ball's position relative to the goal is not directly available.
        # We can approximate it: pos(ball_rel_goal) = pos(agent_rel_goal) - pos(agent_rel_ball)
        # Note: goal_team_1_rel_robot is agent_rel_goal
        goal_pos_rel_agent = info.get("goal_team_1_rel_robot", np.array([0,0,0]))
        ball_pos_rel_goal = goal_pos_rel_agent - ball_pos_rel_agent

        # Calculate distances
        dist_agent_to_ball = np.linalg.norm(agent_pos - ball_pos_rel_agent)
        dist_ball_to_goal = np.linalg.norm(ball_pos_rel_goal)

        # The potential function
        potential = -self.k1 * dist_agent_to_ball - self.k2 * dist_ball_to_goal
        return potential

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.previous_potential = self._calculate_potential(info)
        
        processed_obs = self.preprocessor.modify_state(obs, info).squeeze(0)
        return processed_obs.astype(np.float32), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        current_potential = self._calculate_potential(info)
        
        # PBRS formula
        shaped_reward = self.gamma * current_potential - self.previous_potential
        
        # Update potential for the next step
        self.previous_potential = current_potential
        
        # Add shaped reward to the original reward
        total_reward = reward + shaped_reward
        
        processed_obs = self.preprocessor.modify_state(obs, info).squeeze(0)
        
        return processed_obs.astype(np.float32), total_reward, terminated, truncated, info

# --- Custom Callback for Saving Best Models ---
class SaveBestModelCallback(BaseCallback):
    def __init__(self, save_path, save_prefix, verbose=0):
        super().__init__(verbose)
        self.save_path = save_path
        self.save_prefix = save_prefix
        self.best_mean_reward = -np.inf
        os.makedirs(save_path, exist_ok=True)

    def _on_step(self) -> bool:
        # Check if 'monitor' information is available in infos
        if 'episode' in self.training_env.get_attr('info')[0]:
            # In VecEnv, infos is a list of dicts. We check the first one.
            mean_reward = np.mean([info['episode']['r'] for info in self.locals['infos'] if 'episode' in info])
            
            if mean_reward > self.best_mean_reward:
                self.best_mean_reward = mean_reward
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

    # Wrapper arguments
    wrapper_kwargs = {
        'gamma': args.gamma,
        'k1': args.k1,
        'k2': args.k2
    }

    # Create the vectorized environment with the PBRS wrapper
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
        verbose=1,
        tensorboard_log=log_dir,
        policy_kwargs=dict(net_arch=[256, 128, 64]),
        learning_rate=args.lr,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        gamma=args.gamma,
        clip_range=args.clip_range,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )

    callback = SaveBestModelCallback(save_path=save_path, save_prefix=save_prefix, verbose=1)

    try:
        model.learn(total_timesteps=args.total_timesteps, callback=callback)
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
    finally:
        # --- Save Final Model ---
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
