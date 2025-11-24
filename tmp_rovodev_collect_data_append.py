"""
Modified collect_data.py that can append to existing data
"""
import os
import sys

# Make repo root importable without absolute paths
repo_root = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(__file__)), ".."))
sys.path.append(repo_root)

import time
import argparse
import sai_mujoco  # noqa: F401
import gymnasium as gym
import numpy as np
from booster_control.se3_keyboard import Se3Keyboard
from booster_control.t1_utils import LowerT1JoyStick
from imitation_learning.scripts.preprocessor import Preprocessor

def get_task_one_hot(env_name):
    if "GoaliePenaltyKick" in env_name:
        task_one_hot = np.array([1.0, 0.0, 0.0])
    elif "ObstaclePenaltyKick" in env_name:
        task_one_hot = np.array([0.0, 1.0, 0.0])
    elif "KickToTarget" in env_name:
        task_one_hot = np.array([0.0, 0.0, 1.0])
    return task_one_hot

def load_existing_data(dataset_directory):
    """Load existing data if file exists, otherwise return empty dataset"""
    try:
        existing_data = np.load(dataset_directory, allow_pickle=True)
        dataset = {
            "observations": list(existing_data["observations"]),
            "actions": list(existing_data["actions"]),
            "done": list(existing_data["done"])
        }
        print(f"已載入現有數據: {len(dataset['observations'])} 個時間步")
        
        # 計算現有episode數
        existing_episodes = np.sum(existing_data["done"])
        print(f"現有episode數: {existing_episodes}")
        return dataset, existing_episodes
    except FileNotFoundError:
        print("未找到現有數據檔，將創建新的數據集")
        return {
            "observations": [],
            "actions": [],
            "done": []
        }, 0

def teleop(env_name: str = "LowerT1GoaliePenaltyKick-v0", pos_sensitivity:float = 0.1*10, rot_sensitivity:float = 1.5*10, dataset_directory = "./data/dataset_kick.npz"):

    # Load existing data
    dataset, existing_episode_count = load_existing_data(dataset_directory)
    
    env = gym.make(env_name, render_mode="human")
    lower_t1_robot = LowerT1JoyStick(env.unwrapped)
    preprocessor = Preprocessor()

    # Initialize the T1 SE3 keyboard controller with the viewer
    keyboard_controller = Se3Keyboard(renderer=env.unwrapped.mujoco_renderer, pos_sensitivity=pos_sensitivity, rot_sensitivity=rot_sensitivity)

    # Set the reset environment callback
    keyboard_controller.set_reset_env_callback(env.reset)

    # Print keyboard control instructions
    print("\nKeyboard Controls:")
    print(keyboard_controller)

    # Main teleoperation loop
    episode_count = existing_episode_count  # 從現有episode數開始計算
    task_one_hot = get_task_one_hot(env_name)
    
    while True:
        # Reset environment for new episode
        terminated = truncated = False
        observation, info = env.reset()
        episode_count += 1

        episode = {
            "observations": [],
            "actions": [],
            "done": []
        }

        print(f"\nStarting episode {episode_count}")
        # Episode loop  
        while not (terminated or truncated):

            preprocessed_observation = preprocessor.modify_state(observation.copy(), info.copy(), task_one_hot)
            # Get keyboard input and apply it directly to the environment
            if keyboard_controller.should_quit():
                print(f"\n[INFO] ESC pressed — exiting teleop.")
                print(f"總共收集了 {len(dataset['observations'])} 個時間步")
                print(f"總共 {episode_count} 個episodes")
                np.savez(dataset_directory, observations=dataset["observations"], actions=dataset["actions"], done = dataset["done"])
                env.close()
                return
            
            command = keyboard_controller.advance()
            ctrl, actions = lower_t1_robot.get_actions(command, observation, info)

            episode["observations"].append(preprocessed_observation)
            episode["actions"].append(actions)
            
            observation, reward, terminated, truncated, info = env.step(ctrl)
            episode["done"].append(terminated)

            if terminated or truncated:
                break
        
        # Append episode data to dataset
        dataset["observations"].extend(episode["observations"])
        dataset["actions"].extend(episode["actions"])
        dataset["done"].extend(episode["done"])

        # Print episode result
        if info.get("success", True):
            print(f"Episode {episode_count} completed successfully!")
        else:
            print(f"Episode {episode_count} completed without success")
        
        print(f"目前總步數: {len(dataset['observations'])}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Teleoperate T1 robot in a gymnasium environment with data appending.")
    parser.add_argument("--env", type=str, default="LowerT1GoaliePenaltyKick-v0", help="The environment to teleoperate.")
    parser.add_argument("--pos_sensitivity", type=float, default=0.1*10, help="SE3 Keyboard position sensitivity.")
    parser.add_argument("--rot_sensitivity", type=float, default=0.5*10, help="SE3 Keyboard rotation sensitivity.")
    parser.add_argument("--data_set_directory", type=str, default="./data/dataset_kick.npz", help="Dataset file path.")

    args = parser.parse_args()

    teleop(args.env, args.pos_sensitivity, args.rot_sensitivity, args.data_set_directory)