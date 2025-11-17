"""
Local Model Watch and Submit - TensorBoard + Watch + Submit
After downloading model files from Colab, complete locally:
1. TensorBoard monitoring of training process
2. Watch actual model performance
3. Decide whether to submit to leaderboard
"""

from sai_rl import SAIClient
from stable_baselines3 import PPO
import numpy as np
import os
import subprocess
import threading
import time
import webbrowser

# Your model file path (modify this to actual downloaded model path)
MODEL_PATH = "./saved_models/ppo_standalone_20251117_105737.zip"  # Modify this!

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

def start_tensorboard():
    """Start TensorBoard and open browser"""
    if not os.path.exists("./runs"):
        print("WARNING: runs/ folder not found, skipping TensorBoard")
        return None
    
    try:
        print("Starting TensorBoard...")
        # Start TensorBoard in background
        proc = subprocess.Popen(
            ["tensorboard", "--logdir=./runs", "--port=6006"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        # Wait for TensorBoard to start
        time.sleep(3)
        
        # Open browser
        print("Opening browser: http://localhost:6006")
        webbrowser.open("http://localhost:6006")
        
        return proc
        
    except FileNotFoundError:
        print("ERROR: tensorboard command not found, please install: pip install tensorboard")
        return None
    except Exception as e:
        print(f"WARNING: TensorBoard start failed: {e}")
        return None

def ask_user_submission(model, sai):
    """Ask user whether to submit model"""
    while True:
        print("\n" + "="*50)
        print("After watching model performance, do you want to submit to leaderboard?")
        print("   y - Yes, submit model")
        print("   n - No, don't submit")
        print("   r - Watch again")
        
        choice = input("Please choose (y/n/r): ").lower().strip()
        
        if choice == 'y':
            print("\nSubmitting model to leaderboard...")
            try:
                # Ask for model name
                model_name = input("Enter model name (default: My PPO Model): ").strip()
                if not model_name:
                    model_name = "My PPO Model"
                
                sai.submit(model_name, model, action_function, Preprocessor)
                print("SUCCESS: Model submitted successfully!")
                return True
            except Exception as e:
                print(f"ERROR: Submission failed: {e}")
                return False
                
        elif choice == 'n':
            print("OK, not submitting model")
            return False
            
        elif choice == 'r':
            print("\nStarting watch again...")
            try:
                sai.watch(model, action_function, Preprocessor)
            except KeyboardInterrupt:
                print("\nWatch stopped")
            continue
            
        else:
            print("ERROR: Invalid choice, please enter y, n, or r")

def action_function(policy):
    """Action function compatible with SAI submission format"""
    expected_bounds = [-1, 1]
    action_percent = (policy - expected_bounds[0]) / (expected_bounds[1] - expected_bounds[0])
    bounded_percent = np.minimum(np.maximum(action_percent, 0), 1)
    # Use official action space bounds from docs/About.md line 55
    action_low = np.array([-45,-45,-30,-65,-24,-15,-45,-45,-30,-65,-24,-15])
    action_high = np.array([45,45,30,65,24,15,45,45,30,65,24,15])
    return action_low + (action_high - action_low) * bounded_percent

def main():
    global env
    
    print("Local Model Summary - TensorBoard + Watch + Submit")
    print("=" * 60)
    
    # Check if model file exists
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: Model file not found: {MODEL_PATH}")
        print("\nPlease follow these steps:")
        print("1. Download saved_models/ folder from Colab")
        print("2. Modify MODEL_PATH variable in this script")
        print("3. Ensure model file path is correct")
        print(f"\nExample filename: ppo_standalone_20241117_105737.zip")
        
        # Show available model files
        if os.path.exists("./saved_models"):
            print(f"\nFound model files:")
            for file in os.listdir("./saved_models"):
                if file.endswith(".zip"):
                    print(f"   - {os.path.join('./saved_models', file)}")
        return
    
    # Start TensorBoard
    tensorboard_proc = start_tensorboard()
    
    try:
        # Initialize SAI client
        sai = SAIClient(comp_id="booster-soccer-showdown", api_key="sai_LFcuaCZiqEkUbNVolQ3wbk5yU7H11jfv")
        print("SUCCESS: SAI client initialized")
        
        # Create environment
        env = sai.make_env(use_custom_eval=False)
        print("SUCCESS: Environment created")
        
        # Load trained model
        print(f"Loading model: {MODEL_PATH}")
        model = PPO.load(MODEL_PATH)
        print("SUCCESS: Model loaded")
        
        print("\n" + "="*60)
        print("Now you can:")
        print("1. View TensorBoard (browser opened automatically)")
        print("2. Watch model actual performance")
        print("3. Decide whether to submit to leaderboard")
        print("="*60)
        
        # Ask if user wants to start watching
        input("\nPress Enter to start watching model performance...")
        
        # Start watching model
        print("\nStarting model watch...")
        print("   Press Ctrl+C to stop watching")
        
        try:
            sai.watch(model, action_function, Preprocessor,use_custom_eval=False)
        except KeyboardInterrupt:
            print("\nWatch stopped")
        
        # Ask about submission
        ask_user_submission(model, sai)
        
    except Exception as e:
        print(f"\nERROR: {e}")
        print("\nPossible solutions:")
        print("1. Ensure all required dependencies are installed")
        print("2. Check network connection")
        print("3. Verify API key is correct")
        print("4. Ensure model file is complete")
    
    finally:
        # Close TensorBoard
        if tensorboard_proc:
            print("\nClosing TensorBoard...")
            tensorboard_proc.terminate()
        
        # Close environment
        if 'env' in globals():
            env.close()
            print("SUCCESS: Environment closed")
        
        print("\nLocal summary complete! Thanks for using!")

if __name__ == "__main__":
    main()