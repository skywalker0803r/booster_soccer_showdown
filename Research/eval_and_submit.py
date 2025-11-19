# -*- coding: utf-8 -*-
# eval_and_submit.py

import torch
import numpy as np
import os
from sai_rl import SAIClient

# Import necessary classes and functions from main.py, td3_model.py, utils.py
from td3_model import TD3_FF
from utils import Preprocessor

# =================================================================
# 1. Configuration and helper functions (copied from main.py)
# =================================================================
MODEL_NAME = "Booster-TD3-PureCuriosity-v1" 
MODEL_PATH = "checkpoint_300k_20251119_214742.pth"
N_FEATURES = 45 # State dimension output by Preprocessor
NEURONS = [256, 256] 
LEARNING_RATE = 3e-4 # Required parameter for TD3_FF initialization

# Initialize SAIClient
sai = SAIClient(
    comp_id="booster-soccer-showdown" , 
    api_key="sai_LFcuaCZiqEkUbNVolQ3wbk5yU7H11jfv",
)
# Create an environment instance to get action space dimensions
env = sai.make_env() 
N_ACTIONS = env.action_space.shape[0]

# Map original policy output [-1, 1] to environment action space
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

# =================================================================
# 2. Load model
# =================================================================

def load_td3_model(model_path):
    """Load TD3 model weights"""
    if not os.path.exists(model_path):
        print(f"Error: Model file {model_path} not found. Please check path or filename.")
        print("Available model files:")
        pth_files = [f for f in os.listdir('.') if f.endswith('.pth')]
        for f in pth_files:
            print(f"  - {f}")
        return None

    # Initialize model architecture first
    td3_agent = TD3_FF(
        N_FEATURES, 
        env.action_space, 
        NEURONS, 
        torch.nn.functional.relu, 
        LEARNING_RATE
    )
    
    # Load weights
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    
    # Check if new format (contains model_state_dict)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        td3_agent.load_state_dict(checkpoint['model_state_dict'])
        print(f"Successfully loaded model weights (new format): {model_path}")
        print(f"   - Episode: {checkpoint.get('episode', 'Unknown')}")
        print(f"   - Best reward: {checkpoint.get('best_reward', 'Unknown')}")
        print(f"   - Timestep: {checkpoint.get('timestep', 'Unknown')}")
    else:
        # Old format, load directly
        td3_agent.load_state_dict(checkpoint)
        print(f"Successfully loaded model weights (old format): {model_path}")
    
    td3_agent.eval() # Set to evaluation mode
    
    # Close initialized environment instance
    env.close() 
    return td3_agent


# =================================================================
# 3. Execute operations
# =================================================================

def main_flow():
    """Main execution flow"""
    
    loaded_model = load_td3_model(MODEL_PATH)
    if loaded_model is None:
        return

    # --- Watch model performance (Watch) ---
    print("\n--- Watching model performance (sai.watch) ---")
    print("This will play the model running in the environment locally...")
    try:
        sai.watch(
            model=loaded_model,
            action_function=action_function,
            preprocessor_class=Preprocessor,
        )
        print("Watching ended.")
    except Exception as e:
        print(f"sai.watch execution failed: {e}")
    
    # --- Evaluate model performance (Benchmark) ---
    print("\n--- Evaluating model performance (sai.benchmark) ---")
    try:
        results = sai.benchmark(
            model=loaded_model.actor,
            action_function=action_function,
            preprocessor_class=Preprocessor,
        )
        print("\n=== Benchmark Results ===")
        print(results)
        print("=========================")
    except Exception as e:
        print(f"sai.benchmark execution failed: {e}")


    # --- Submit model (Submit) ---
    submit_prompt = input("\nDo you want to submit the model to the competition? (Enter y to submit): ")
    
    if submit_prompt.lower() == 'y':
        submission_name = input("Please enter the submission name (e.g., 'TD3_Final_Tuning'): ")
        print(f"--- Submitting model: {submission_name} ---")
        try:
            submission = sai.submit(
                name=submission_name,
                model=loaded_model,
                action_function=action_function,
                preprocessor_class=Preprocessor,
            )
            print("\n=== Submission Results ===")
            print(submission)
            print("==========================")
        except Exception as e:
            print(f"sai.submit execution failed: {e}")
    else:
        print("Model submission cancelled.")

if __name__ == "__main__":
    main_flow()