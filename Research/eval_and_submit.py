# -*- coding: utf-8 -*-
# eval_and_submit.py
# Updated to support PPOCMA model

import torch
import numpy as np
import os
import glob
import copy
from sai_rl import SAIClient

# Import the correct model and utility classes
from ppo_cma_model import PPOCMA
from utils import Preprocessor

# =================================================================
# 1. Configuration (Synced with main.py)
# =================================================================
# Find the latest best model automatically
try:
    list_of_files = glob.glob('best_Booster-PPOCMA-A100-*.pth')
    latest_file = max(list_of_files, key=os.path.getctime)
    MODEL_PATH = latest_file
    print(f"✅ Automatically found the latest best model: {MODEL_PATH}")
except (ValueError, FileNotFoundError):
    MODEL_PATH = "best_model.pth" # Fallback
    print(f"⚠️ Could not find a model automatically. Using fallback: {MODEL_PATH}")


# --- Hyperparameters (Must match the trained model's architecture) ---
N_FEATURES = 45 
# These hyperparameters define the model structure and must match main.py
BUFFER_CAPACITY = 8192
BATCH_SIZE = 1024
NEURONS = [512, 512, 256]
PPO_EPOCHS = 15
CMA_POPULATION_SIZE = 64
# --- The following params are for agent initialization, but less critical for eval ---
LEARNING_RATE_ACTOR = 3e-4
LEARNING_RATE_CRITIC = 1e-3
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_EPSILON = 0.2
ENTROPY_COEF = 0.01
MAX_GRAD_NORM = 0.5
CMA_SIGMA = 0.1
CMA_UPDATE_FREQ = 10

# Initialize SAIClient and environment to get action space info
sai = SAIClient(
    comp_id="booster-soccer-showdown",
    api_key="sai_LFcuaCZiqEkUbNVolQ3wbk5yU7H11jfv",
)
env = sai.make_env() 
N_ACTIONS = env.action_space.shape[0]

# Action function to scale model output
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
# 2. Load PPOCMA model
# =================================================================

class ActorWrapper(torch.nn.Module):
    """
    A wrapper for the ActorNetwork to ensure its forward method returns a single
    tensor (the mean), which is expected by the SAI evaluation tools.
    """
    def __init__(self, actor_network):
        super().__init__()
        self.actor_network = actor_network

    def forward(self, state):
        # The sai.benchmark tool will pass the state tensor here.
        # We call the original actor but only return the 'mean' part.
        mean, _ = self.actor_network(state)
        return mean

def load_ppocma_model(model_path):
    """Load PPOCMA model weights"""
    if not os.path.exists(model_path):
        print(f"❌ Error: Model file '{model_path}' not found.")
        return None

    # Use CPU for evaluation by default, it's generally sufficient
    device = torch.device('cpu')
    print(f"Evaluating on device: {device}")

    # 1. Initialize the agent with the same architecture as during training
    ppo_cma_agent = PPOCMA(
        state_dim=N_FEATURES,
        action_dim=N_ACTIONS,
        hidden_dims=NEURONS,
        lr_actor=LEARNING_RATE_ACTOR,
        lr_critic=LEARNING_RATE_CRITIC,
        gamma=GAMMA,
        gae_lambda=GAE_LAMBDA,
        clip_epsilon=CLIP_EPSILON,
        entropy_coef=ENTROPY_COEF,
        max_grad_norm=MAX_GRAD_NORM,
        ppo_epochs=PPO_EPOCHS,
        batch_size=BATCH_SIZE,
        buffer_capacity=BUFFER_CAPACITY,
        cma_population_size=CMA_POPULATION_SIZE,
        cma_sigma=CMA_SIGMA,
        cma_update_freq=CMA_UPDATE_FREQ
    )

    try:
        # 2. Load the checkpoint
        checkpoint = torch.load(model_path, map_location=device)
        
        # 3. Load weights into the agent
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            ppo_cma_agent.load_state_dict(checkpoint['model_state_dict'])
            print(f"✅ Successfully loaded model weights from: {model_path}")
            print(f"   - Trained for Episode: {checkpoint.get('episode', 'N/A')}")
            print(f"   - Best recorded reward: {checkpoint.get('best_reward', 'N/A')}")
        else:
            # Fallback for older format if necessary
            ppo_cma_agent.load_state_dict(checkpoint)
            print(f"✅ Successfully loaded model weights (direct state dict): {model_path}")
        
        # 4. Set the actor network to evaluation mode
        ppo_cma_agent.actor.eval()
        
        env.close() 
        return ppo_cma_agent

    except Exception as e:
        print(f"❌ Error loading model: {e}")
        env.close()
        return None


# =================================================================
# 3. Execute operations
# =================================================================

def main_flow():
    """Main execution flow"""
    
    # Load the PPOCMA agent
    loaded_agent = load_ppocma_model(MODEL_PATH)
    if loaded_agent is None:
        return

    # Wrap the actor network to conform to the evaluation tool's interface
    evaluation_model = ActorWrapper(loaded_agent.actor)

    # --- Watch model performance (Watch) ---
    print("\n--- 👁️ Watching model performance (sai.watch) ---")
    print("Press Ctrl+C in the console to stop watching.")
    try:
        sai.watch(
            model=evaluation_model,
            action_function=action_function,
            preprocessor_class=Preprocessor,
        )
        print("Watching ended.")
    except Exception as e:
        print(f"❌ sai.watch execution failed: {e}")
    
    # --- Evaluate model performance (Benchmark) ---
    print("\n--- 📊 Evaluating model performance (sai.benchmark) ---")
    try:
        results = sai.benchmark(
            model=evaluation_model,
            action_function=action_function,
            preprocessor_class=Preprocessor,
        )
        print("\n=== Benchmark Results ===")
        print(results)
        print("=========================")
    except Exception as e:
        print(f"❌ sai.benchmark execution failed: {e}")


    # --- Submit model (Submit) ---
    submit_prompt = input("\nDo you want to submit this model to the competition? (y/n): ")
    
    if submit_prompt.lower() == 'y':
        submission_name = input("Please enter a name for this submission (e.g., 'PPOCMA_Curiosity_Run1'): ")
        if not submission_name:
            submission_name = f"PPOCMA_{os.path.basename(MODEL_PATH)}"

        print(f"--- 🚀 Submitting model: {submission_name} ---")
        try:
            # For submission, it's also recommended to submit the wrapped actor model
            submission = sai.submit(
                name=submission_name,
                model=evaluation_model,
                action_function=action_function,
                preprocessor_class=Preprocessor,
            )
            print("\n=== Submission Results ===")
            print(submission)
            print("==========================")
        except Exception as e:
            print(f"❌ sai.submit execution failed: {e}")
    else:
        print("Model submission cancelled.")

if __name__ == "__main__":
    main_flow()
