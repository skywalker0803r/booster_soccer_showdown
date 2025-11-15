import torch.nn.functional as F
import numpy as np
import sys
import pathlib
import os
from functools import partial as bind

from sai_rl import SAIClient
from dreamerv3_adapter import DreamerV3Model, SAIEnvWrapper, create_dreamerv3_trainer

# Add DreamerV3 to path
dreamer_path = pathlib.Path(__file__).parent.parent / 'dreamerv3'
sys.path.insert(0, str(dreamer_path))
sys.path.insert(1, str(dreamer_path / 'dreamerv3'))

import elements
import embodied

## Initialize the SAI client
sai = SAIClient(comp_id="booster-soccer-showdown", api_key="sai_LFcuaCZiqEkUbNVolQ3wbk5yU7H11jfv")

## Make the environment
env = sai.make_env()

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

## Create the DreamerV3 model
print("Creating DreamerV3 model...")
model = DreamerV3Model(
    env, 
    Preprocessor,
    config_overrides={
        'run.steps': 2000000,  # 2M steps should be enough for good performance
        'run.train_ratio': 32,  # High training ratio for sample efficiency
        'batch_size': 16,       # Reasonable batch size
        'batch_length': 32,     # Sequence length for world model
        'agent.horizon': 15,    # Planning horizon
        'run.log_every': 1000,  # Log every 1000 steps
        'run.save_every': 10000, # Save every 10000 steps
    }
)

## Define an action function (DreamerV3 outputs normalized actions)
def action_function(policy):
    # DreamerV3 already outputs actions in [-1, 1], but we need to map to env action space
    expected_bounds = [-1, 1]
    action_percent = (policy - expected_bounds[0]) / (
        expected_bounds[1] - expected_bounds[0]
    )
    bounded_percent = np.minimum(np.maximum(action_percent, 0), 1)
    return (
        env.action_space.low
        + (env.action_space.high - env.action_space.low) * bounded_percent
    )

def dreamerv3_training_loop():
    """Custom training loop for DreamerV3"""
    print("Starting DreamerV3 training...")
    
    # Create DreamerV3 components
    make_env, make_agent, make_replay = create_dreamerv3_trainer(env, Preprocessor)
    
    # Create logdir
    logdir = pathlib.Path(f"./dreamer_logs_{elements.timestamp()}")
    logdir.mkdir(exist_ok=True)
    
    # Initialize components
    agent = make_agent()
    replay = make_replay()
    env_wrapper = make_env()
    
    print(f"Observation space: {env_wrapper.obs_space}")
    print(f"Action space: {env_wrapper.act_space}")
    
    # Simple training parameters
    max_episodes = 1000
    steps_per_episode = 1000
    batch_size = 16
    batch_length = 32
    update_frequency = 4
    
    total_steps = 0
    episode_count = 0
    
    # Initialize agent states
    policy_state = agent.init_policy(1)
    train_state = agent.init_train(batch_size)
    
    print(f"Training for {max_episodes} episodes...")
    
    for episode in range(max_episodes):
        obs = env_wrapper._reset()
        episode_reward = 0
        episode_steps = 0
        
        # Convert single obs to batch format
        obs_batch = {k: v[np.newaxis, ...] if k != 'is_first' else np.array([True]) 
                    for k, v in obs.items()}
        
        for step in range(steps_per_episode):
            # Get action from agent
            policy_state, action, _ = agent.policy(policy_state, obs_batch, mode='train')
            
            # Step environment
            next_obs = env_wrapper.step(action)
            
            # Store transition in replay buffer
            transition = {**obs, **action, **next_obs}
            replay.add(transition, worker=0)
            
            episode_reward += next_obs['reward']
            episode_steps += 1
            total_steps += 1
            
            # Training step
            if len(replay) >= batch_size * batch_length and total_steps % update_frequency == 0:
                # Sample batch and train
                batch = replay.sample(batch_size, batch_length)
                train_state, outs, mets = agent.train(train_state, batch)
                
                if total_steps % 1000 == 0:
                    print(f"Episode {episode}, Step {total_steps}, Reward: {episode_reward:.3f}")
                    for key, value in mets.items():
                        if not key.startswith('timer'):
                            print(f"  {key}: {value:.4f}")
            
            obs = next_obs
            obs_batch = {k: v[np.newaxis, ...] if k != 'is_first' else np.array([False]) 
                        for k, v in obs.items()}
            
            if next_obs['is_last']:
                break
        
        episode_count += 1
        print(f"Episode {episode} completed: Reward = {episode_reward:.3f}, Steps = {episode_steps}")
        
        # Save checkpoint periodically
        if episode % 50 == 0:
            print(f"Saving checkpoint at episode {episode}")
            # Here you would implement proper checkpointing
    
    env_wrapper.close()
    print("Training completed!")
    return agent

# Train with DreamerV3
print("=== Training with DreamerV3 ===")
trained_agent = dreamerv3_training_loop()

# For compatibility with SAI evaluation, we need to wrap the trained agent
class DreamerV3Wrapper:
    def __init__(self, agent, preprocessor_class):
        self.agent = agent
        self.preprocessor = preprocessor_class()
        self.policy_state = agent.init_policy(1)
    
    def __call__(self, state_tensor):
        """PyTorch-like interface"""
        if hasattr(state_tensor, 'detach'):
            state = state_tensor.detach().cpu().numpy()
        else:
            state = state_tensor
            
        if len(state.shape) == 1:
            state = state[np.newaxis, :]
            
        obs = {
            'vector': state.astype(np.float32),
            'is_first': np.array([False]),
            'is_last': np.array([False]), 
            'is_terminal': np.array([False]),
            'reward': np.array([0.0]),
        }
        
        self.policy_state, action, _ = self.agent.policy(
            self.policy_state, obs, mode='eval'
        )
        
        return torch.FloatTensor(action['action'])
    
    def select_action(self, state):
        """DDPG-like interface"""
        action_tensor = self(state)
        return action_tensor.detach().cpu().numpy()

# Wrap the trained agent for compatibility
model = DreamerV3Wrapper(trained_agent, Preprocessor)

print("=== Starting Evaluation ===")

## Benchmark the model locally
sai.benchmark(model, action_function, Preprocessor)

## Submit to leaderboard
print("=== Submitting to Leaderboard ===")
sai.submit("Vedanta_DreamerV3", model, action_function, Preprocessor)

print("=== Training and Submission Complete ===")