import sys
import os
import pathlib
import numpy as np
import torch
from functools import partial as bind

# Add DreamerV3 to path
dreamer_path = pathlib.Path(__file__).parent.parent / 'dreamerv3'
sys.path.insert(0, str(dreamer_path))
sys.path.insert(1, str(dreamer_path / 'dreamerv3'))

try:
    import elements
    import embodied
    import embodied.envs
    from dreamerv3.agent import Agent
    import ruamel.yaml as yaml
except ImportError as e:
    print(f"Warning: Could not import DreamerV3 dependencies: {e}")
    print("Make sure DreamerV3 is properly installed.")
    # Fallback imports
    import yaml as ruamel_yaml
    yaml = ruamel_yaml


class SAIEnvWrapper(embodied.Env):
    """Wrapper to make SAI environment compatible with DreamerV3"""
    
    def __init__(self, sai_env, preprocessor_class):
        self._env = sai_env
        self._preprocessor = preprocessor_class()
        self._done = True
        self._info = None
        self._episode_step = 0
        
    @property
    def obs_space(self):
        """Define observation space for DreamerV3"""
        # DreamerV3 expects a specific format
        spaces = {
            'vector': elements.Space(np.float32, (89,)),  # Our preprocessed features
            'reward': elements.Space(np.float32),
            'is_first': elements.Space(bool),
            'is_last': elements.Space(bool),
            'is_terminal': elements.Space(bool),
        }
        return spaces
    
    @property
    def act_space(self):
        """Define action space for DreamerV3"""
        # DreamerV3 expects named actions
        spaces = {
            'action': elements.Space(np.float32, (12,), -1.0, 1.0),  # Normalized actions
            'reset': elements.Space(bool),
        }
        return spaces
    
    def step(self, action):
        if action['reset'] or self._done:
            return self._reset()
            
        # Convert normalized action back to environment action
        normalized_action = action['action']
        env_action = self._denormalize_action(normalized_action)
        
        obs, reward, terminated, truncated, info = self._env.step(env_action)
        self._done = terminated or truncated
        self._info = info
        self._episode_step += 1
        
        # Preprocess observation
        processed_obs = self._preprocessor.modify_state(obs, info).squeeze()
        
        return {
            'vector': processed_obs.astype(np.float32),
            'reward': np.float32(reward),
            'is_first': False,
            'is_last': bool(self._done),
            'is_terminal': bool(terminated),
        }
    
    def _reset(self):
        obs, info = self._env.reset()
        self._done = False
        self._info = info
        self._episode_step = 0
        
        # Preprocess observation
        processed_obs = self._preprocessor.modify_state(obs, info).squeeze()
        
        return {
            'vector': processed_obs.astype(np.float32),
            'reward': np.float32(0.0),
            'is_first': True,
            'is_last': False,
            'is_terminal': False,
        }
    
    def _denormalize_action(self, normalized_action):
        """Convert normalized [-1, 1] action to environment action space"""
        low = self._env.action_space.low
        high = self._env.action_space.high
        return low + (high - low) * (normalized_action + 1) / 2
    
    def render(self):
        try:
            return self._env.render()
        except:
            return np.zeros((64, 64, 3), dtype=np.uint8)
    
    def close(self):
        self._env.close()


class DreamerV3Model:
    """PyTorch-like interface for DreamerV3 agent"""
    
    def __init__(self, sai_env, preprocessor_class, config_overrides=None):
        self.env_wrapper = SAIEnvWrapper(sai_env, preprocessor_class)
        self.preprocessor_class = preprocessor_class
        
        # Load DreamerV3 config
        config_path = dreamer_path / 'dreamerv3' / 'configs.yaml'
        configs = yaml.YAML(typ='safe').load(config_path.read_text())
        
        # Use a lightweight config for faster training
        self.config = elements.Config(configs['defaults'])
        
        # Apply optimizations for our task
        optimized_config = {
            'batch_size': 16,
            'batch_length': 32,
            'report_length': 16,
            'env.image': False,  # We don't use images
            'replay.size': 1e6,  # Smaller replay buffer
            'run.train_ratio': 32,
            'run.log_every': 300,
            'run.save_every': 1800,
            'agent.horizon': 15,  # Shorter planning horizon for faster training
            'agent.rssm.deter': 512,  # Smaller model
            'agent.rssm.stoch': 32,
            'agent.rssm.discrete': 32,
            'agent.enc.simple.depth': 2,  # Simpler encoder
            'agent.dec.simple.depth': 2,  # Simpler decoder
        }
        
        if config_overrides:
            optimized_config.update(config_overrides)
            
        for key, value in optimized_config.items():
            self.config = self.config.update({key: value})
        
        # Create agent
        obs_space = self.env_wrapper.obs_space
        act_space = {k: v for k, v in self.env_wrapper.act_space.items() if k != 'reset'}
        
        self.agent = Agent(obs_space, act_space, elements.Config(
            **self.config.agent,
            batch_size=self.config.batch_size,
            batch_length=self.config.batch_length,
            replay_context=self.config.replay_context,
        ))
        
        # Initialize agent state
        self.policy_state = None
        
    def __call__(self, state_tensor):
        """PyTorch-like forward pass for compatibility"""
        # Convert tensor to numpy if needed
        if hasattr(state_tensor, 'detach'):
            state = state_tensor.detach().cpu().numpy()
        else:
            state = state_tensor
            
        if len(state.shape) == 1:
            state = state[np.newaxis, :]
            
        # Create observation dict for DreamerV3
        obs = {
            'vector': state.astype(np.float32),
            'is_first': np.array([False] * len(state)),
            'is_last': np.array([False] * len(state)), 
            'is_terminal': np.array([False] * len(state)),
            'reward': np.array([0.0] * len(state)),
        }
        
        # Get action from DreamerV3
        if self.policy_state is None:
            self.policy_state = self.agent.init_policy(len(state))
            
        self.policy_state, action, _ = self.agent.policy(self.policy_state, obs, mode='train')
        
        # Return normalized action [-1, 1]
        return torch.FloatTensor(action['action'])
    
    def select_action(self, state):
        """DDPG-like interface for action selection"""
        action_tensor = self(state)
        return action_tensor.detach().cpu().numpy()
    
    def train(self, states, actions, rewards, next_states, dones, epochs=1):
        """Placeholder for DDPG-like training interface"""
        # DreamerV3 training is handled differently, this is for compatibility
        return 0.0, 0.0  # dummy losses
    
    def save_model(self, path):
        """Save the DreamerV3 model"""
        # This would need to be implemented with proper checkpointing
        pass
        
    def load_model(self, path):
        """Load the DreamerV3 model"""
        # This would need to be implemented with proper checkpointing
        pass


def create_dreamerv3_trainer(sai_env, preprocessor_class, config_overrides=None):
    """Create a DreamerV3 trainer that can be used with the existing training loop"""
    
    def make_env():
        return SAIEnvWrapper(sai_env, preprocessor_class)
    
    def make_agent():
        env = make_env()
        obs_space = env.obs_space
        act_space = {k: v for k, v in env.act_space.items() if k != 'reset'}
        env.close()
        
        # Load config
        config_path = dreamer_path / 'dreamerv3' / 'configs.yaml'
        configs = yaml.YAML(typ='safe').load(config_path.read_text())
        config = elements.Config(configs['defaults'])
        
        # Apply optimizations
        optimized_config = {
            'batch_size': 16,
            'batch_length': 32,
            'env.image': False,
            'replay.size': 5e5,
            'agent.horizon': 15,
            'agent.rssm.deter': 512,
            'agent.rssm.stoch': 32,
            'agent.rssm.discrete': 32,
        }
        
        if config_overrides:
            optimized_config.update(config_overrides)
            
        for key, value in optimized_config.items():
            config = config.update({key: value})
        
        return Agent(obs_space, act_space, elements.Config(
            **config.agent,
            batch_size=config.batch_size,
            batch_length=config.batch_length,
            replay_context=getattr(config, 'replay_context', 1),
        ))
    
    def make_replay():
        config_path = dreamer_path / 'dreamerv3' / 'configs.yaml'
        configs = yaml.YAML(typ='safe').load(config_path.read_text())
        config = elements.Config(configs['defaults'])
        
        optimized_config = {
            'replay.size': 5e5,
            'replay.chunksize': 1024,
        }
        
        if config_overrides:
            optimized_config.update(config_overrides)
            
        for key, value in optimized_config.items():
            config = config.update({key: value})
        
        return embodied.replay.Replay(
            make_env().obs_space,
            make_env().act_space,
            **config.replay,
        )
    
    return make_env, make_agent, make_replay