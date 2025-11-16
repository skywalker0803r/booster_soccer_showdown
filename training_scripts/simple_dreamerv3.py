"""
Simplified DreamerV3 implementation for the soccer competition
This version focuses on the core world model and policy learning concepts
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque
import random


class RSSMCore(nn.Module):
    """Simplified RSSM (Recurrent State Space Model) - core of DreamerV3"""
    
    def __init__(self, obs_dim, action_dim, hidden_dim=512, stoch_dim=32, discrete_dim=32):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.stoch_dim = stoch_dim
        self.discrete_dim = discrete_dim
        
        # Encoder: obs -> latent representation
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
        
        # Recurrent transition model
        self.rnn = nn.GRUCell(hidden_dim + action_dim, hidden_dim)
        
        # Stochastic state prediction
        self.prior_net = nn.Linear(hidden_dim, stoch_dim * discrete_dim)
        self.posterior_net = nn.Linear(hidden_dim * 2, stoch_dim * discrete_dim)
        
        # Decoder: latent -> obs reconstruction
        decoder_input_dim = hidden_dim + stoch_dim * discrete_dim
        self.obs_decoder = nn.Sequential(
            nn.Linear(decoder_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, obs_dim)
        )
        
        # Reward prediction
        self.reward_predictor = nn.Sequential(
            nn.Linear(decoder_input_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def encode_obs(self, obs):
        """Encode observation to latent representation"""
        return self.obs_encoder(obs)
    
    def get_stochastic_state(self, logits):
        """Sample stochastic state from categorical distribution"""
        # Reshape logits for categorical sampling
        logits = logits.reshape(-1, self.stoch_dim, self.discrete_dim)
        # Sample from categorical distribution
        dist = torch.distributions.Categorical(logits=logits)
        stoch_discrete = dist.sample()
        # Convert to one-hot
        stoch_onehot = F.one_hot(stoch_discrete, self.discrete_dim).float()
        return stoch_onehot.reshape(-1, self.stoch_dim * self.discrete_dim)
    
    def transition(self, prev_state, action):
        """Predict next deterministic state and stochastic state prior"""
        # Combine previous state with action
        rnn_input = torch.cat([prev_state['deter'], action], dim=-1)
        
        # Update deterministic state
        deter = self.rnn(rnn_input, prev_state['deter'])
        
        # Predict stochastic state (prior)
        prior_logits = self.prior_net(deter)
        stoch = self.get_stochastic_state(prior_logits)
        
        return {
            'deter': deter,
            'stoch': stoch,
            'prior_logits': prior_logits
        }
    
    def observe(self, obs, prev_state, action):
        """Update state with observation (posterior)"""
        # Get deterministic state from transition
        transition_state = self.transition(prev_state, action)
        
        # Encode observation
        obs_embed = self.encode_obs(obs)
        
        # Compute posterior stochastic state
        posterior_input = torch.cat([transition_state['deter'], obs_embed], dim=-1)
        posterior_logits = self.posterior_net(posterior_input)
        stoch_posterior = self.get_stochastic_state(posterior_logits)
        
        state = {
            'deter': transition_state['deter'],
            'stoch': stoch_posterior,
            'prior_logits': transition_state['prior_logits'],
            'posterior_logits': posterior_logits
        }
        
        return state
    
    def decode_obs(self, state):
        """Decode latent state to observation"""
        latent = torch.cat([state['deter'], state['stoch']], dim=-1)
        return self.obs_decoder(latent)
    
    def predict_reward(self, state):
        """Predict reward from latent state"""
        latent = torch.cat([state['deter'], state['stoch']], dim=-1)
        return self.reward_predictor(latent)
    
    def init_state(self, batch_size):
        """Initialize state"""
        device = next(self.parameters()).device
        return {
            'deter': torch.zeros(batch_size, self.hidden_dim, device=device),
            'stoch': torch.zeros(batch_size, self.stoch_dim * self.discrete_dim, device=device)
        }


class Actor(nn.Module):
    """Policy network for continuous control"""
    
    def __init__(self, state_dim, action_dim, hidden_dim=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
    
    def forward(self, state):
        latent = torch.cat([state['deter'], state['stoch']], dim=-1)
        return torch.tanh(self.net(latent))


class Critic(nn.Module):
    """Value function network"""
    
    def __init__(self, state_dim, hidden_dim=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, state):
        latent = torch.cat([state['deter'], state['stoch']], dim=-1)
        return self.net(latent)


class SimpleDreamerV3(nn.Module):
    """Simplified DreamerV3 agent"""
    
    def __init__(self, obs_dim, action_dim, hidden_dim=512, stoch_dim=32, discrete_dim=32):
        super().__init__()
        
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.state_dim = hidden_dim + stoch_dim * discrete_dim
        
        # World model components
        self.rssm = RSSMCore(obs_dim, action_dim, hidden_dim, stoch_dim, discrete_dim)
        
        # Policy components
        self.actor = Actor(self.state_dim, action_dim, hidden_dim)
        self.critic = Critic(self.state_dim, hidden_dim)
        
        # Optimizers
        self.world_model_optimizer = torch.optim.Adam(self.rssm.parameters(), lr=3e-4)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=8e-5)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=8e-5)
        
        # Target critic for stability
        self.target_critic = Critic(self.state_dim, hidden_dim)
        self.target_critic.load_state_dict(self.critic.state_dict())
        
        self.gamma = 0.99
        self.tau = 0.005
        self.horizon = 15  # Planning horizon
        
    def encode_sequence(self, obs_seq, action_seq):
        """Encode a sequence of observations and actions"""
        batch_size, seq_len = obs_seq.shape[:2]
        
        # Initialize state
        state = self.rssm.init_state(batch_size)
        states = []
        
        for t in range(seq_len):
            if t == 0:
                # First step: encode observation directly
                obs_embed = self.rssm.encode_obs(obs_seq[:, t])
                posterior_input = torch.cat([state['deter'], obs_embed], dim=-1)
                posterior_logits = self.rssm.posterior_net(posterior_input)
                stoch = self.rssm.get_stochastic_state(posterior_logits)
                state = {
                    'deter': state['deter'],
                    'stoch': stoch,
                    'prior_logits': self.rssm.prior_net(state['deter']),
                    'posterior_logits': posterior_logits
                }
            else:
                # Subsequent steps: observe with previous action
                state = self.rssm.observe(obs_seq[:, t], state, action_seq[:, t-1])
            
            states.append(state)
        
        return states
    
    def imagine_sequence(self, init_state, actor, horizon):
        """Imagine future trajectories using the world model"""
        states = [init_state]
        actions = []
        
        state = init_state
        for _ in range(horizon):
            # Sample action from policy
            action = actor(state)
            actions.append(action)
            
            # Predict next state
            state = self.rssm.transition(state, action)
            states.append(state)
        
        return states, actions
    
    def compute_world_model_loss(self, obs_seq, action_seq, reward_seq):
        """Compute world model loss (reconstruction + prediction)"""
        batch_size, seq_len = obs_seq.shape[:2]
        
        # Encode sequence
        states = self.encode_sequence(obs_seq, action_seq)
        
        total_loss = 0
        reconstruction_loss = 0
        reward_loss = 0
        kl_loss = 0
        
        for t, state in enumerate(states):
            # Observation reconstruction loss
            obs_recon = self.rssm.decode_obs(state)
            reconstruction_loss += F.mse_loss(obs_recon, obs_seq[:, t])
            
            # Reward prediction loss
            if t < len(states) - 1:  # No reward for last state
                reward_pred = self.rssm.predict_reward(state)
                reward_loss += F.mse_loss(reward_pred.squeeze(), reward_seq[:, t])
            
            # KL divergence between prior and posterior
            if 'posterior_logits' in state:
                # Reshape logits for categorical distribution
                prior_logits = state['prior_logits'].reshape(-1, self.rssm.stoch_dim, self.rssm.discrete_dim)
                posterior_logits = state['posterior_logits'].reshape(-1, self.rssm.stoch_dim, self.rssm.discrete_dim)
                
                prior_dist = torch.distributions.Categorical(logits=prior_logits)
                posterior_dist = torch.distributions.Categorical(logits=posterior_logits)
                kl_loss += torch.distributions.kl_divergence(posterior_dist, prior_dist).mean()
        
        reconstruction_loss /= len(states)
        reward_loss /= (len(states) - 1)
        kl_loss /= len(states)
        
        total_loss = reconstruction_loss + reward_loss + kl_loss
        
        return total_loss, reconstruction_loss, reward_loss, kl_loss
    
    def compute_actor_loss(self, init_states):
        """Compute actor loss using imagined trajectories"""
        # Imagine trajectories
        imag_states, imag_actions = self.imagine_sequence(init_states, self.actor, self.horizon)
        
        # Compute values for imagined states (detached to avoid gradient conflicts)
        values = []
        with torch.no_grad():
            for state in imag_states:
                value = self.critic(state)
                values.append(value)
        
        # Compute rewards for imagined trajectories
        rewards = []
        for i in range(len(imag_states) - 1):
            reward = self.rssm.predict_reward(imag_states[i])
            rewards.append(reward)
        
        # Compute returns using lambda returns
        returns = self._compute_lambda_returns(rewards, values, self.gamma, lambda_=0.95)
        
        # Actor loss (policy gradient)
        advantages = returns[:-1] - torch.stack(values[:-1], dim=0).squeeze(-1)
        actor_loss = -(advantages.detach() * torch.stack([a.sum(dim=-1) for a in imag_actions], dim=0)).mean()
        
        return actor_loss
    
    def compute_critic_loss(self, init_states):
        """Compute critic loss using imagined trajectories"""
        # Imagine trajectories (detached to avoid gradient conflicts)
        with torch.no_grad():
            imag_states, imag_actions = self.imagine_sequence(init_states, self.actor, self.horizon)
        
        # Compute values for imagined states
        values = []
        for state in imag_states:
            value = self.critic(state)
            values.append(value)
        
        # Compute rewards for imagined trajectories
        rewards = []
        for i in range(len(imag_states) - 1):
            with torch.no_grad():
                reward = self.rssm.predict_reward(imag_states[i])
            rewards.append(reward)
        
        # Compute returns using lambda returns
        with torch.no_grad():
            returns = self._compute_lambda_returns(rewards, [v.detach() for v in values], self.gamma, lambda_=0.95)
        
        # Critic loss
        critic_targets = returns[:-1]  # Exclude last value
        critic_preds = torch.stack(values[:-1], dim=0).squeeze(-1)
        critic_loss = F.mse_loss(critic_preds, critic_targets)
        
        return critic_loss
    
    def _compute_lambda_returns(self, rewards, values, gamma, lambda_=0.95):
        """Compute lambda returns for policy learning"""
        returns = []
        last_value = values[-1].squeeze(-1)
        
        for t in reversed(range(len(rewards))):
            reward = rewards[t].squeeze(-1)
            value = values[t].squeeze(-1)
            next_value = values[t + 1].squeeze(-1)
            
            delta = reward + gamma * next_value - value
            last_value = value + delta + gamma * lambda_ * (last_value - next_value)
            returns.insert(0, last_value)
        
        returns.append(values[-1].squeeze(-1))  # Add final value
        return torch.stack(returns, dim=0)
    
    def update_target_critic(self):
        """Soft update of target critic"""
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
    
    def select_action(self, obs, state=None):
        """Select action for a single observation"""
        with torch.no_grad():
            device = next(self.parameters()).device
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
            
            if state is None:
                # Initialize state for first observation
                state = self.rssm.init_state(1)
                # Encode first observation
                obs_embed = self.rssm.encode_obs(obs_tensor)
                posterior_input = torch.cat([state['deter'], obs_embed], dim=-1)
                posterior_logits = self.rssm.posterior_net(posterior_input)
                stoch = self.rssm.get_stochastic_state(posterior_logits)
                state = {
                    'deter': state['deter'],
                    'stoch': stoch
                }
            else:
                # Update state with new observation (using previous action would be better but we don't have it here)
                obs_embed = self.rssm.encode_obs(obs_tensor)
                posterior_input = torch.cat([state['deter'], obs_embed], dim=-1)
                posterior_logits = self.rssm.posterior_net(posterior_input)
                stoch = self.rssm.get_stochastic_state(posterior_logits)
                state = {
                    'deter': state['deter'],
                    'stoch': stoch
                }
            
            # Get action from policy
            action = self.actor(state)
            
        action_numpy = action.squeeze(0).cpu().numpy()
        # Ensure action is always a 1D array of correct size
        if action_numpy.ndim == 0:
            action_numpy = np.array([action_numpy])
        elif action_numpy.shape[0] != 12:
            action_numpy = np.resize(action_numpy, (12,))
        return action_numpy, state
    
    def train_step(self, obs_seq, action_seq, reward_seq):
        """Single training step"""
        # World model update
        self.world_model_optimizer.zero_grad()
        wm_loss, recon_loss, reward_loss, kl_loss = self.compute_world_model_loss(obs_seq, action_seq, reward_seq)
        wm_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.rssm.parameters(), 100.0)
        self.world_model_optimizer.step()
        
        # Get states for policy learning (completely detached from world model gradients)
        with torch.no_grad():
            states = self.encode_sequence(obs_seq, action_seq)
            # Use random states for policy learning
            batch_size = obs_seq.shape[0]
            random_idx = torch.randint(0, len(states), (batch_size,))
            init_states = {
                'deter': torch.stack([states[idx]['deter'][i] for i, idx in enumerate(random_idx)]).detach(),
                'stoch': torch.stack([states[idx]['stoch'][i] for i, idx in enumerate(random_idx)]).detach()
            }
        
        # Separate actor and critic updates to avoid graph conflicts
        # Actor update
        self.actor_optimizer.zero_grad()
        actor_loss = self.compute_actor_loss(init_states)
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 100.0)
        self.actor_optimizer.step()
        
        # Critic update  
        self.critic_optimizer.zero_grad()
        critic_loss = self.compute_critic_loss(init_states)
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 100.0)
        self.critic_optimizer.step()
        
        # Update target critic
        self.update_target_critic()
        
        return {
            'world_model_loss': wm_loss.item(),
            'reconstruction_loss': recon_loss.item(),
            'reward_loss': reward_loss.item(),
            'kl_loss': kl_loss.item(),
            'actor_loss': actor_loss.item(),
            'critic_loss': critic_loss.item()
        }
    
    def forward(self, x):
        """Standard PyTorch forward method required by SAI"""
        # Ensure input is tensor and on correct device with correct dtype
        device = next(self.parameters()).device
        if isinstance(x, np.ndarray):
            x = torch.FloatTensor(x).to(device)
        elif isinstance(x, torch.Tensor):
            x = x.float().to(device)  # Convert to float32 if needed
        
        # Handle batch dimensions properly
        batch_size = x.shape[0] if x.dim() > 1 else 1
        if x.dim() == 1:
            x = x.unsqueeze(0)
        elif x.dim() > 2:
            x = x.view(-1, self.obs_dim)
        
        with torch.no_grad():
            # Simple forward pass for SAI compatibility
            # Initialize state
            state = self.rssm.init_state(batch_size)
            
            # Encode observation
            obs_embed = self.rssm.encode_obs(x)
            
            # Get posterior state
            posterior_input = torch.cat([state['deter'], obs_embed], dim=-1)
            posterior_logits = self.rssm.posterior_net(posterior_input)
            stoch = self.rssm.get_stochastic_state(posterior_logits)
            
            # Create full state
            full_state = {
                'deter': state['deter'],
                'stoch': stoch
            }
            
            # Get action from actor
            action = self.actor(full_state)
            
            # Return tensor (squeeze if single input)
            if batch_size == 1:
                return action.squeeze(0)
            else:
                return action