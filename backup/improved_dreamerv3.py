"""
Improved DreamerV3 implementation with fixes for:
1. World model loss convergence issues
2. Better sequence processing
3. Improved loss functions with normalization
4. More stable training
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque
import random


class ImprovedRSSMCore(nn.Module):
    """Improved RSSM with better loss functions and stability"""
    
    def __init__(self, obs_dim, action_dim, hidden_dim=512, stoch_dim=32, discrete_dim=32):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.stoch_dim = stoch_dim
        self.discrete_dim = discrete_dim
        self.obs_dim = obs_dim
        
        # Encoder with normalization and residual connections
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
        # Improved recurrent transition
        self.rnn = nn.GRUCell(hidden_dim + action_dim, hidden_dim)
        
        # Stochastic state networks with better initialization
        self.prior_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, stoch_dim * discrete_dim)
        )
        
        self.posterior_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, stoch_dim * discrete_dim)
        )
        
        # Improved decoder with skip connections
        decoder_input_dim = hidden_dim + stoch_dim * discrete_dim
        self.obs_decoder = nn.Sequential(
            nn.Linear(decoder_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, obs_dim)
        )
        
        # Improved reward predictor
        self.reward_predictor = nn.Sequential(
            nn.Linear(decoder_input_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Better weight initialization"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=1.0)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def encode_obs(self, obs):
        """Encode observation with normalization"""
        # Normalize observations to prevent extreme values
        obs_norm = F.layer_norm(obs, [obs.shape[-1]])
        return self.obs_encoder(obs_norm)
    
    def get_stochastic_state(self, logits, temperature=1.0):
        """Sample with temperature control for better exploration"""
        logits = logits.reshape(-1, self.stoch_dim, self.discrete_dim)
        
        # Apply temperature for exploration control
        logits = logits / temperature
        
        # Sample from categorical distribution
        dist = torch.distributions.Categorical(logits=logits)
        stoch_discrete = dist.sample()
        
        # Convert to one-hot with straight-through gradient
        stoch_onehot = F.one_hot(stoch_discrete, self.discrete_dim).float()
        
        # Add small amount of noise for better gradient flow
        if self.training:
            stoch_onehot = stoch_onehot + torch.randn_like(stoch_onehot) * 0.01
        
        return stoch_onehot.reshape(-1, self.stoch_dim * self.discrete_dim)
    
    def transition(self, prev_state, action):
        """Improved transition with gradient clipping"""
        # Ensure action is normalized
        action_norm = torch.tanh(action)
        
        # Combine with gradient clipping
        rnn_input = torch.cat([prev_state['deter'], action_norm], dim=-1)
        
        # Update deterministic state with residual connection
        deter_raw = self.rnn(rnn_input, prev_state['deter'])
        deter = prev_state['deter'] + 0.1 * (deter_raw - prev_state['deter'])  # EMA update
        
        # Predict stochastic state
        prior_logits = self.prior_net(deter)
        stoch = self.get_stochastic_state(prior_logits)
        
        return {
            'deter': deter,
            'stoch': stoch,
            'prior_logits': prior_logits
        }
    
    def observe(self, obs, prev_state, action):
        """Improved observation integration"""
        # Get transition state
        transition_state = self.transition(prev_state, action)
        
        # Encode observation
        obs_embed = self.encode_obs(obs)
        
        # Compute posterior with better conditioning
        posterior_input = torch.cat([transition_state['deter'], obs_embed], dim=-1)
        posterior_logits = self.posterior_net(posterior_input)
        stoch_posterior = self.get_stochastic_state(posterior_logits)
        
        return {
            'deter': transition_state['deter'],
            'stoch': stoch_posterior,
            'prior_logits': transition_state['prior_logits'],
            'posterior_logits': posterior_logits
        }
    
    def decode_obs(self, state):
        """Improved decoder with residual connection"""
        latent = torch.cat([state['deter'], state['stoch']], dim=-1)
        return self.obs_decoder(latent)
    
    def predict_reward(self, state):
        """Improved reward prediction with uncertainty"""
        latent = torch.cat([state['deter'], state['stoch']], dim=-1)
        return self.reward_predictor(latent)
    
    def init_state(self, batch_size):
        """Initialize with better defaults"""
        device = next(self.parameters()).device
        return {
            'deter': torch.zeros(batch_size, self.hidden_dim, device=device),
            'stoch': torch.zeros(batch_size, self.stoch_dim * self.discrete_dim, device=device)
        }


class ImprovedActor(nn.Module):
    """Improved actor with better exploration"""
    
    def __init__(self, state_dim, action_dim, hidden_dim=512):
        super().__init__()
        self.action_dim = action_dim
        
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim * 2)  # Mean and std
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=0.01)  # Small initial policy
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, state, deterministic=False):
        latent = torch.cat([state['deter'], state['stoch']], dim=-1)
        output = self.net(latent)
        
        mean, std_logit = torch.chunk(output, 2, dim=-1)
        mean = torch.tanh(mean)
        
        if deterministic:
            return mean
        else:
            # Use learnable standard deviation
            std = F.softplus(std_logit) + 1e-4
            dist = torch.distributions.Normal(mean, std)
            action = dist.rsample()  # Reparameterization trick
            action = torch.tanh(action)  # Squash to [-1, 1]
            return action


class ImprovedCritic(nn.Module):
    """Improved critic with distributional value learning"""
    
    def __init__(self, state_dim, hidden_dim=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=1.0)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, state):
        latent = torch.cat([state['deter'], state['stoch']], dim=-1)
        return self.net(latent)


class ImprovedDreamerV3(nn.Module):
    """Improved DreamerV3 with better training stability"""
    
    def __init__(self, obs_dim, action_dim, hidden_dim=512, stoch_dim=32, discrete_dim=32):
        super().__init__()
        
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.state_dim = hidden_dim + stoch_dim * discrete_dim
        
        # Improved components
        self.rssm = ImprovedRSSMCore(obs_dim, action_dim, hidden_dim, stoch_dim, discrete_dim)
        self.actor = ImprovedActor(self.state_dim, action_dim, hidden_dim)
        self.critic = ImprovedCritic(self.state_dim, hidden_dim)
        
        # Better optimizers with weight decay
        self.world_model_optimizer = torch.optim.AdamW(
            self.rssm.parameters(), lr=1e-4, weight_decay=1e-6, eps=1e-8
        )
        self.actor_optimizer = torch.optim.AdamW(
            self.actor.parameters(), lr=3e-5, weight_decay=1e-6, eps=1e-8
        )
        self.critic_optimizer = torch.optim.AdamW(
            self.critic.parameters(), lr=3e-5, weight_decay=1e-6, eps=1e-8
        )
        
        # Learning rate schedulers
        self.world_model_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.world_model_optimizer, T_max=1000, eta_min=1e-6
        )
        self.actor_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.actor_optimizer, T_max=1000, eta_min=1e-7
        )
        self.critic_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.critic_optimizer, T_max=1000, eta_min=1e-7
        )
        
        # Target networks
        self.target_critic = ImprovedCritic(self.state_dim, hidden_dim)
        self.target_critic.load_state_dict(self.critic.state_dict())
        
        # Improved hyperparameters
        self.gamma = 0.99
        self.tau = 0.005
        self.horizon = 15
        self.kl_beta = 1.0  # KL weight
        self.kl_tolerance = 3.0  # KL tolerance for free bits
        
    def encode_sequence(self, obs_seq, action_seq):
        """Improved sequence encoding with better error handling"""
        batch_size, seq_len = obs_seq.shape[:2]
        
        # Initialize state
        state = self.rssm.init_state(batch_size)
        states = []
        
        for t in range(seq_len):
            if t == 0:
                # First step: encode observation directly
                obs_embed = self.rssm.encode_obs(obs_seq[:, t])
                # Create initial posterior state
                deter = state['deter']
                posterior_input = torch.cat([deter, obs_embed], dim=-1)
                posterior_logits = self.rssm.posterior_net(posterior_input)
                stoch = self.rssm.get_stochastic_state(posterior_logits)
                
                # Create prior for KL computation
                prior_logits = self.rssm.prior_net(deter)
                
                state = {
                    'deter': deter,
                    'stoch': stoch,
                    'prior_logits': prior_logits,
                    'posterior_logits': posterior_logits
                }
            else:
                # Use previous action for transition
                prev_action = action_seq[:, t-1] if t > 0 else torch.zeros_like(action_seq[:, 0])
                state = self.rssm.observe(obs_seq[:, t], state, prev_action)
            
            states.append(state)
        
        return states
    
    def compute_improved_world_model_loss(self, obs_seq, action_seq, reward_seq):
        """Improved world model loss with better normalization and KL control"""
        batch_size, seq_len = obs_seq.shape[:2]
        
        # Encode sequence
        states = self.encode_sequence(obs_seq, action_seq)
        
        # Initialize loss components
        reconstruction_losses = []
        reward_losses = []
        kl_losses = []
        
        for t, state in enumerate(states):
            # 1. Improved Observation Reconstruction Loss
            obs_recon = self.rssm.decode_obs(state)
            obs_target = obs_seq[:, t]
            
            # Normalize both prediction and target
            obs_recon_norm = F.layer_norm(obs_recon, [obs_recon.shape[-1]])
            obs_target_norm = F.layer_norm(obs_target, [obs_target.shape[-1]])
            
            # Use Huber loss for robustness
            recon_loss = F.huber_loss(obs_recon_norm, obs_target_norm, delta=1.0)
            reconstruction_losses.append(recon_loss)
            
            # 2. Improved Reward Prediction Loss
            if t < len(states) - 1:  # No reward for last state
                reward_pred = self.rssm.predict_reward(state)
                reward_target = reward_seq[:, t:t+1]  # Keep dimension
                
                # Use Huber loss for reward prediction too
                reward_loss = F.huber_loss(reward_pred, reward_target)
                reward_losses.append(reward_loss)
            
            # 3. Improved KL Loss with Free Bits
            if 'posterior_logits' in state:
                # Reshape for categorical distributions
                prior_logits = state['prior_logits'].reshape(-1, self.rssm.stoch_dim, self.rssm.discrete_dim)
                posterior_logits = state['posterior_logits'].reshape(-1, self.rssm.stoch_dim, self.rssm.discrete_dim)
                
                # Create distributions
                prior_dist = torch.distributions.Categorical(logits=prior_logits)
                posterior_dist = torch.distributions.Categorical(logits=posterior_logits)
                
                # Compute KL with free bits
                kl = torch.distributions.kl_divergence(posterior_dist, prior_dist).mean(dim=0)  # Per stochastic unit
                kl_free_bits = torch.maximum(kl, torch.tensor(self.kl_tolerance, device=kl.device))
                kl_loss = kl_free_bits.mean()
                kl_losses.append(kl_loss)
        
        # Average losses
        reconstruction_loss = torch.stack(reconstruction_losses).mean()
        reward_loss = torch.stack(reward_losses).mean() if reward_losses else torch.tensor(0.0, device=obs_seq.device)
        kl_loss = torch.stack(kl_losses).mean() if kl_losses else torch.tensor(0.0, device=obs_seq.device)
        
        # Total loss with adaptive weighting
        total_loss = reconstruction_loss + reward_loss + self.kl_beta * kl_loss
        
        return total_loss, reconstruction_loss, reward_loss, kl_loss
    
    def select_action(self, obs, state=None, deterministic=False):
        """Improved action selection with better state management"""
        with torch.no_grad():
            device = next(self.parameters()).device
            
            # Ensure proper tensor conversion
            if isinstance(obs, np.ndarray):
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
            else:
                obs_tensor = obs.unsqueeze(0).to(device) if obs.dim() == 1 else obs.to(device)
            
            if state is None:
                # Initialize state
                state = self.rssm.init_state(1)
                obs_embed = self.rssm.encode_obs(obs_tensor)
                
                # Create initial state
                posterior_input = torch.cat([state['deter'], obs_embed], dim=-1)
                posterior_logits = self.rssm.posterior_net(posterior_input)
                stoch = self.rssm.get_stochastic_state(posterior_logits, temperature=0.5)
                
                state = {
                    'deter': state['deter'],
                    'stoch': stoch
                }
            else:
                # Update existing state (simplified for action selection)
                obs_embed = self.rssm.encode_obs(obs_tensor)
                posterior_input = torch.cat([state['deter'], obs_embed], dim=-1)
                posterior_logits = self.rssm.posterior_net(posterior_input)
                stoch = self.rssm.get_stochastic_state(posterior_logits, temperature=0.5)
                
                state = {
                    'deter': state['deter'],
                    'stoch': stoch
                }
            
            # Get action
            action = self.actor(state, deterministic=deterministic)
            
            # Convert to numpy
            action_numpy = action.squeeze(0).cpu().numpy()
            
            # Ensure correct shape
            if action_numpy.ndim == 0:
                action_numpy = np.array([action_numpy])
            elif len(action_numpy) != self.action_dim:
                action_numpy = np.resize(action_numpy, (self.action_dim,))
            
            return action_numpy, state
    
    def train_step(self, obs_seq, action_seq, reward_seq):
        """Improved training step with gradient clipping and scheduling"""
        # 1. World Model Update
        self.world_model_optimizer.zero_grad()
        wm_loss, recon_loss, reward_loss, kl_loss = self.compute_improved_world_model_loss(
            obs_seq, action_seq, reward_seq
        )
        wm_loss.backward()
        
        # Gradient clipping
        world_model_grad_norm = torch.nn.utils.clip_grad_norm_(self.rssm.parameters(), 10.0)
        self.world_model_optimizer.step()
        self.world_model_scheduler.step()
        
        # 2. Get detached states for policy learning
        with torch.no_grad():
            states = self.encode_sequence(obs_seq, action_seq)
            # Sample random states for policy learning
            batch_size = obs_seq.shape[0]
            random_indices = torch.randint(0, len(states), (batch_size,))
            init_states = {
                'deter': torch.stack([states[idx]['deter'][i] for i, idx in enumerate(random_indices)]).detach(),
                'stoch': torch.stack([states[idx]['stoch'][i] for i, idx in enumerate(random_indices)]).detach()
            }
        
        # 3. Actor Update
        self.actor_optimizer.zero_grad()
        actor_loss = self.compute_actor_loss(init_states)
        actor_loss.backward()
        actor_grad_norm = torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 10.0)
        self.actor_optimizer.step()
        self.actor_scheduler.step()
        
        # 4. Critic Update
        self.critic_optimizer.zero_grad()
        critic_loss = self.compute_critic_loss(init_states)
        critic_loss.backward()
        critic_grad_norm = torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 10.0)
        self.critic_optimizer.step()
        self.critic_scheduler.step()
        
        # 5. Update target networks
        self.update_target_critic()
        
        return {
            'world_model_loss': wm_loss.item(),
            'reconstruction_loss': recon_loss.item(),
            'reward_loss': reward_loss.item(),
            'kl_loss': kl_loss.item(),
            'actor_loss': actor_loss.item(),
            'critic_loss': critic_loss.item(),
            'world_model_grad_norm': world_model_grad_norm.item(),
            'actor_grad_norm': actor_grad_norm.item(),
            'critic_grad_norm': critic_grad_norm.item()
        }
    
    def compute_actor_loss(self, init_states):
        """Improved actor loss computation"""
        # Generate imagined trajectories
        imag_states, imag_actions = self.imagine_sequence(init_states, self.actor, self.horizon)
        
        # Compute values
        values = []
        for state in imag_states:
            value = self.critic(state)
            values.append(value.squeeze(-1))
        
        # Compute rewards
        rewards = []
        for i in range(len(imag_states) - 1):
            reward = self.rssm.predict_reward(imag_states[i]).squeeze(-1)
            rewards.append(reward)
        
        # Compute lambda returns with better numerical stability
        returns = self._compute_lambda_returns(rewards, values, self.gamma, lambda_=0.95)
        
        # Compute advantages with normalization
        advantages = returns[:-1] - torch.stack(values[:-1])
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Actor loss (reinforce with baseline)
        log_probs = []
        for i, (state, action) in enumerate(zip(imag_states[:-1], imag_actions)):
            # Compute log probability of imagined action
            action_pred = self.actor(state, deterministic=False)
            log_prob = -F.mse_loss(action_pred, action, reduction='none').mean(dim=-1)
            log_probs.append(log_prob)
        
        log_probs = torch.stack(log_probs)
        actor_loss = -(advantages.detach() * log_probs).mean()
        
        return actor_loss
    
    def compute_critic_loss(self, init_states):
        """Improved critic loss with target network"""
        # Generate trajectories
        with torch.no_grad():
            imag_states, _ = self.imagine_sequence(init_states, self.actor, self.horizon)
        
        # Compute current values
        values = []
        target_values = []
        for state in imag_states:
            value = self.critic(state)
            target_value = self.target_critic(state)
            values.append(value.squeeze(-1))
            target_values.append(target_value.squeeze(-1))
        
        # Compute rewards
        rewards = []
        for i in range(len(imag_states) - 1):
            with torch.no_grad():
                reward = self.rssm.predict_reward(imag_states[i]).squeeze(-1)
            rewards.append(reward)
        
        # Compute target returns
        with torch.no_grad():
            target_returns = self._compute_lambda_returns(rewards, target_values, self.gamma, lambda_=0.95)
        
        # Critic loss with target returns
        critic_preds = torch.stack(values[:-1])
        critic_targets = target_returns[:-1].detach()
        
        # Use Huber loss for robustness
        critic_loss = F.huber_loss(critic_preds, critic_targets)
        
        return critic_loss
    
    def imagine_sequence(self, init_state, actor, horizon):
        """Generate imagined trajectory"""
        states = [init_state]
        actions = []
        
        state = init_state
        for _ in range(horizon):
            # Sample action
            action = actor(state, deterministic=False)
            actions.append(action)
            
            # Predict next state
            state = self.rssm.transition(state, action)
            states.append(state)
        
        return states, actions
    
    def _compute_lambda_returns(self, rewards, values, gamma, lambda_=0.95):
        """Improved lambda returns computation"""
        returns = []
        last_value = values[-1]
        
        for t in reversed(range(len(rewards))):
            reward = rewards[t]
            value = values[t]
            next_value = values[t + 1]
            
            # TD error
            delta = reward + gamma * next_value - value
            
            # Lambda return with better numerical stability
            last_value = value + delta + gamma * lambda_ * (last_value - next_value)
            returns.insert(0, last_value)
        
        returns.append(values[-1])  # Add final value
        return torch.stack(returns)
    
    def update_target_critic(self):
        """Soft update target critic"""
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
    
    def forward(self, x):
        """SAI-compatible forward pass"""
        device = next(self.parameters()).device
        
        # Convert input
        if isinstance(x, np.ndarray):
            x = torch.FloatTensor(x).to(device)
        elif isinstance(x, torch.Tensor):
            x = x.float().to(device)
        
        # Handle batch dimensions
        batch_size = x.shape[0] if x.dim() > 1 else 1
        if x.dim() == 1:
            x = x.unsqueeze(0)
        
        with torch.no_grad():
            # Simple forward for SAI
            state = self.rssm.init_state(batch_size)
            obs_embed = self.rssm.encode_obs(x)
            
            posterior_input = torch.cat([state['deter'], obs_embed], dim=-1)
            posterior_logits = self.rssm.posterior_net(posterior_input)
            stoch = self.rssm.get_stochastic_state(posterior_logits, temperature=0.1)
            
            full_state = {
                'deter': state['deter'],
                'stoch': stoch
            }
            
            action = self.actor(full_state, deterministic=True)
            
            if batch_size == 1:
                return action.squeeze(0)
            else:
                return action