import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from collections import deque
import random

from utils import Preprocessor, action_function

# --- SAC Hyperparameters ---
GAMMA = 0.99
TAU = 0.005
ALPHA = 0.2
LR = 3e-4
BUFFER_SIZE = 1_000_000
BATCH_SIZE = 256
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --- Replay Buffer ---
class ReplayBuffer:
    def __init__(self, size=BUFFER_SIZE):
        self.buffer = deque(maxlen=size)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size=BATCH_SIZE):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return (torch.FloatTensor(state).to(DEVICE),
                torch.FloatTensor(action).to(DEVICE),
                torch.FloatTensor(reward).unsqueeze(1).to(DEVICE),
                torch.FloatTensor(next_state).to(DEVICE),
                torch.FloatTensor(done).unsqueeze(1).to(DEVICE))

    def __len__(self):
        return len(self.buffer)


# --- Actor Network ---
class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.mean = nn.Linear(256, act_dim)
        self.log_std = nn.Linear(256, act_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mean = self.mean(x)
        log_std = self.log_std(x).clamp(-20, 2)
        std = log_std.exp()
        z = mean + std * torch.randn_like(std)
        action = torch.tanh(z)
        log_prob = -0.5 * ((z - mean) / (std + 1e-6)).pow(2) - log_std - 0.5 * np.log(2*np.pi)
        log_prob = log_prob.sum(1, keepdim=True)
        return action, log_prob


# --- Critic Network ---
class Critic(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim + act_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.q = nn.Linear(256, 1)

    def forward(self, obs, act):
        x = torch.cat([obs, act], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.q(x)


# --- SAC Agent ---
class SACAgent:
    def __init__(self, obs_dim, act_dim, env):
        self.env = env
        self.preproc = Preprocessor()
        self.obs_dim = obs_dim
        self.act_dim = act_dim

        # Networks
        self.actor = Actor(obs_dim, act_dim).to(DEVICE)
        self.critic1 = Critic(obs_dim, act_dim).to(DEVICE)
        self.critic2 = Critic(obs_dim, act_dim).to(DEVICE)
        self.critic1_target = Critic(obs_dim, act_dim).to(DEVICE)
        self.critic2_target = Critic(obs_dim, act_dim).to(DEVICE)
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())

        # Optimizers
        self.actor_opt = optim.Adam(self.actor.parameters(), lr=LR)
        self.critic1_opt = optim.Adam(self.critic1.parameters(), lr=LR)
        self.critic2_opt = optim.Adam(self.critic2.parameters(), lr=LR)

        # Replay buffer
        self.buffer = ReplayBuffer()

    def select_action(self, obs_raw, info):
        obs = self.preproc.modify_state(obs_raw, info)
        obs_tensor = torch.FloatTensor(obs).to(DEVICE)
        with torch.no_grad():
            action, _ = self.actor(obs_tensor)
        action = action.cpu().numpy()[0]
        return action_function(action, self.env)

    def update(self):
        if len(self.buffer) < BATCH_SIZE:
            return

        state, action, reward, next_state, done = self.buffer.sample()

        with torch.no_grad():
            next_action, next_log_prob = self.actor(next_state)
            q1_target = self.critic1_target(next_state, next_action)
            q2_target = self.critic2_target(next_state, next_action)
            q_target = torch.min(q1_target, q2_target) - ALPHA * next_log_prob
            y = reward + (1 - done) * GAMMA * q_target

        # Critic update
        q1 = self.critic1(state, action)
        q2 = self.critic2(state, action)
        critic1_loss = F.mse_loss(q1, y)
        critic2_loss = F.mse_loss(q2, y)
        self.critic1_opt.zero_grad()
        critic1_loss.backward()
        self.critic1_opt.step()
        self.critic2_opt.zero_grad()
        critic2_loss.backward()
        self.critic2_opt.step()

        # Actor update
        action_new, log_prob = self.actor(state)
        q1_new = self.critic1(state, action_new)
        q2_new = self.critic2(state, action_new)
        q_new = torch.min(q1_new, q2_new)
        actor_loss = (ALPHA * log_prob - q_new).mean()
        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        # Soft update
        for target_param, param in zip(self.critic1_target.parameters(), self.critic1.parameters()):
            target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)
        for target_param, param in zip(self.critic2_target.parameters(), self.critic2.parameters()):
            target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)

