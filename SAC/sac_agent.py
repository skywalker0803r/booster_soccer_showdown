import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from collections import deque
import random

from utils import Preprocessor, action_function
from rnd_module import RNDModule, RNDBuffer

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
        
        # Ensure states have correct shape (batch_size, feature_dim)
        if len(state.shape) == 3:
            state = state.squeeze(1)  # Remove extra dimension if present
        if len(next_state.shape) == 3:
            next_state = next_state.squeeze(1)  # Remove extra dimension if present
            
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
    def __init__(self, obs_dim, act_dim, env, use_rnd=True, rnd_scale=0.1):
        self.env = env
        self.preproc = Preprocessor()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.use_rnd = use_rnd

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
        
        # RND 模組
        if self.use_rnd:
            self.rnd = RNDModule(obs_dim, intrinsic_reward_scale=rnd_scale)
            self.rnd_buffer = RNDBuffer(maxlen=10000)
            self.rnd_update_freq = 10  # 每10步更新一次RND
            self.step_count = 0
            print(f"RND 模組已啟用 - 獎勵縮放: {rnd_scale}")
        else:
            self.rnd = None
            print("RND 模組未啟用")

    def select_action(self, obs_raw, info):
        obs = self.preproc.modify_state(obs_raw, info)
        # Ensure obs is properly shaped - flatten if needed and add batch dimension if missing
        if len(obs.shape) > 1:
            obs = obs.flatten()
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(DEVICE)  # Add batch dimension
        with torch.no_grad():
            action, _ = self.actor(obs_tensor)
        action = action.cpu().numpy()[0]
        return action_function(action, self.env)

    def update(self):
        if len(self.buffer) < BATCH_SIZE:
            return
        
        # 更新步數計數器
        self.step_count += 1

        state, action, reward, next_state, done = self.buffer.sample()

        # 計算內在獎勵（如果啟用RND）
        if self.use_rnd and self.rnd is not None:
            with torch.no_grad():
                intrinsic_rewards = []
                for i in range(state.shape[0]):
                    intrinsic_reward = self.rnd.compute_intrinsic_reward(state[i])
                    intrinsic_rewards.append(intrinsic_reward[0])
                
                intrinsic_rewards = torch.FloatTensor(intrinsic_rewards).unsqueeze(1).to(DEVICE)
                # 結合外在獎勵和內在獎勵
                combined_reward = reward + intrinsic_rewards
        else:
            combined_reward = reward

        with torch.no_grad():
            next_action, next_log_prob = self.actor(next_state)
            q1_target = self.critic1_target(next_state, next_action)
            q2_target = self.critic2_target(next_state, next_action)
            q_target = torch.min(q1_target, q2_target) - ALPHA * next_log_prob
            y = combined_reward + (1 - done) * GAMMA * q_target

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
        
        # 更新 RND 網絡
        if self.use_rnd and self.rnd is not None:
            # 將當前批次的狀態加入 RND buffer
            for i in range(state.shape[0]):
                self.rnd_buffer.push(state[i])
            
            # 定期更新 RND 網絡
            if self.step_count % self.rnd_update_freq == 0 and len(self.rnd_buffer) > 0:
                rnd_states = self.rnd_buffer.sample(min(64, len(self.rnd_buffer)))
                rnd_loss = self.rnd.update(rnd_states)
                
                if self.step_count % 100 == 0:  # 每100步打印一次RND統計
                    rnd_stats = self.rnd.get_statistics()
                    print(f"[RND] 步驟 {self.step_count}: 損失={rnd_loss:.4f}, "
                          f"平均內在獎勵={rnd_stats['mean_intrinsic_reward']:.4f}")
    
    def get_intrinsic_reward(self, obs_raw, info):
        """獲取單個觀測的內在獎勵（用於調試）"""
        if not self.use_rnd or self.rnd is None:
            return 0.0
        
        obs = self.preproc.modify_state(obs_raw, info)
        if len(obs.shape) > 1:
            obs = obs.flatten()
        obs_tensor = torch.FloatTensor(obs).to(DEVICE)
        return self.rnd.compute_intrinsic_reward(obs_tensor)[0]
    
    def save_rnd_model(self, filepath):
        """保存 RND 模型"""
        if self.use_rnd and self.rnd is not None:
            self.rnd.save(filepath)
        else:
            print("RND 模組未啟用，無法保存")
    
    def load_rnd_model(self, filepath):
        """加載 RND 模型"""
        if self.use_rnd and self.rnd is not None:
            self.rnd.load(filepath)
        else:
            print("RND 模組未啟用，無法加載")

