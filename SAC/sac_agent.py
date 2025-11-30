import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from collections import deque
import random

from utils import Preprocessor, action_function
from rnd_module import RNDModule  # 移除 RNDBuffer 導入

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
    def __init__(self, obs_dim, act_dim, env, use_rnd=True, rnd_scale=0.1, logger=None, gdrive_saver=None):
        self.env = env
        self.preproc = Preprocessor()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.use_rnd = use_rnd
        self.logger = logger
        self.gdrive_saver = gdrive_saver
        
        # 跟蹤最佳性能
        self.best_episode_reward = float('-inf')
        self.current_episode = 0

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
            self.rnd_update_freq = 10  # 每10步更新一次RND
            self.step_count = 0
            print(f"RND 模組已啟用 - 獎勵縮放: {rnd_scale}")
            print("✅ RND 使用 next_state 計算內在獎勵（符合原始論文）")
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

        # 計算內在獎勵（如果啟用RND）- 使用 next_state 根據原始論文
        if self.use_rnd and self.rnd is not None:
            with torch.no_grad():
                intrinsic_rewards = []
                for i in range(next_state.shape[0]):  # 修正：使用 next_state
                    intrinsic_reward = self.rnd.compute_intrinsic_reward(next_state[i])
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
        
        # 記錄 SAC 訓練指標到 TensorBoard
        if self.logger is not None:
            self.logger.log_sac_update(
                step=self.step_count,
                actor_loss=actor_loss.item(),
                critic1_loss=critic1_loss.item(),
                critic2_loss=critic2_loss.item(),
                q1_value=q1,
                q2_value=q2,
                log_prob=log_prob,
                alpha=ALPHA
            )
            
            # 記錄動作分佈
            if self.step_count % 50 == 0:  # 每50步記錄一次動作分佈
                self.logger.log_action_distribution(self.step_count, action_new)
            
            # 記錄網絡參數統計
            if self.step_count % 100 == 0:  # 每100步記錄一次網絡統計
                self.logger.log_network_metrics(self.step_count, 'Actor', self.actor)
                self.logger.log_network_metrics(self.step_count, 'Critic1', self.critic1)
                self.logger.log_network_metrics(self.step_count, 'Critic2', self.critic2)

        # Soft update
        for target_param, param in zip(self.critic1_target.parameters(), self.critic1.parameters()):
            target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)
        for target_param, param in zip(self.critic2_target.parameters(), self.critic2.parameters()):
            target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)
        
        # 更新 RND 網絡 - 修正：使用 next_state 並直接從主 buffer 採樣
        if self.use_rnd and self.rnd is not None:
            # 定期更新 RND 網絡
            if self.step_count % self.rnd_update_freq == 0:
                # 使用 next_state 來更新 RND 預測網絡（符合原始論文）
                rnd_states = [next_state[i] for i in range(next_state.shape[0])]
                rnd_loss = self.rnd.update(rnd_states)
                rnd_stats = self.rnd.get_statistics()
                
                # 記錄 RND 指標到 TensorBoard
                if self.logger is not None:
                    self.logger.log_rnd_update(
                        step=self.step_count,
                        rnd_loss=rnd_loss,
                        mean_intrinsic_reward=rnd_stats['mean_intrinsic_reward'],
                        std_intrinsic_reward=rnd_stats['std_intrinsic_reward'],
                        obs_count=rnd_stats['obs_count'],
                        rnd_buffer_size=None  # 不再使用獨立的 RND buffer
                    )
                    
                    # 記錄獎勵分解
                    if len(intrinsic_rewards) > 0:
                        avg_ext_reward = reward.mean().item()
                        avg_int_reward = intrinsic_rewards.mean().item()
                        avg_total_reward = combined_reward.mean().item()
                        self.logger.log_reward_breakdown(
                            self.step_count, avg_ext_reward, avg_int_reward, avg_total_reward
                        )
                
                if self.step_count % 100 == 0:  # 每100步打印一次RND統計
                    print(f"[RND] 步驟 {self.step_count}: 損失={rnd_loss:.4f}, "
                          f"平均內在獎勵={rnd_stats['mean_intrinsic_reward']:.4f}")
            
            # 記錄 Buffer 使用情況
            if self.logger is not None and self.step_count % 50 == 0:
                self.logger.log_buffer_metrics(
                    self.step_count, 
                    len(self.buffer), 
                    self.buffer.buffer.maxlen
                )
    
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
    
    def update_episode_performance(self, episode, episode_reward):
        """更新回合性能並觸發自動保存"""
        self.current_episode = episode
        
        # 檢查是否是新的最佳性能
        is_new_best = episode_reward > self.best_episode_reward
        if is_new_best:
            self.best_episode_reward = episode_reward
        
        # 觸發 Google Drive 自動保存
        if self.gdrive_saver is not None:
            try:
                # 準備保存指標
                metrics = {
                    'episode': episode,
                    'reward': episode_reward,
                    'is_best': is_new_best,
                    'step_count': self.step_count if hasattr(self, 'step_count') else 0
                }
                
                # 添加 RND 統計（如果可用）
                if self.use_rnd and self.rnd is not None:
                    rnd_stats = self.rnd.get_statistics()
                    metrics.update({
                        'rnd_mean_intrinsic_reward': rnd_stats.get('mean_intrinsic_reward', 0),
                        'rnd_obs_count': rnd_stats.get('obs_count', 0)
                    })
                
                # 嘗試保存到 Google Drive
                saved = self.gdrive_saver.save_model(self, episode, episode_reward, metrics)
                
                if saved:
                    print(f"💾 模型已自動保存到 Google Drive (回合 {episode}, 獎勵 {episode_reward:.2f})")
                    
                    # 記錄到 TensorBoard
                    if self.logger is not None:
                        self.logger.log_text("GoogleDrive_Save", 
                                           f"回合 {episode}: 獎勵 {episode_reward:.2f} - {'最佳' if is_new_best else '定期'}保存")
                
            except Exception as e:
                print(f"❌ Google Drive 自動保存失敗: {e}")
    
    def save_checkpoint(self, filepath, episode=None, reward=None):
        """保存完整的訓練檢查點"""
        if episode is None:
            episode = self.current_episode
        if reward is None:
            reward = self.best_episode_reward
            
        checkpoint = {
            # SAC 網絡狀態
            'actor_state_dict': self.actor.state_dict(),
            'critic1_state_dict': self.critic1.state_dict(),
            'critic2_state_dict': self.critic2.state_dict(),
            'critic1_target_state_dict': self.critic1_target.state_dict(),
            'critic2_target_state_dict': self.critic2_target.state_dict(),
            
            # 優化器狀態
            'actor_optimizer_state_dict': self.actor_opt.state_dict(),
            'critic1_optimizer_state_dict': self.critic1_opt.state_dict(),
            'critic2_optimizer_state_dict': self.critic2_opt.state_dict(),
            
            # 訓練狀態
            'episode': episode,
            'reward': reward,
            'best_episode_reward': self.best_episode_reward,
            'step_count': getattr(self, 'step_count', 0),
            
            # 模型配置
            'obs_dim': self.obs_dim,
            'act_dim': self.act_dim,
            'use_rnd': self.use_rnd,
        }
        
        # 添加 RND 狀態（如果啟用）
        if self.use_rnd and self.rnd is not None:
            checkpoint['rnd_state'] = {
                'network_state_dict': self.rnd.rnd_network.state_dict(),
                'optimizer_state_dict': self.rnd.optimizer.state_dict(),
                'running_mean': self.rnd.running_mean,
                'running_var': self.rnd.running_var,
                'obs_count': self.rnd.obs_count,
                'reward_history': list(self.rnd.reward_history)
            }
        
        torch.save(checkpoint, filepath)
        print(f"✅ 檢查點已保存: {filepath}")
        
        return checkpoint
    
    def load_checkpoint(self, filepath):
        """加載完整的訓練檢查點"""
        try:
            checkpoint = torch.load(filepath, map_location=DEVICE)
            
            # 恢復 SAC 網絡狀態
            self.actor.load_state_dict(checkpoint['actor_state_dict'])
            self.critic1.load_state_dict(checkpoint['critic1_state_dict'])
            self.critic2.load_state_dict(checkpoint['critic2_state_dict'])
            self.critic1_target.load_state_dict(checkpoint['critic1_target_state_dict'])
            self.critic2_target.load_state_dict(checkpoint['critic2_target_state_dict'])
            
            # 恢復優化器狀態
            self.actor_opt.load_state_dict(checkpoint['actor_optimizer_state_dict'])
            self.critic1_opt.load_state_dict(checkpoint['critic1_optimizer_state_dict'])
            self.critic2_opt.load_state_dict(checkpoint['critic2_optimizer_state_dict'])
            
            # 恢復訓練狀態
            self.current_episode = checkpoint.get('episode', 0)
            self.best_episode_reward = checkpoint.get('best_episode_reward', float('-inf'))
            self.step_count = checkpoint.get('step_count', 0)
            
            # 恢復 RND 狀態（如果啟用）
            if self.use_rnd and self.rnd is not None and 'rnd_state' in checkpoint:
                rnd_state = checkpoint['rnd_state']
                self.rnd.rnd_network.load_state_dict(rnd_state['network_state_dict'])
                self.rnd.optimizer.load_state_dict(rnd_state['optimizer_state_dict'])
                self.rnd.running_mean = rnd_state['running_mean']
                self.rnd.running_var = rnd_state['running_var']
                self.rnd.obs_count = rnd_state['obs_count']
                self.rnd.reward_history.extend(rnd_state['reward_history'])
            
            print(f"✅ 檢查點已加載: {filepath}")
            print(f"   回合: {self.current_episode}, 最佳獎勵: {self.best_episode_reward:.2f}")
            
            return checkpoint
            
        except Exception as e:
            print(f"❌ 加載檢查點失敗: {e}")
            return None
    
    def force_save_to_gdrive(self, reason="manual"):
        """強制保存到 Google Drive"""
        if self.gdrive_saver is not None:
            return self.gdrive_saver.manual_save(self, self.current_episode, self.best_episode_reward, reason)
        else:
            print("❌ Google Drive 保存器未配置")
            return False
    
    def get_gdrive_statistics(self):
        """獲取 Google Drive 保存統計"""
        if self.gdrive_saver is not None:
            return self.gdrive_saver.get_statistics()
        return None

