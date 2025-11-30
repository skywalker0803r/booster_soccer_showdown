from sac_agent import SACAgent
from utils import Preprocessor
from sai_rl import SAIClient
import time
import numpy as np

# 環境
sai = SAIClient(
    comp_id="booster-soccer-showdown", 
    api_key="sai_LFcuaCZiqEkUbNVolQ3wbk5yU7H11jfv",
)
env = sai.make_env()
obs_raw,info = env.reset()
obs = Preprocessor().modify_state(obs_raw, info)
obs_dim = obs.shape[-1] if len(obs.shape) > 1 else obs.shape[0]
act_dim = env.action_space.shape[0]

agent = SACAgent(obs_dim, act_dim, env)

print(f"=== SAC 訓練開始 ===")
print(f"觀測維度: {obs_dim}")
print(f"動作維度: {act_dim}")
print(f"Buffer 大小: {agent.buffer.__dict__['buffer'].maxlen}")
print(f"訓練總回合數: 1000")
print("=" * 50)

# 訓練統計
total_rewards = []
episode_lengths = []
start_time = time.time()

for episode in range(1000):
    obs_raw, info = env.reset()
    done = False
    episode_reward = 0
    episode_steps = 0
    episode_start_time = time.time()
    
    while not done:
        action = agent.select_action(obs_raw, info)
        next_obs_raw, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        episode_reward += reward
        episode_steps += 1
        
        next_obs = Preprocessor().modify_state(next_obs_raw, info)
        # Ensure observations are flattened before storing in buffer
        current_obs = Preprocessor().modify_state(obs_raw, info)
        if len(current_obs.shape) > 1:
            current_obs = current_obs.flatten()
        if len(next_obs.shape) > 1:
            next_obs = next_obs.flatten()
        
        agent.buffer.push(
            current_obs,
            action,
            reward,
            next_obs,
            done
        )
        agent.update()
        obs_raw = next_obs_raw
    
    # 記錄統計資訊
    total_rewards.append(episode_reward)
    episode_lengths.append(episode_steps)
    episode_time = time.time() - episode_start_time
    
    # 每 10 回合打印詳細資訊
    if episode % 10 == 0:
        avg_reward_10 = np.mean(total_rewards[-10:]) if len(total_rewards) >= 10 else np.mean(total_rewards)
        avg_length_10 = np.mean(episode_lengths[-10:]) if len(episode_lengths) >= 10 else np.mean(episode_lengths)
        buffer_size = len(agent.buffer)
        elapsed_time = time.time() - start_time
        
        print(f"回合 {episode:4d} | "
              f"獎勵: {episode_reward:8.2f} | "
              f"步數: {episode_steps:3d} | "
              f"時間: {episode_time:.1f}s | "
              f"Buffer: {buffer_size:6d}")
        print(f"         | "
              f"平均獎勵(10回合): {avg_reward_10:8.2f} | "
              f"平均步數(10回合): {avg_length_10:6.1f} | "
              f"總時間: {elapsed_time/60:.1f}min")
        print("-" * 80)
    
    # 每 100 回合打印總結
    if episode % 100 == 0 and episode > 0:
        avg_reward_100 = np.mean(total_rewards[-100:])
        max_reward_100 = np.max(total_rewards[-100:])
        min_reward_100 = np.min(total_rewards[-100:])
        print(f"\n=== 第 {episode} 回合總結 ===")
        print(f"過去100回合 - 平均獎勵: {avg_reward_100:.2f}, 最高: {max_reward_100:.2f}, 最低: {min_reward_100:.2f}")
        print(f"Buffer 使用率: {len(agent.buffer)/agent.buffer.__dict__['buffer'].maxlen*100:.1f}%")
        print("=" * 50)

# 訓練完成統計
total_time = time.time() - start_time
print(f"\n=== 訓練完成 ===")
print(f"總訓練時間: {total_time/3600:.2f} 小時")
print(f"平均每回合時間: {total_time/1000:.2f} 秒")
print(f"總平均獎勵: {np.mean(total_rewards):.2f}")
print(f"最佳獎勵: {np.max(total_rewards):.2f}")
print(f"最後100回合平均獎勵: {np.mean(total_rewards[-100:]):.2f}")
print("=" * 50)
