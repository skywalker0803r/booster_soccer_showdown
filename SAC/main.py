from sac_agent import SACAgent
from utils import Preprocessor
from sai_rl import SAIClient
from tensorboard_logger import SAC_RND_TensorBoardLogger
# from gdrive_saver import GoogleDriveAutoSaver  # Colab 環境不需要
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

# 創建 TensorBoard Logger
logger = SAC_RND_TensorBoardLogger(
    log_dir="tensorboard_logs",
    experiment_name="SAC_RND_Soccer",
    comment="booster_soccer_showdown"
)

# Colab 環境直接使用本地保存（文件會自動同步到 Google Drive）
gdrive_saver = None
print("💾 Colab 模式：使用本地保存（文件自動同步到 Google Drive）")

# 記錄超參數
hyperparameters = {
    'obs_dim': obs_dim,
    'act_dim': act_dim,
    'gamma': 0.99,
    'tau': 0.005,
    'alpha': 0.2,
    'learning_rate': 3e-4,
    'buffer_size': 1_000_000,
    'batch_size': 256,
    'rnd_scale': 0.1,
    'rnd_update_freq': 10,
    'total_episodes': 1000,
    'environment': 'colab'
}
logger.log_hyperparameters(hyperparameters)

# 創建帶有RND模組、Logger和Google Drive保存器的SAC代理
agent = SACAgent(obs_dim, act_dim, env, use_rnd=True, rnd_scale=0.1, logger=logger, gdrive_saver=gdrive_saver)

print(f"=== SAC + RND + TensorBoard 訓練開始 ===")
print(f"觀測維度: {obs_dim}")
print(f"動作維度: {act_dim}")
print(f"Buffer 大小: {agent.buffer.__dict__['buffer'].maxlen}")
print(f"訓練總回合數: 1000")
print(f"TensorBoard 日誌: {logger.log_dir}")
print(f"啟動 TensorBoard: tensorboard --logdir=tensorboard_logs")

# Colab 環境狀態
print(f"💾 保存模式: Colab 本地保存（自動同步到 Google Drive）")

print("=" * 60)

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
    
    # 記錄到 TensorBoard
    logger.log_episode_summary(
        episode=episode,
        episode_reward=episode_reward,
        episode_steps=episode_steps,
        episode_time=episode_time
    )
    
    # 記錄滑動平均
    logger.log_moving_averages(episode, total_rewards, episode_lengths)
    
    # 更新代理性能並觸發自動保存
    agent.update_episode_performance(episode, episode_reward)
    
    # 每 10 回合打印詳細資訊
    if episode % 10 == 0:
        avg_reward_10 = np.mean(total_rewards[-10:]) if len(total_rewards) >= 10 else np.mean(total_rewards)
        avg_length_10 = np.mean(episode_lengths[-10:]) if len(episode_lengths) >= 10 else np.mean(episode_lengths)
        buffer_size = len(agent.buffer)
        elapsed_time = time.time() - start_time
        
        # 獲取 RND 統計資訊
        rnd_info = ""
        if agent.use_rnd and agent.rnd is not None:
            rnd_stats = agent.rnd.get_statistics()
            rnd_buffer_size = len(agent.rnd_buffer) if hasattr(agent, 'rnd_buffer') else 0
            rnd_info = f" | RND內在獎勵: {rnd_stats['mean_intrinsic_reward']:6.3f} | RND_Buffer: {rnd_buffer_size:4d}"
        
        print(f"回合 {episode:4d} | "
              f"獎勵: {episode_reward:8.2f} | "
              f"步數: {episode_steps:3d} | "
              f"時間: {episode_time:.1f}s | "
              f"Buffer: {buffer_size:6d}{rnd_info}")
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
        
        # 保存本地檢查點
        checkpoint_name = f"checkpoint_episode_{episode}.pth"
        agent.save_checkpoint(checkpoint_name, episode, episode_reward)
        print(f"📁 本地檢查點已保存: {checkpoint_name}")
        
        print("=" * 60)

# 訓練完成統計
total_time = time.time() - start_time
print(f"\n=== 訓練完成 ===")
print(f"總訓練時間: {total_time/3600:.2f} 小時")
print(f"平均每回合時間: {total_time/1000:.2f} 秒")
print(f"總平均獎勵: {np.mean(total_rewards):.2f}")
print(f"最佳獎勵: {np.max(total_rewards):.2f}")
print(f"最後100回合平均獎勵: {np.mean(total_rewards[-100:]):.2f}")

# 記錄最終學習曲線和統計
final_metrics = {
    'Final_Average_Reward': np.mean(total_rewards),
    'Final_Best_Reward': np.max(total_rewards),
    'Final_Last100_Average': np.mean(total_rewards[-100:]) if len(total_rewards) >= 100 else np.mean(total_rewards),
    'Total_Training_Time': total_time,
    'Episodes_Completed': len(total_rewards)
}
logger.log_learning_curves(len(total_rewards), final_metrics)

# 保存 RND 模型
if agent.use_rnd:
    agent.save_rnd_model("rnd_model.pth")
    final_rnd_stats = agent.rnd.get_statistics()
    
    # 記錄 RND 最終統計到 TensorBoard
    logger.log_text("Final_RND_Statistics", f"""
    RND 最終統計:
    - 觀測處理次數: {final_rnd_stats['obs_count']}
    - 最終平均內在獎勵: {final_rnd_stats['mean_intrinsic_reward']:.4f}
    - 內在獎勵標準差: {final_rnd_stats['std_intrinsic_reward']:.4f}
    - RND Buffer 大小: {len(agent.rnd_buffer)}
    """)
    
    print(f"\nRND 最終統計:")
    print(f"- 觀測處理次數: {final_rnd_stats['obs_count']}")
    print(f"- 最終平均內在獎勵: {final_rnd_stats['mean_intrinsic_reward']:.4f}")
    print(f"- 內在獎勵標準差: {final_rnd_stats['std_intrinsic_reward']:.4f}")
    print(f"- RND Buffer 大小: {len(agent.rnd_buffer)}")

# 記錄訓練總結到 TensorBoard
training_summary = f"""
=== 訓練總結 ===
- 總訓練時間: {total_time/3600:.2f} 小時
- 平均每回合時間: {total_time/1000:.2f} 秒
- 總平均獎勵: {np.mean(total_rewards):.2f}
- 最佳獎勵: {np.max(total_rewards):.2f}
- 最後100回合平均獎勵: {np.mean(total_rewards[-100:]):.2f}
- 使用 RND: {'是' if agent.use_rnd else '否'}
- TensorBoard 日誌: {logger.log_dir}
"""
logger.log_text("Training_Summary", training_summary)

# 保存最終模型
print("\n=== 保存最終模型 ===")

# 保存最終本地檢查點
final_checkpoint = f"final_checkpoint_{len(total_rewards)}episodes.pth"
agent.save_checkpoint(final_checkpoint)
print(f"💾 最終檢查點已保存: {final_checkpoint}")

# 關閉 TensorBoard Logger
logger.close()

print("\n" + "=" * 70)
print("✅ 訓練完成！所有數據已保存")
print(f"📊 TensorBoard: tensorboard --logdir=tensorboard_logs")
print(f"💾 本地檢查點: {final_checkpoint}")
print(f"☁️  Colab 文件會自動同步到 Google Drive")
print("=" * 70)
