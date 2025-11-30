from sac_agent import SACAgent
from utils import Preprocessor
from sai_rl import SAIClient
from tensorboard_logger import SAC_RND_TensorBoardLogger
from gdrive_saver import GoogleDriveAutoSaver
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

# 創建 Google Drive 自動保存器
gdrive_saver = GoogleDriveAutoSaver(
    save_folder_name="SAC_RND_Soccer_Models",
    save_interval=50,  # 每50回合檢查是否保存
    keep_best_n=10,    # 保留最佳10個模型
    auto_auth=True     # 自動認證
)

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
    'gdrive_save_interval': 50,
    'gdrive_keep_best': 10
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
        
        # 顯示 Google Drive 保存統計
        gdrive_stats = agent.get_gdrive_statistics()
        if gdrive_stats:
            print(f"Google Drive 保存統計:")
            print(f"  - 已保存模型數: {gdrive_stats['total_saved']}")
            print(f"  - 最佳保存獎勵: {gdrive_stats['best_reward']:.2f}")
            print(f"  - 最後保存回合: {gdrive_stats['last_save_episode']}")
            if 'latest_save' in gdrive_stats:
                latest = gdrive_stats['latest_save']
                print(f"  - 最新保存: 回合 {latest['episode']}, 獎勵 {latest['reward']:.2f}")
        
        # 保存本地檢查點
        checkpoint_name = f"checkpoint_episode_{episode}.pth"
        agent.save_checkpoint(checkpoint_name, episode, episode_reward)
        
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

# 強制保存最終模型到 Google Drive
print("\n=== 保存最終模型 ===")
final_save_success = agent.force_save_to_gdrive("training_completed")
if final_save_success:
    print("💾 最終模型已保存到 Google Drive")

# 保存最終本地檢查點
final_checkpoint = f"final_checkpoint_{len(total_rewards)}episodes.pth"
agent.save_checkpoint(final_checkpoint)

# 顯示最終 Google Drive 統計
final_gdrive_stats = agent.get_gdrive_statistics()
if final_gdrive_stats:
    print(f"\n📊 Google Drive 最終統計:")
    print(f"  - 總保存模型數: {final_gdrive_stats['total_saved']}")
    print(f"  - 最佳模型獎勵: {final_gdrive_stats['best_reward']:.2f}")
    if final_gdrive_stats['total_saved'] > 0:
        print(f"  - 平均保存獎勵: {final_gdrive_stats.get('average_reward', 0):.2f}")
        latest_save = final_gdrive_stats.get('latest_save', {})
        if latest_save:
            print(f"  - 最後保存: 回合 {latest_save.get('episode', 0)}, 獎勵 {latest_save.get('reward', 0):.2f}")

# 記錄 Google Drive 統計到 TensorBoard
if final_gdrive_stats:
    gdrive_summary = f"""
    Google Drive 保存總結:
    - 總保存模型數: {final_gdrive_stats['total_saved']}
    - 最佳模型獎勵: {final_gdrive_stats['best_reward']:.2f}
    - 保存間隔: 每 {gdrive_saver.save_interval} 回合
    - 保留模型數: {gdrive_saver.keep_best_n}
    """
    logger.log_text("GoogleDrive_Final_Summary", gdrive_summary)

# 關閉 TensorBoard Logger
logger.close()

print("\n" + "=" * 70)
print("✅ 訓練完成！所有數據已保存")
print(f"📊 TensorBoard: tensorboard --logdir=tensorboard_logs")
print(f"💾 本地檢查點: {final_checkpoint}")
print(f"☁️  Google Drive: {final_gdrive_stats['total_saved'] if final_gdrive_stats else 0} 個模型已上傳")
print("=" * 70)
