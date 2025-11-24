import numpy as np
import matplotlib.pyplot as plt

# 載入數據
data = np.load('data/dataset_kick.npz', allow_pickle=True)

print("=== 數據集基本信息 ===")
print("數據檔案包含的key:", data.files)
print()

# 檢查各項數據的形狀
for key in data.files:
    print(f"{key}:")
    print(f"  - 形狀: {data[key].shape}")
    print(f"  - 數據類型: {data[key].dtype}")
    print()

# 分析observations
obs = data['observations']
print("=== Observations 詳細分析 ===")
print(f"總步數: {len(obs)}")

# 檢查第一個observation的內容
if len(obs) > 0:
    first_obs = obs[0]
    print(f"每個observation的形狀: {first_obs.shape}")
    print(f"Observation範圍: min={first_obs.min():.4f}, max={first_obs.max():.4f}")
    print()

# 分析actions
actions = data['actions']
print("=== Actions 詳細分析 ===")
print(f"總動作數: {len(actions)}")
if len(actions) > 0:
    print(f"動作維度: {actions.shape}")
    print(f"動作範圍: min={actions.min():.4f}, max={actions.max():.4f}")
    print(f"動作統計:")
    for i in range(min(actions.shape[1], 10)):  # 只顯示前10個維度
        print(f"  維度{i}: mean={actions[:, i].mean():.4f}, std={actions[:, i].std():.4f}")
    print()

# 分析episodes
if 'dones' in data.files:
    dones = data['dones']
    episode_ends = np.where(dones)[0]
    num_episodes = len(episode_ends)
    print(f"=== Episode 分析 ===")
    print(f"總episode數: {num_episodes}")
    
    # 計算每個episode的長度
    episode_lengths = []
    start_idx = 0
    for end_idx in episode_ends:
        episode_lengths.append(end_idx + 1 - start_idx)
        start_idx = end_idx + 1
    
    if episode_lengths:
        print(f"Episode長度統計:")
        print(f"  平均: {np.mean(episode_lengths):.1f} 步")
        print(f"  最短: {np.min(episode_lengths)} 步")
        print(f"  最長: {np.max(episode_lengths)} 步")
        print(f"  標準差: {np.std(episode_lengths):.1f}")

# 繪製數據視覺化
plt.figure(figsize=(15, 10))

# 子圖1: 動作分佈
plt.subplot(2, 3, 1)
plt.hist(actions[:, 0], bins=50, alpha=0.7)
plt.title('動作維度0的分佈')
plt.xlabel('動作值')
plt.ylabel('頻率')

# 子圖2: Episode長度
if 'episode_lengths' in locals() and episode_lengths:
    plt.subplot(2, 3, 2)
    plt.plot(episode_lengths, 'o-')
    plt.title('各Episode長度')
    plt.xlabel('Episode編號')
    plt.ylabel('步數')

# 子圖3: 動作時序圖
plt.subplot(2, 3, 3)
plt.plot(actions[:500, 0], label='維度0', alpha=0.7)  # 只顯示前500步
if actions.shape[1] > 1:
    plt.plot(actions[:500, 1], label='維度1', alpha=0.7)
plt.title('動作時序圖 (前500步)')
plt.xlabel('時間步')
plt.ylabel('動作值')
plt.legend()

# 子圖4: 動作相關性
if actions.shape[1] > 1:
    plt.subplot(2, 3, 4)
    plt.scatter(actions[:, 0], actions[:, 1], alpha=0.5)
    plt.title('動作維度0 vs 維度1')
    plt.xlabel('維度0')
    plt.ylabel('維度1')

# 子圖5: 觀測值分佈
if len(obs) > 0 and obs[0].size > 1:
    plt.subplot(2, 3, 5)
    obs_sample = np.array([o.flatten()[:10] for o in obs[:100]])  # 取前100個obs的前10個特徵
    plt.boxplot(obs_sample, labels=[f'特徵{i}' for i in range(min(10, obs_sample.shape[1]))])
    plt.title('觀測值特徵分佈')
    plt.xticks(rotation=45)

# 子圖6: 成功率分析(基於episode長度推斷)
if 'episode_lengths' in locals() and episode_lengths:
    plt.subplot(2, 3, 6)
    # 假設較長的episode更可能是成功的
    success_threshold = np.median(episode_lengths)
    success_rate = np.mean(np.array(episode_lengths) > success_threshold)
    
    labels = ['推測失敗', '推測成功']
    sizes = [1-success_rate, success_rate]
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
    plt.title(f'推測成功率\n(基於episode長度>{success_threshold:.0f})')

plt.tight_layout()
plt.savefig('data_analysis_report.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n=== 數據品質評估 ===")
if 'episode_lengths' in locals() and episode_lengths:
    print(f"數據量評估: {'充足' if num_episodes >= 20 else '建議收集更多'}")
    print(f"多樣性評估: {'良好' if np.std(episode_lengths) > 10 else '建議增加多樣性'}")
    
print("\n分析報告已保存為: data_analysis_report.png")