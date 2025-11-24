import numpy as np

# 載入數據
data = np.load('data/dataset_kick.npz', allow_pickle=True)
dones = data['done']

# 找到所有episode的結束點
episode_ends = np.where(dones)[0]
num_episodes = len(episode_ends)

print("=== Episode 詳細分析 ===")
print(f"總Episode數: {num_episodes}")

# 計算每個episode的長度
episode_lengths = []
start_idx = 0
for i, end_idx in enumerate(episode_ends):
    length = end_idx + 1 - start_idx
    episode_lengths.append(length)
    print(f"Episode {i+1}: {length} 步")
    start_idx = end_idx + 1

print(f"\nEpisode長度統計:")
print(f"  平均: {np.mean(episode_lengths):.1f} 步")
print(f"  最短: {np.min(episode_lengths)} 步")
print(f"  最長: {np.max(episode_lengths)} 步")
print(f"  標準差: {np.std(episode_lengths):.1f}")

# 推測成功率（基於episode長度）
median_length = np.median(episode_lengths)
long_episodes = np.sum(np.array(episode_lengths) > median_length)
short_episodes = num_episodes - long_episodes

print(f"\n基於Episode長度的成功率推測:")
print(f"  中位數長度: {median_length:.1f} 步")
print(f"  長Episode (可能成功): {long_episodes} 個")
print(f"  短Episode (可能失敗): {short_episodes} 個")
print(f"  推測成功率: {long_episodes/num_episodes:.1%}")

print(f"\n總數據點: {len(data['observations'])} 個時間步")
print(f"數據密度: {len(data['observations'])/num_episodes:.1f} 步/episode")