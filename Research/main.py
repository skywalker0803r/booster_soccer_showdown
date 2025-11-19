# main.py

import numpy as np
import torch
from sai_rl import SAIClient 
from ddpg_model import DDPG_FF, ReplayBuffer # 匯入 DDPG 類和 Replay Buffer 類
from utils import Preprocessor # 負責將原始 obs 做處理
from logger import TensorBoardLogger # 匯入 TensorBoard 紀錄器

# =================================================================
# 1. 初始化 SAIClient 和環境
# =================================================================
sai = SAIClient(
    comp_id="booster-soccer-showdown" , 
    api_key="sai_LFcuaCZiqEkUbNVolQ3wbk5yU7H11jfv",
)

# 創建環境
env = sai.make_env()
print(f"環境已創建。觀察空間: {env.observation_space} | 動作空間: {env.action_space}")

# 狀態維度 (Preprocessor 輸出)
N_FEATURES = 45 
# 動作維度
N_ACTIONS = env.action_space.shape[0]

# =================================================================
# 2. 輔助函數：動作轉換 和 探索噪音
# =================================================================

# 將原始 policy 輸出 [-1, 1] 映射到環境動作空間
def action_function(policy):
    expected_bounds = [-1, 1]
    action_percent = (policy - expected_bounds[0]) / (
        expected_bounds[1] - expected_bounds[0]
    )
    bounded_percent = np.minimum(np.maximum(action_percent, 0), 1)
    return (
        env.action_space.low
        + (env.action_space.high - env.action_space.low) * bounded_percent
    )

# DDPG 標準的 Ornstein-Uhlenbeck 過程噪音
class OUNoise:
    def __init__(self, mu=0.0, sigma=0.15, theta=0.2, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()
    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros(N_ACTIONS)
    def __call__(self):
        # Ornstein-Uhlenbeck 過程公式
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
            self.sigma * np.sqrt(self.dt) * np.random.normal(size=N_ACTIONS)
        self.x_prev = x
        return x

# =================================================================
# 3. 超參數和初始化
# =================================================================
TOTAL_TIMESTEPS = 2000_000 # 設定總訓練步數
MODEL_NAME = "Booster-DDPG-FF-v1" 
BUFFER_CAPACITY = 1000_000
BATCH_SIZE = 256
LEARNING_RATE = 3e-4 
NEURONS = [256, 256] 
EXPLORE_STEPS = 5000 
UPDATE_FREQ = 1 # 每採集一步經驗，更新模型一次

# 初始化模型和 Buffer
ddpg_agent = DDPG_FF(
    N_FEATURES, 
    env.action_space, 
    NEURONS, 
    torch.nn.functional.relu, # 使用 ReLU 作為隱藏層激活函數
    LEARNING_RATE
)
replay_buffer = ReplayBuffer(BUFFER_CAPACITY, (N_FEATURES,), N_ACTIONS)
noise_process = OUNoise() 

# 確保模型在 CUDA 上 (如果可用)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ddpg_agent.to(device)

# 初始化 TensorBoard 紀錄器
logger = TensorBoardLogger(model_name=MODEL_NAME) #

# 初始化回合獎勵追蹤
episode_reward_sum = 0 #


# =================================================================
# 4. 自定義 DDPG 訓練循環
# =================================================================
current_obs, info = env.reset()
# 使用 Preprocessor 轉換狀態
state = Preprocessor().modify_state(current_obs, info)[0] 
state = torch.tensor(state).float().to(device)

print(f"DDPG 訓練開始，設備：{device}")

for t in range(1, TOTAL_TIMESTEPS + 1):
    # 1. 採集動作 (探索階段加噪音)
    with torch.no_grad():
        raw_action_tensor = ddpg_agent(state.unsqueeze(0)) # 取得 Actor 輸出 [-1, 1] 範圍
    raw_action = raw_action_tensor.cpu().numpy().flatten()
    
    # 添加 Ornstein-Uhlenbeck 噪音進行探索
    if t < EXPLORE_STEPS:
        raw_action += noise_process()
    else:
        # 隨著訓練進行，噪音衰減
        noise_process.sigma = max(0.01, 0.15 * (1 - (t - EXPLORE_STEPS) / (TOTAL_TIMESTEPS - EXPLORE_STEPS)))
        raw_action += noise_process()

    # 動作約束和環境執行
    action = action_function(raw_action)
    next_obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    
    # 累積回合獎勵
    episode_reward_sum += reward #

    # 狀態轉換
    next_state_np = Preprocessor().modify_state(next_obs, info)[0]
    next_state = torch.tensor(next_state_np).float().to(device)

    # 2. 儲存經驗到 Buffer
    # 注意：儲存的 action 是未經 action_function 處理的 [-1, 1] 範圍的 raw_action
    replay_buffer.add(
        state.cpu().numpy(), 
        raw_action, 
        reward, 
        next_state_np, 
        done
    )

    # 3. 模型更新
    if replay_buffer.size > BATCH_SIZE and t % UPDATE_FREQ == 0:
        # 從 Buffer 採樣，並移動到 GPU
        states, actions, rewards, next_states, dones = replay_buffer.sample(BATCH_SIZE)
        
        states = torch.tensor(states).float().to(device)
        actions = torch.tensor(actions).float().to(device)
        rewards = torch.tensor(rewards).float().to(device)
        next_states = torch.tensor(next_states).float().to(device)
        dones = torch.tensor(dones).float().to(device)
        
        critic_loss, actor_loss = ddpg_agent.model_update(states, actions, rewards, next_states, dones)
        
        # 紀錄損失
        logger.set_step(t) #
        logger.log_scalar("Loss/Critic_Loss", critic_loss) #
        logger.log_scalar("Loss/Actor_Loss", actor_loss) #

    # 4. 準備下一循環
    if done:
        # 紀錄回合總獎勵
        logger.log_scalar("Train/Episode_Reward", episode_reward_sum, step=t) #
        
        current_obs, info = env.reset()
        state = Preprocessor().modify_state(current_obs, info)[0]
        state = torch.tensor(state).float().to(device)
        noise_process.reset()
        
        # 重設回合獎勵
        episode_reward_sum = 0 #
    else:
        state = next_state
    
    if t % 50000 == 0:
        print(f"Time Step: {t}/{TOTAL_TIMESTEPS} | Buffer Size: {replay_buffer.size}")

# =================================================================
# 5. 模型儲存和評估
# =================================================================
# DDPG 自製模型需要手動保存其 Actor 和 Critic 的狀態
torch.save(ddpg_agent.state_dict(), f"{MODEL_NAME}.pth")
print(f"模型已保存為 {MODEL_NAME}.pth")

# 由於 sai.benchmark 不支持自定義模型，我們將嘗試提交 Actor 模型進行評估
# **已修改為傳遞 ddpg_agent 實例，以使用內建的 .cpu() 處理**
try:
    sai.benchmark(ddpg_agent, action_function, Preprocessor) 
except Exception as e:
    print(f"SAI Benchmark 失敗 (預期): {e}")
    print("請嘗試直接將訓練好的模型權重提交給 SAI 平台。")


# 關閉環境
env.close()
# 關閉 Logger
logger.close() #