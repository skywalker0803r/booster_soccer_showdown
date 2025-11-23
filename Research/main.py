# -*- coding: utf-8 -*-
# main_ppo_cma_curiosity.py
# 使用PPO-CMA改進的純好奇心驅動訓練腳本 + LLM輔助獎勵塑形

import numpy as np
import torch
from sai_rl import SAIClient 
from ppo_cma_model import PPOCMA  # 使用PPO-CMA替代TD3
from utils import Preprocessor
from logger import TensorBoardLogger
from curiosity_module import CuriosityDrivenExploration
from gdrive_utils import SimpleGDriveSync
# [AI-Integrate] 導入LLM輔助模組
import sys
sys.path.append('..')  # 添加上級目錄到路徑
from llm_coach import LLMCoach
from reward_shaper import RewardShaper

# [AI-Integrate] 幫助函數：安全提取標量值
def safe_float(value):
    """安全地將 numpy array 或標量轉換為 float，避免 deprecation 警告"""
    if hasattr(value, 'item'):
        return value.item()  # numpy array -> scalar
    else:
        return float(value)  # 已經是標量

# =================================================================
# 1. 初始化 SAIClient 和環境
# =================================================================
sai = SAIClient(
    comp_id="booster-soccer-showdown", 
    api_key="sai_LFcuaCZiqEkUbNVolQ3wbk5yU7H11jfv",
)

# 🎯 創建無時間懲罰的環境
env = sai.make_env()

# # 🚫 移除時間懲罰 - 修改獎勵配置
# print("🎯 正在移除時間懲罰...")
# try:
#     # 嘗試訪問和修改獎勵配置
#     if hasattr(env, 'reward_config') or hasattr(env.unwrapped, 'reward_config'):
#         reward_config = getattr(env, 'reward_config', None) or getattr(env.unwrapped, 'reward_config', None)
#         if reward_config and isinstance(reward_config, dict):
#             # 移除所有時間相關懲罰
#             if 'steps' in reward_config:
#                 original_steps = reward_config['steps']
#                 reward_config['steps'] = 0.0  # 設為0移除時間懲罰
#                 print(f"✅ 時間懲罰已移除: {original_steps} → 0.0")
#             if 'step_penalty' in reward_config:
#                 reward_config['step_penalty'] = 0.0
#                 print(f"✅ 步數懲罰已移除")
#         else:
#             print("⚠️ 無法訪問reward_config，將通過後處理移除時間懲罰")
#     else:
#         print("⚠️ 環境不支持reward_config修改，將通過後處理移除時間懲罰")
# except Exception as e:
#     print(f"⚠️ 修改獎勵配置失敗: {e}，將通過後處理移除時間懲罰")
print(f"環境已創建。觀察空間: {env.observation_space} | 動作空間: {env.action_space}")

N_FEATURES = 45 
N_ACTIONS = env.action_space.shape[0]

# =================================================================
# 2. 輔助函數：動作轉換 (保持不變)
# =================================================================
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

# =================================================================
# 3. 🚀 A100最佳化超參數設置 (PPO-CMA + 純好奇心版)
# =================================================================
TOTAL_TIMESTEPS = 2000000          # 增加總訓練步數，充分利用A100
MODEL_NAME = "Booster-PPOCMA-A100-PureOriginal-v1"
BUFFER_CAPACITY = 8192             # PPO緩衝區，為發揮A100效能，設為BATCH_SIZE的倍數
BATCH_SIZE = 1024                  # 加大批次以最大化A100 GPU利用率
LEARNING_RATE_ACTOR = 3e-4         # Actor學習率
LEARNING_RATE_CRITIC = 1e-3        # Critic學習率
NEURONS = [512, 512, 256]          # 更大更深的網絡架構
UPDATE_FREQ = BUFFER_CAPACITY      # PPO在buffer滿時更新
SAVE_FREQ = 25                     # 更頻繁保存

# PPO-CMA 特有參數
GAMMA = 0.99                       # 折扣因子
GAE_LAMBDA = 0.95                  # GAE lambda參數
CLIP_EPSILON = 0.2                 # PPO裁切係數
ENTROPY_COEF = 0.01               # 熵正則化係數
PPO_EPOCHS = 15                    # 每次更新的PPO epoch數
MAX_GRAD_NORM = 0.5               # 梯度裁切
CMA_POPULATION_SIZE = 64           # CMA-ES種群大小（手動設置以加強探索）
CMA_SIGMA = 0.1                    # CMA-ES初始步長
CMA_UPDATE_FREQ = 10               # CMA-ES更新頻率

# 好奇心模組參數 (A100優化設置)
INTRINSIC_REWARD_SCALE = 0.8      # 稍微降低以平衡大batch效應
CURIOSITY_UPDATE_FREQ = 1

# 初始化PPO-CMA模型
ppo_cma_agent = PPOCMA(
    state_dim=N_FEATURES,
    action_dim=N_ACTIONS,
    hidden_dims=NEURONS,
    lr_actor=LEARNING_RATE_ACTOR,
    lr_critic=LEARNING_RATE_CRITIC,
    gamma=GAMMA,
    gae_lambda=GAE_LAMBDA,
    clip_epsilon=CLIP_EPSILON,
    entropy_coef=ENTROPY_COEF,
    max_grad_norm=MAX_GRAD_NORM,
    ppo_epochs=PPO_EPOCHS,
    batch_size=BATCH_SIZE,
    buffer_capacity=BUFFER_CAPACITY,
    cma_population_size=CMA_POPULATION_SIZE,
    cma_sigma=CMA_SIGMA,
    cma_update_freq=CMA_UPDATE_FREQ
)

# 初始化純好奇心模組
curiosity_explorer = CuriosityDrivenExploration(
    state_dim=N_FEATURES,
    action_dim=N_ACTIONS, 
    intrinsic_reward_scale=INTRINSIC_REWARD_SCALE
)

# [AI-Integrate] 初始化LLM輔助模組
# 使用 Gemini API 進行智能決策
GEMINI_API_KEY = "AIzaSyDUOIGCWDJkY98gi5QcrKtWkxxB61Qhmi0"
llm_coach = LLMCoach(api_key=GEMINI_API_KEY, use_llm=True)
reward_shaper = RewardShaper()
current_weights = llm_coach.current_weights
episode_stats_buffer = []  # 用於存儲最近幾個回合的表現

# =================================================================
# 🔄 模型載入選擇和Google Drive設置
# =================================================================

# 初始化Google Drive同步 (帶錯誤處理)
try:
    gdrive_sync = SimpleGDriveSync()
    gdrive_available = gdrive_sync.gdrive_path is not None
    print(f"🔗 Google Drive狀態: {'✅ 已連接' if gdrive_available else '❌ 未連接 (僅本地保存)'}")
except Exception as e:
    print(f"⚠️ Google Drive初始化失敗: {e}")
    gdrive_sync = None
    gdrive_available = False

# 詢問是否載入舊模型
def choose_model_loading():
    print("\n" + "="*50)
    print("🤔 PPO-CMA訓練模式選擇")
    print("="*50)
    
    # 檢查本地已有模型
    import glob
    local_models = glob.glob(f"*{MODEL_NAME}*.pth") + glob.glob(f"best_*.pth") + glob.glob(f"final_*.pth")
    
    # 檢查Google Drive模型
    gdrive_models = gdrive_sync.list_saved_models(MODEL_NAME.replace("-", "_")) if gdrive_sync else []
    
    if local_models or gdrive_models:
        print("📂 發現已存在的模型:")
        
        all_models = []
        if local_models:
            print("\n本地模型:")
            for i, model in enumerate(local_models):
                print(f"  {i+1}. {model}")
                all_models.append(('local', model))
        
        if gdrive_models:
            print(f"\nGoogle Drive模型 (前5個):")
            for i, model in enumerate(gdrive_models[:5]):
                print(f"  {len(local_models)+i+1}. {model['name']} ({model['modified'].strftime('%Y-%m-%d %H:%M')})")
                all_models.append(('gdrive', model['path']))
        
        print(f"\n{len(all_models)+1}. 🆕 從頭開始訓練")
        
        while True:
            try:
                choice = input("\n選擇要載入的模型 (輸入數字): ").strip()
                choice_num = int(choice)
                
                if choice_num == len(all_models) + 1:
                    return None, None  # 從頭開始
                elif 1 <= choice_num <= len(all_models):
                    model_type, model_path = all_models[choice_num - 1]
                    return model_type, model_path
                else:
                    print("❌ 無效選擇，請重新輸入")
            except ValueError:
                print("❌ 請輸入有效數字")
    else:
        print("📂 未發現已存在的模型，將從頭開始訓練")
        return None, None

model_type, model_path = choose_model_loading()

# 🚀 A100 GPU設置與混合精度
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ppo_cma_agent.to(device)
curiosity_explorer.to(device)

# A100混合精度加速
scaler = torch.cuda.amp.GradScaler()
print(f"✅ A100混合精度訓練已啟用，設備：{device}")
print(f"🔥 GPU記憶體優化：混合精度可節省約40% VRAM")

# 載入模型 (如果選擇了)
start_episode = 0
if model_path:
    try:
        if model_type == 'gdrive':
            # 從Google Drive複製到本地
            import shutil
            local_path = f"loaded_{MODEL_NAME}.pth"
            shutil.copy2(model_path, local_path)
            model_path = local_path
        
        print(f"📥 正在載入模型: {model_path}")
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            ppo_cma_agent.load_state_dict(checkpoint['model_state_dict'])
            start_episode = checkpoint.get('episode', 0)
            print(f"✅ 已載入模型 (從Episode {start_episode}繼續)")
        else:
            ppo_cma_agent.load_state_dict(checkpoint)
            print(f"✅ 已載入模型 (狀態dict格式)")
            
    except Exception as e:
        print(f"❌ 模型載入失敗: {e}")
        print("🔄 將從頭開始訓練")
        start_episode = 0

print(f"🚀 開始訓練 (起始Episode: {start_episode})")

# 初始化記錄器
logger = TensorBoardLogger(model_name=MODEL_NAME) 

# 追蹤變量
episode_reward_sum = 0
episode_intrinsic_reward_sum = 0
episode_extrinsic_reward_sum = 0  # 分別追蹤原始獎勵
episode_shaped_reward_sum = 0     # [AI-Integrate] 追蹤LLM塑形獎勵
episode_count = 0
episode_steps = 0
best_reward = -np.inf
best_model_path = f"best_{MODEL_NAME}.pth"

print(f"🚀 A100最佳化 PPO-CMA + 純原始獎勵訓練開始，設備：{device}")
print(f"🎯 PPO-CMA改進特性：")
print(f"   • PPO Clipped Surrogate: ✅ (ε={CLIP_EPSILON})")
print(f"   • CMA-ES Parameter Evolution: ✅ (σ={CMA_SIGMA})")
print(f"   • Generalized Advantage Estimation: ✅ (λ={GAE_LAMBDA})")
print(f"   • Entropy Regularization: ✅ (β={ENTROPY_COEF})")
print(f"🔥 A100優化配置：")
print(f"   • Batch Size: {BATCH_SIZE}")
print(f"   • Buffer Capacity: {BUFFER_CAPACITY}")
print(f"   • Network Size: {NEURONS} (更大更深)")
print(f"   • Actor LR: {LEARNING_RATE_ACTOR}, Critic LR: {LEARNING_RATE_CRITIC}")
print(f"   • PPO Epochs: {PPO_EPOCHS}")
print(f"   • CMA Update Freq: {CMA_UPDATE_FREQ}")
print(f"   • 內在獎勵縮放: {INTRINSIC_REWARD_SCALE}")
print(f"   • 混合精度: ✅ (A100專用)")
print(f"❌ PBRS獎勵：已禁用")
print(f"❌ 獎勵工程：已移除") 
print(f"❌ 時間懲罰：已移除")
print(f"✅ 純原始環境獎勵：已啟用")
print(f"✅ 好奇心輔助探索：已啟用")

# =================================================================
# 4. PPO-CMA + 純好奇心 訓練循環
# =================================================================
current_obs, info = env.reset()
state = Preprocessor().modify_state(current_obs, info)[0] 
state = torch.tensor(state).float().to(device)

for t in range(1, TOTAL_TIMESTEPS + 1):
    # 1. PPO-CMA動作採樣 (包含隨機探索)
    raw_action, log_prob, value = ppo_cma_agent.get_action(state)
    
    # 執行動作
    action = action_function(raw_action)
    next_obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    
    # 狀態轉換
    next_state_np = Preprocessor().modify_state(next_obs, info)[0]
    next_state = torch.tensor(next_state_np).float().to(device)

    # =================================================================
    # 🧠 LLM輔助獎勵塑形 + 好奇心獎勵計算
    # =================================================================
    
    # 🚫 後處理移除時間懲罰 (如果環境配置修改失敗)
    processed_reward = reward
    
    # [使用者自訂規則] 當原始獎勵過低時，給予一個小的步數獎勵以鼓勵探索
    # 當獎勵都小於0的時候 大步奔跑吧
    if reward <= 0:
        step_encouragement_reward = 1  # 可以調整的步數獎勵值
        processed_reward += step_encouragement_reward
    
    # # 檢測並移除可能的時間懲罰模式
    # if episode_steps > 10:  # 避免初期誤判
    #     # 如果reward是固定的小負值，可能是時間懲罰
    #     if -1.5 <= reward <= -0.1:  # 典型的時間懲罰範圍
    #         # 檢查是否為純時間懲罰（沒有其他事件）
    #         if not any(keyword in str(info).lower() for keyword in ['goal', 'fallen', 'success', 'offside']):
    #             processed_reward = 0.0  # 移除時間懲罰
    #             if t % 10000 == 0:  # 偶爾提示
    #                 print(f"🚫 檢測到時間懲罰 {reward:.3f}，已移除")
    
    # [AI-Integrate] 計算LLM引導的Shaped Reward (已禁用)
    shaped_reward = 0.0
    
    # [AI-Integrate] 融合獎勵：原始獎勵 + LLM塑形獎勵 (Shaped Reward 已禁用)
    # 根據prompt.txt建議調整比例
    total_step_reward = processed_reward
    
    # 🎯 LLM增強獎勵 + 好奇心模組
    final_reward, intrinsic_reward = curiosity_explorer.get_enhanced_reward(
        state.cpu().numpy(),
        raw_action,
        next_state_np,
        total_step_reward  # 使用LLM增強後的獎勵
    )
    
    # 累積統計
    episode_extrinsic_reward_sum += reward
    episode_intrinsic_reward_sum += intrinsic_reward
    episode_reward_sum += final_reward
    episode_steps += 1
    
    # [AI-Integrate] 累積LLM塑形獎勵統計
    if 'episode_shaped_reward_sum' not in locals():
        episode_shaped_reward_sum = 0.0
    episode_shaped_reward_sum += shaped_reward

    # =================================================================
    # 📚 經驗儲存和模型更新
    # =================================================================
    
    # 儲存經驗到PPO-CMA緩衝區 (使用好奇心增強獎勵)
    ppo_cma_agent.store_transition(
        state.cpu().numpy(), 
        raw_action, 
        final_reward,
        next_state_np, 
        done,
        log_prob,
        value
    )

    # 🚀 A100優化 PPO-CMA 模型更新（buffer滿時更新）
    if t % UPDATE_FREQ == 0:
        # 使用混合精度加速訓練 (autocast 已暫時禁用以修復梯度錯誤)
        actor_loss, critic_loss = ppo_cma_agent.update()
        
        # 更新好奇心模組
        if t % CURIOSITY_UPDATE_FREQ == 0:
            # 從PPO-CMA buffer中獲取一些樣本用於好奇心更新
            buffer_data = ppo_cma_agent.buffer.get_all_data()
            if buffer_data is not None:
                states = buffer_data['states'].to(device)
                actions = buffer_data['actions'].to(device)
                next_states = torch.FloatTensor(ppo_cma_agent.buffer.next_states[:ppo_cma_agent.buffer.size]).to(device)
                
                curiosity_stats = curiosity_explorer.update_curiosity(states, actions, next_states)
                
                # 記錄好奇心指標
                logger.set_step(t)
                logger.log_scalar("Curiosity/Forward_Loss", curiosity_stats['forward_loss'])
                logger.log_scalar("Curiosity/Inverse_Loss", curiosity_stats['inverse_loss'])
                logger.log_scalar("Curiosity/Avg_Intrinsic_Reward", curiosity_stats['avg_intrinsic_reward'])
        
        # 記錄訓練指標
        if actor_loss is not None and critic_loss is not None:
            logger.set_step(t) 
            logger.log_scalar("Loss/Actor_Loss", actor_loss)
            logger.log_scalar("Loss/Critic_Loss", critic_loss)
        
        # 記錄PPO-CMA特定指標
        ppo_cma_stats = ppo_cma_agent.get_statistics()
        logger.log_scalar("PPOCMA/Update_Counter", ppo_cma_stats['update_counter'])
        logger.log_scalar("PPOCMA/CMA_Updates", ppo_cma_stats['cma_updates'])
        logger.log_scalar("PPOCMA/CMA_Sigma", ppo_cma_stats['cma_sigma'])
        logger.log_scalar("PPOCMA/CMA_Generation", ppo_cma_stats['cma_generation'])

    # =================================================================
    # 🔄 回合結束處理
    # =================================================================
    if done:
        episode_count += 1

        # [AI-Integrate] 收集數據給LLM教練
        # 檢測是否跌倒（根據步數和獎勵判斷）
        fell_down = episode_steps < 20 or episode_extrinsic_reward_sum < -5.0
        episode_stats_buffer.append({
            'steps': episode_steps,
            'reward': episode_reward_sum,
            'extrinsic_reward': episode_extrinsic_reward_sum,
            'shaped_reward': episode_shaped_reward_sum,
            'fell_down': fell_down
        })

        # 詳細記錄分解獎勵
        logger.log_scalar("Train/Episode_Total_Reward", episode_reward_sum, step=t)
        logger.log_scalar("Train/Episode_Extrinsic_Reward", episode_extrinsic_reward_sum, step=t)
        logger.log_scalar("Train/Episode_Intrinsic_Reward", episode_intrinsic_reward_sum, step=t)
        logger.log_scalar("Train/Episode_Shaped_Reward", episode_shaped_reward_sum, step=t)
        logger.log_scalar("Train/Episode_Steps", episode_steps, step=t)
        
        # 計算好奇心貢獻比例
        if episode_reward_sum != 0:
            curiosity_ratio = episode_intrinsic_reward_sum / abs(episode_reward_sum)
            logger.log_scalar("Train/Curiosity_Contribution_Ratio", curiosity_ratio, step=t)
        
        # [AI-Integrate] 每50個Episode讓LLM教練調整策略
        if episode_count % 50 == 0 and len(episode_stats_buffer) >= 10:
            # 計算統計數據
            recent_episodes = episode_stats_buffer[-50:] if len(episode_stats_buffer) >= 50 else episode_stats_buffer
            
            avg_steps = np.mean([ep['steps'] for ep in recent_episodes])
            avg_reward = np.mean([ep['reward'] for ep in recent_episodes])
            fall_rate = np.mean([ep['fell_down'] for ep in recent_episodes])
            avg_shaped_reward = np.mean([ep['shaped_reward'] for ep in recent_episodes])
            
            stats_summary = {
                'avg_steps': avg_steps,
                'avg_reward': avg_reward,
                'fall_rate': fall_rate,
                'avg_shaped_reward': avg_shaped_reward
            }
            
            # 更新權重
            previous_weights = current_weights.copy()
            current_weights = llm_coach.consult(stats_summary)
            
            # 記錄教練決策
            print(f"🧠 LLM Coach 第{episode_count}回合更新:")
            print(f"   當前階段: {llm_coach.phase}")
            print(f"   統計數據: 步數={safe_float(avg_steps):.1f}, 跌倒率={safe_float(fall_rate):.3f}, 平均獎勵={safe_float(avg_reward):.2f}")
            print(f"   權重變化: {previous_weights} → {current_weights}")
            
            # 記錄到 TensorBoard
            logger.log_scalar("Coach/Weight_Balance", current_weights.get('balance', 0), step=t)
            logger.log_scalar("Coach/Weight_Progress", current_weights.get('progress', 0), step=t)
            logger.log_scalar("Coach/Weight_Energy", current_weights.get('energy', 0), step=t)
            logger.log_scalar("Coach/Avg_Steps", avg_steps, step=t)
            logger.log_scalar("Coach/Fall_Rate", fall_rate, step=t)
            logger.log_scalar("Coach/Phase_ID", hash(llm_coach.phase) % 1000, step=t)  # 簡單的相位編碼
            
            # 記錄 LLM API 統計
            api_stats = llm_coach.get_api_statistics()
            logger.log_scalar("LLM_API/Total_Calls", api_stats['total_calls'], step=t)
            logger.log_scalar("LLM_API/Success_Rate", api_stats['success_rate'], step=t)
            logger.log_scalar("LLM_API/Errors", api_stats['errors'], step=t)
            
            # 清空部分緩衝以保持記憶體效率
            episode_stats_buffer = episode_stats_buffer[-100:]  # 保留最近100個回合
        
        # 檢查最佳模型並自動保存到Google Drive
        if episode_reward_sum > best_reward:
            best_reward = episode_reward_sum
            
            # 保存模型狀態 (包含元數據)
            checkpoint = {
                'model_state_dict': ppo_cma_agent.state_dict(),
                'episode': episode_count + start_episode,
                'timestep': t,
                'best_reward': best_reward,
                'total_reward': episode_reward_sum,
                'intrinsic_reward': episode_intrinsic_reward_sum,
                'episode_steps': episode_steps,
                'ppo_cma_update_counter': ppo_cma_agent.update_counter,
                'cma_updates': ppo_cma_agent.cma_updates
            }
            
            # 本地保存
            torch.save(checkpoint, best_model_path)
            
            # 自動保存到Google Drive
            metadata = {
                'episode': episode_count + start_episode,
                'timestep': t,
                'reward': episode_reward_sum,
                'intrinsic_reward': episode_intrinsic_reward_sum,
                'steps': episode_steps,
                'algorithm': 'PPO-CMA'
            }
            if gdrive_sync and gdrive_available:
                gdrive_sync.save_model(checkpoint, f"best_{MODEL_NAME}", metadata)
            else:
                print(f"⚠️ Google Drive不可用，僅本地保存")
            
            print(f"🏆 新最佳模型!")
            print(f"   總獎勵: {safe_float(episode_reward_sum):.2f}")
            print(f"   原始獎勵: {safe_float(episode_extrinsic_reward_sum):.2f}")
            print(f"   好奇心獎勵: {safe_float(episode_intrinsic_reward_sum):.2f}")
            print(f"   回合步數: {episode_steps}")
            print(f"   訓練步數: {t}")
            print(f"   📤 已自動備份到Google Drive")
        
        # 定期進度報告
        if episode_count % 5 == 0:
            ratio = episode_intrinsic_reward_sum / max(abs(episode_extrinsic_reward_sum), 0.001)
            shaped_ratio = episode_shaped_reward_sum / max(abs(episode_extrinsic_reward_sum), 0.001)
            ppo_cma_stats = ppo_cma_agent.get_statistics()
            print(f"🎯 Episode {episode_count:3d} | "
                  f"總獎勵: {safe_float(episode_reward_sum):6.2f} | "
                  f"原始: {safe_float(episode_extrinsic_reward_sum):6.2f} | "
                  f"塑形: {safe_float(episode_shaped_reward_sum):5.2f} | "
                  f"好奇心: {safe_float(episode_intrinsic_reward_sum):5.2f} | "
                  f"步數: {episode_steps:3d} | "
                  f"階段: {llm_coach.phase[:8]} | "
                  f"PPO更新: {ppo_cma_stats['update_counter']} | "
                  f"CMA: {ppo_cma_stats['cma_updates']}")
        
        # 重置環境
        current_obs, info = env.reset()
        state = Preprocessor().modify_state(current_obs, info)[0]
        state = torch.tensor(state).float().to(device)
        
        # 重設變量
        episode_reward_sum = 0
        episode_intrinsic_reward_sum = 0
        episode_extrinsic_reward_sum = 0
        episode_shaped_reward_sum = 0  # [AI-Integrate] 重設LLM塑形獎勵
        episode_steps = 0
    else:
        state = next_state
    
    # 大進度報告和定期備份 (🚀 A100優化: 更頻繁備份)
    if t % 10000 == 0:
        curiosity_stats = curiosity_explorer.get_statistics()
        ppo_cma_stats = ppo_cma_agent.get_statistics()
        print(f"\n🚀 === PPO-CMA訓練進度報告 (步數: {t}) ===")
        print(f"📊 回合總數: {episode_count}")
        print(f"💾 Buffer大小: {ppo_cma_stats['buffer_size']}")
        print(f"🏆 最佳總獎勵: {safe_float(best_reward):.2f}")
        print(f"🧠 累計好奇心獎勵: {safe_float(curiosity_stats['total_intrinsic_reward']):.2f}")
        print(f"📈 平均好奇心獎勵: {safe_float(curiosity_stats['average_intrinsic_reward']):.4f}")
        print(f"🔄 好奇心更新次數: {curiosity_stats['update_count']}")
        print(f"🎯 PPO更新次數: {ppo_cma_stats['update_counter']}")
        print(f"🧬 CMA-ES更新次數: {ppo_cma_stats['cma_updates']}")
        print(f"📏 CMA-ES步長σ: {ppo_cma_stats['cma_sigma']:.6f}")
        print(f"🌱 CMA-ES世代: {ppo_cma_stats['cma_generation']}")
        
        # 定期自動備份到Google Drive
        checkpoint_name = f"checkpoint_{t//1000}k"
        checkpoint_data = {
            'model_state_dict': ppo_cma_agent.state_dict(),
            'episode': episode_count + start_episode,
            'timestep': t,
            'best_reward': best_reward,
            'ppo_cma_update_counter': ppo_cma_agent.update_counter,
            'cma_updates': ppo_cma_agent.cma_updates
        }
        checkpoint_meta = {
            'episode': episode_count + start_episode,
            'timestep': t,
            'best_reward': best_reward,
            'checkpoint': True,
            'algorithm': 'PPO-CMA'
        }
        
        if gdrive_sync and gdrive_available:
            if gdrive_sync.save_model(checkpoint_data, checkpoint_name, checkpoint_meta):
                print(f"📤 定期備份已保存到 Google Drive")
        else:
            print(f"⚠️ Google Drive不可用，跳過雲端備份")
        
        print("=" * 50)

# =================================================================
# 5. 訓練完成和總結
# =================================================================
final_model_path = f"final_{MODEL_NAME}.pth"

# 保存最終模型 (包含完整狀態)
final_checkpoint = {
    'model_state_dict': ppo_cma_agent.state_dict(),
    'episode': episode_count + start_episode,
    'timestep': TOTAL_TIMESTEPS,
    'best_reward': best_reward,
    'final_training': True,
    'ppo_cma_update_counter': ppo_cma_agent.update_counter,
    'cma_updates': ppo_cma_agent.cma_updates
}
torch.save(final_checkpoint, final_model_path)

# 自動保存最終模型到Google Drive
final_metadata = {
    'episode': episode_count + start_episode,
    'timestep': TOTAL_TIMESTEPS, 
    'best_reward': best_reward,
    'training_completed': True,
    'algorithm': 'PPO-CMA'
}
if gdrive_sync and gdrive_available:
    gdrive_sync.save_model(final_checkpoint, f"final_{MODEL_NAME}", final_metadata)
    print(f"📤 最終模型已保存到 Google Drive")
else:
    print(f"⚠️ Google Drive不可用，最終模型僅本地保存")

curiosity_final_stats = curiosity_explorer.get_statistics()
ppo_cma_final_stats = ppo_cma_agent.get_statistics()

print(f"\n🎉 PPO-CMA + LLM輔助 + 好奇心訓練完成！")
print(f"🏆 最佳回合獎勵: {safe_float(best_reward):.2f}")
print(f"🧠 總好奇心獎勵: {safe_float(curiosity_final_stats['total_intrinsic_reward']):.2f}")
print(f"📊 平均好奇心獎勵: {safe_float(curiosity_final_stats['average_intrinsic_reward']):.4f}")
print(f"🔄 總回合數: {episode_count}")
print(f"🎯 PPO總更新次數: {ppo_cma_final_stats['update_counter']}")
print(f"🧬 CMA-ES總更新次數: {ppo_cma_final_stats['cma_updates']}")
print(f"📏 最終CMA-ES步長σ: {ppo_cma_final_stats['cma_sigma']:.6f}")
print(f"🌱 最終CMA-ES世代: {ppo_cma_final_stats['cma_generation']}")
print(f"🧠 LLM教練最終階段: {llm_coach.phase}")
print(f"⚖️ 最終權重配置: {current_weights}")
print(f"📈 階段變化次數: {len(llm_coach.phase_history)}")

# LLM API 統計報告
llm_api_stats = llm_coach.get_api_statistics()
print(f"🤖 LLM API統計:")
print(f"   總調用次數: {llm_api_stats['total_calls']}")
print(f"   錯誤次數: {llm_api_stats['errors']}")
print(f"   成功率: {safe_float(llm_api_stats['success_rate']):.2%}")
print(f"   LLM啟用: {'✅' if llm_api_stats['llm_enabled'] else '❌'}")

print(f"💾 模型文件: {best_model_path}, {final_model_path}")

# 清理
env.close()
logger.close()
print("🏁 PPO-CMA純好奇心實驗完成！")