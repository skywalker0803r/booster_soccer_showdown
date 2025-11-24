#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BC預訓練 + 純PPO訓練系統
簡化版本：移除好奇心、獎勵塑形、LLM教練、CMA-ES等複雜組件
專注於測試專家數據預訓練的效果
"""

import numpy as np
import torch
import os
import sys
from sai_rl import SAIClient 
from ppo_cma_model import PPOCMA
from utils import Preprocessor
from logger import TensorBoardLogger
from simple_bc_integration import BCPretrainer
from gdrive_utils import SimpleGDriveSync
import glob

def safe_float(value):
    """安全地將numpy array或標量轉換為float"""
    if hasattr(value, 'item'):
        return value.item()
    else:
        return float(value)

# =================================================================
# 1. 環境設置
# =================================================================
print("🚀 初始化BC預訓練 + 純PPO系統")

# 創建SAI環境
sai = SAIClient(
    comp_id="booster-soccer-showdown", 
    api_key="sai_LFcuaCZiqEkUbNVolQ3wbk5yU7H11jfv",
)
env = sai.make_env()
print(f"✅ 環境已創建 | 觀察: {env.observation_space} | 動作: {env.action_space}")

# =================================================================
# 2. 超參數配置
# =================================================================
TOTAL_TIMESTEPS = 1000000         # 訓練步數 (簡化版減少50%)
MODEL_NAME = "BC-PPO-Simple"
BATCH_SIZE = 512                  # 簡化版減小batch size
BUFFER_CAPACITY = 4096            # 對應減小buffer
LEARNING_RATE_ACTOR = 3e-4
LEARNING_RATE_CRITIC = 1e-3
HIDDEN_DIMS = [256, 256]        # 簡化網絡結構
SAVE_FREQ = 50

# PPO參數
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_EPSILON = 0.2
ENTROPY_COEF = 0.01
PPO_EPOCHS = 5                    # 減少PPO epochs
MAX_GRAD_NORM = 0.5

# =================================================================
# 3. 動作轉換函數
# =================================================================
def action_function(policy):
    """將策略輸出轉換為環境動作"""
    expected_bounds = [-1, 1]
    action_percent = (policy - expected_bounds[0]) / (expected_bounds[1] - expected_bounds[0])
    bounded_percent = np.minimum(np.maximum(action_percent, 0), 1)
    return env.action_space.low + (env.action_space.high - env.action_space.low) * bounded_percent

# =================================================================
# 4. 模型初始化
# =================================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🔥 使用設備: {device}")

# 創建簡化版PPO模型 (移除CMA-ES參數)
ppo_agent = PPOCMA(
    state_dim=45,
    action_dim=env.action_space.shape[0],
    lr_actor=LEARNING_RATE_ACTOR,
    lr_critic=LEARNING_RATE_CRITIC,
    gamma=GAMMA,
    gae_lambda=GAE_LAMBDA,
    clip_epsilon=CLIP_EPSILON,
    entropy_coef=ENTROPY_COEF,
    hidden_dims=HIDDEN_DIMS,
    buffer_capacity=BUFFER_CAPACITY,
    batch_size=BATCH_SIZE,
    ppo_epochs=PPO_EPOCHS,
    max_grad_norm=MAX_GRAD_NORM,
    cma_population_size=0,  # 禁用CMA-ES
    cma_sigma=0.0,          # 禁用CMA-ES
    cma_update_freq=999999  # 禁用CMA-ES
)

# 手動將模型移到正確的設備
ppo_agent = ppo_agent.to(device)

# 手動禁用CMA-ES機制
ppo_agent.use_cma = False
print("✅ CMA-ES機制已禁用，使用純PPO")

print("✅ PPO模型已創建")

# =================================================================
# 4.5. Google Drive設置和模型選擇
# =================================================================

# 初始化Google Drive同步
try:
    gdrive_sync = SimpleGDriveSync()
    gdrive_available = gdrive_sync.gdrive_path is not None
    print(f"🔗 Google Drive: {'✅ 已連接' if gdrive_available else '❌ 未連接'}")
except Exception as e:
    print(f"⚠️ Google Drive初始化失敗: {e}")
    gdrive_sync = None
    gdrive_available = False

def choose_model_loading():
    """選擇載入模型或重新開始"""
    print("\n" + "="*50)
    print("🤔 BC-PPO訓練模式選擇")
    print("="*50)
    
    # 檢查本地模型
    local_models = (glob.glob(f"*{MODEL_NAME}*.pth") + 
                   glob.glob(f"best_*.pth") + 
                   glob.glob(f"final_*.pth") +
                   glob.glob(f"checkpoint_*.pth"))
    
    # 檢查Google Drive模型
    gdrive_models = []
    if gdrive_sync and gdrive_available:
        try:
            gdrive_models = gdrive_sync.list_saved_models(MODEL_NAME.replace("-", "_"))
        except:
            gdrive_models = []
    
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
        
        print(f"\n{len(all_models)+1}. 🆕 從頭開始訓練 (包含BC預訓練)")
        
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

# 選擇模型載入方式
model_type, model_path = choose_model_loading()

# 載入模型 (如果選擇了)
start_episode = 0
if model_path:
    try:
        if model_type == 'gdrive':
            print(f"⬇️ 從Google Drive載入模型: {model_path}")
            checkpoint = gdrive_sync.load_model(model_path)
        else:
            print(f"📂 載入本地模型: {model_path}")
            checkpoint = torch.load(model_path, map_location=device)
        
        if checkpoint:
            ppo_agent.load_state_dict(checkpoint['model_state_dict'])
            start_episode = checkpoint.get('episode', 0)
            start_timestep = checkpoint.get('timestep', 0)
            previous_best = checkpoint.get('best_reward', -np.inf)
            
            print(f"✅ 模型載入成功!")
            print(f"   起始Episode: {start_episode}")
            print(f"   起始步數: {start_timestep}")
            print(f"   歷史最佳獎勵: {previous_best:.2f}")
            
            # 如果載入模型，跳過BC預訓練
            skip_bc = True
        else:
            print("❌ 模型載入失敗，將從頭開始")
            skip_bc = False
    except Exception as e:
        print(f"❌ 載入模型時出錯: {e}")
        print("將從頭開始訓練")
        skip_bc = False
else:
    skip_bc = False

# =================================================================
# 5. BC預訓練 
# =================================================================
expert_data_path = "../data/dataset_kick.npz"
if not skip_bc and os.path.exists(expert_data_path):
    print("🎯 開始BC預訓練...")
    bc_pretrainer = BCPretrainer(ppo_agent, expert_data_path, device)
    bc_loss = bc_pretrainer.pretrain(epochs=50)
    
    # 評估BC性能
    bc_performance = bc_pretrainer.evaluate_bc_performance()
    if bc_performance:
        print(f"📊 BC預訓練完成:")
        print(f"   最終損失: {bc_loss:.6f}")
        print(f"   MSE: {bc_performance['mse']:.6f}")
        print(f"   MAE: {bc_performance['mae']:.6f}")  
        print(f"   平均相關係數: {bc_performance['avg_correlation']:.4f}")
    
    print("✅ BC預訓練完成，開始PPO微調...")
elif skip_bc:
    print("🔄 載入已訓練模型，跳過BC預訓練")
else:
    print("⚠️ 未找到專家數據，僅使用PPO訓練")

# =================================================================
# 6. 訓練設置
# =================================================================
logger = TensorBoardLogger(f"simplified_bc_ppo_{MODEL_NAME}")
episode_count = 0
best_reward = -np.inf
t = 0

# 初始化環境
current_obs, info = env.reset()
state = Preprocessor().modify_state(current_obs, info)[0]
state = torch.tensor(state, dtype=torch.float32).to(device)

# 訓練變數
episode_reward = 0
episode_steps = 0

# 初始化最佳獎勵 (考慮載入的模型)
if model_path and 'previous_best' in locals():
    best_reward = previous_best
    episode_count = start_episode

print(f"🚀 開始PPO訓練 ({TOTAL_TIMESTEPS:,} 步)")
print("="*60)

# =================================================================
# 7. 主訓練循環
# =================================================================
for t in range(1, TOTAL_TIMESTEPS + 1):
    # PPO動作選擇
    with torch.no_grad():
        action, log_prob, value = ppo_agent.get_action(state.cpu().numpy())
    
    # 執行動作
    bounded_action = action_function(action)
    next_obs, reward, done, _, info = env.step(bounded_action)
    
    # 處理下一狀態
    if not done:
        next_state = Preprocessor().modify_state(next_obs, info)[0]
        next_state = torch.tensor(next_state, dtype=torch.float32).to(device)
    else:
        next_state = None
    
    # 存儲經驗 (只使用環境原生獎勵)
    ppo_agent.store_transition(
        state.cpu().numpy(), action, reward, 
        next_state.cpu().numpy() if next_state is not None else None, 
        done, log_prob, value
    )
    
    # 累計統計
    episode_reward += reward
    episode_steps += 1
    
    # 更新模型 (當緩衝區有足夠數據時)
    if ppo_agent.buffer.size >= BATCH_SIZE:
        actor_loss, critic_loss, candidate_params = ppo_agent.update()
        
        # 記錄訓練信息
        if actor_loss is not None:
            logger.log({
                'ppo/policy_loss': actor_loss,
                'ppo/value_loss': critic_loss,
                'training/learning_rate': LEARNING_RATE_ACTOR,
                'environment/episode_length': episode_steps,
                'environment/episode_reward': episode_reward,
                'ppo/update_counter': ppo_agent.update_counter
            }, step=t)
    
    # Episode結束處理
    if done:
        episode_count += 1
        
        # 更新最佳獎勵
        if episode_reward > best_reward:
            best_reward = episode_reward
            # 保存最佳模型
            best_model_data = {
                'model_state_dict': ppo_agent.state_dict(),
                'episode': episode_count,
                'timestep': t,
                'best_reward': best_reward,
                'algorithm': 'BC-PPO'
            }
            
            # 本地保存
            torch.save(best_model_data, f"best_{MODEL_NAME}.pth")
            
            # Google Drive保存
            if gdrive_sync and gdrive_available:
                try:
                    gdrive_sync.save_model(
                        best_model_data, 
                        f"best_{MODEL_NAME}", 
                        {
                            'episode': episode_count,
                            'timestep': t,
                            'best_reward': best_reward,
                            'model_type': 'best',
                            'algorithm': 'BC-PPO'
                        }
                    )
                    print(f"📤 最佳模型已上傳Google Drive (獎勵: {best_reward:.2f})")
                except Exception as e:
                    print(f"⚠️ Google Drive上傳失敗: {e}")
        
        # 輸出進度
        if episode_count % 10 == 0:
            print(f"Episode {episode_count:4d} | "
                  f"獎勵: {episode_reward:7.2f} | "
                  f"最佳: {best_reward:7.2f} | "
                  f"步數: {episode_steps:3d} | "
                  f"時間步: {t:7d}")
        
        # 重置環境
        current_obs, info = env.reset()
        state = Preprocessor().modify_state(current_obs, info)[0]
        state = torch.tensor(state, dtype=torch.float32).to(device)
        
        # 重置變數
        episode_reward = 0
        episode_steps = 0
    else:
        state = next_state
    
    # 定期保存和報告
    if t % 50000 == 0:
        print(f"\n📊 === 訓練進度報告 (步數: {t:,}) ===")
        print(f"回合總數: {episode_count}")
        print(f"最佳獎勵: {best_reward:.2f}")
        print(f"PPO更新次數: {ppo_agent.update_counter}")
        print("="*50)
        
        # 定期保存checkpoint
        checkpoint_data = {
            'model_state_dict': ppo_agent.state_dict(),
            'episode': episode_count,
            'timestep': t,
            'best_reward': best_reward,
            'algorithm': 'BC-PPO'
        }
        
        # 本地保存
        checkpoint_name = f"checkpoint_{MODEL_NAME}_step_{t}"
        torch.save(checkpoint_data, f"{checkpoint_name}.pth")
        
        # Google Drive保存
        if gdrive_sync and gdrive_available:
            try:
                gdrive_sync.save_model(
                    checkpoint_data,
                    checkpoint_name,
                    {
                        'episode': episode_count,
                        'timestep': t,
                        'best_reward': best_reward,
                        'model_type': 'checkpoint',
                        'algorithm': 'BC-PPO'
                    },
                    add_timestamp=False
                )
                print(f"📤 Checkpoint已上傳Google Drive")
            except Exception as e:
                print(f"⚠️ Google Drive上傳失敗: {e}")

# =================================================================
# 8. 訓練完成
# =================================================================
print(f"\n🎉 訓練完成!")
print(f"🏆 最佳獎勵: {best_reward:.2f}")
print(f"📊 總回合數: {episode_count}")
print(f"🎯 PPO更新次數: {ppo_agent.update_counter}")

# 保存最終模型
final_checkpoint = {
    'model_state_dict': ppo_agent.state_dict(),
    'episode': episode_count,
    'timestep': TOTAL_TIMESTEPS,
    'best_reward': best_reward,
    'final_training': True,
    'algorithm': 'BC-PPO'
}

# 本地保存
torch.save(final_checkpoint, f"final_{MODEL_NAME}.pth")

# Google Drive保存
if gdrive_sync and gdrive_available:
    try:
        gdrive_sync.save_model(
            final_checkpoint,
            f"final_{MODEL_NAME}",
            {
                'episode': episode_count,
                'timestep': TOTAL_TIMESTEPS,
                'best_reward': best_reward,
                'training_completed': True,
                'model_type': 'final',
                'algorithm': 'BC-PPO'
            }
        )
        print(f"📤 最終模型已上傳Google Drive")
    except Exception as e:
        print(f"⚠️ 最終模型Google Drive上傳失敗: {e}")

logger.close()
print(f"✅ 模型已保存: final_{MODEL_NAME}.pth")
print("🏁 BC預訓練 + 純PPO訓練完成!")