# -*- coding: utf-8 -*-
# main_pure_curiosity.py
# 純好奇心驅動的DDPG訓練腳本 (不使用OU噪音和PBRS)

import numpy as np
import torch
from sai_rl import SAIClient 
from ddpg_model import DDPG_FF, ReplayBuffer
from utils import Preprocessor  # 不導入calculate_potential
from logger import TensorBoardLogger
from curiosity_module import CuriosityDrivenExploration
from gdrive_utils import SimpleGDriveSync

# =================================================================
# 1. 初始化 SAIClient 和環境
# =================================================================
sai = SAIClient(
    comp_id="booster-soccer-showdown" , 
    api_key="sai_LFcuaCZiqEkUbNVolQ3wbk5yU7H11jfv",
)

env = sai.make_env()
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
# 3. 超參數和初始化 (純好奇心版)
# =================================================================
TOTAL_TIMESTEPS = 1000000
MODEL_NAME = "Booster-DDPG-PureCuriosity-v1"
BUFFER_CAPACITY = 1000000
BATCH_SIZE = 256
LEARNING_RATE = 1e-4
NEURONS = [256, 256] 
UPDATE_FREQ = 1
SAVE_FREQ = 50

# 好奇心模組參數 (關鍵設置)
INTRINSIC_REWARD_SCALE = 1.0  # 增大係數，因為只依賴好奇心
CURIOSITY_UPDATE_FREQ = 1

# 初始化模型
ddpg_agent = DDPG_FF(
    N_FEATURES, 
    env.action_space, 
    NEURONS, 
    torch.nn.functional.relu,
    LEARNING_RATE
)
replay_buffer = ReplayBuffer(BUFFER_CAPACITY, (N_FEATURES,), N_ACTIONS)

# 初始化純好奇心模組
curiosity_explorer = CuriosityDrivenExploration(
    state_dim=N_FEATURES,
    action_dim=N_ACTIONS, 
    intrinsic_reward_scale=INTRINSIC_REWARD_SCALE
)

# =================================================================
# 🔄 模型載入選擇和Google Drive設置
# =================================================================

# 初始化Google Drive同步
gdrive_sync = SimpleGDriveSync()

# 詢問是否載入舊模型
def choose_model_loading():
    print("\n" + "="*50)
    print("🤔 訓練模式選擇")
    print("="*50)
    
    # 檢查本地已有模型
    import glob
    local_models = glob.glob(f"*{MODEL_NAME}*.pth") + glob.glob(f"best_*.pth") + glob.glob(f"final_*.pth")
    
    # 檢查Google Drive模型
    gdrive_models = gdrive_sync.list_saved_models(MODEL_NAME.replace("-", "_"))
    
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

# GPU設置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ddpg_agent.to(device)
curiosity_explorer.to(device)

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
        checkpoint = torch.load(model_path, map_location=device)
        
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            ddpg_agent.load_state_dict(checkpoint['model_state_dict'])
            start_episode = checkpoint.get('episode', 0)
            print(f"✅ 已載入模型 (從Episode {start_episode}繼續)")
        else:
            ddpg_agent.load_state_dict(checkpoint)
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
episode_count = 0
episode_steps = 0
best_reward = -np.inf
best_model_path = f"best_{MODEL_NAME}.pth"

print(f"🧠 純好奇心DDPG訓練開始，設備：{device}")
print(f"🔥 內在獎勵縮放係數：{INTRINSIC_REWARD_SCALE}")
print(f"❌ OU噪音：已禁用")
print(f"❌ PBRS獎勵：已禁用") 
print(f"✅ 純好奇心探索：已啟用")

# =================================================================
# 4. 純好奇心 DDPG 訓練循環
# =================================================================
current_obs, info = env.reset()
state = Preprocessor().modify_state(current_obs, info)[0] 
state = torch.tensor(state).float().to(device)

for t in range(1, TOTAL_TIMESTEPS + 1):
    # 1. 採集動作 (不添加OU噪音，純依賴好奇心探索)
    with torch.no_grad():
        raw_action_tensor = ddpg_agent(state.unsqueeze(0))
    raw_action = raw_action_tensor.cpu().numpy().flatten()
    
    # 🚫 不添加OU噪音 - 純依賴好奇心驅動的探索
    
    # 執行動作
    action = action_function(raw_action)
    next_obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    
    # 狀態轉換
    next_state_np = Preprocessor().modify_state(next_obs, info)[0]
    next_state = torch.tensor(next_state_np).float().to(device)

    # =================================================================
    # 🧠 純好奇心獎勵計算
    # =================================================================
    
    # 🚫 不使用PBRS獎勵
    # 只使用：原始獎勵 + 好奇心獎勵
    
    final_reward, intrinsic_reward = curiosity_explorer.get_enhanced_reward(
        state.cpu().numpy(),
        raw_action,
        next_state_np,
        reward  # 直接使用原始獎勵，不加PBRS
    )
    
    # 累積統計
    episode_extrinsic_reward_sum += reward
    episode_intrinsic_reward_sum += intrinsic_reward
    episode_reward_sum += final_reward
    episode_steps += 1

    # =================================================================
    # 📚 經驗儲存和模型更新
    # =================================================================
    
    # 儲存經驗 (使用好奇心增強獎勵)
    replay_buffer.add(
        state.cpu().numpy(), 
        raw_action, 
        final_reward,
        next_state_np, 
        done
    )

    # DDPG 模型更新
    if replay_buffer.size > BATCH_SIZE and t % UPDATE_FREQ == 0:
        states, actions, rewards, next_states, dones = replay_buffer.sample(BATCH_SIZE)
        
        states = torch.tensor(states).float().to(device)
        actions = torch.tensor(actions).float().to(device)
        rewards = torch.tensor(rewards).float().to(device)
        next_states = torch.tensor(next_states).float().to(device)
        dones = torch.tensor(dones).float().to(device)
        
        critic_loss, actor_loss = ddpg_agent.model_update(states, actions, rewards, next_states, dones)
        
        # 更新好奇心模組
        if t % CURIOSITY_UPDATE_FREQ == 0:
            curiosity_stats = curiosity_explorer.update_curiosity(states, actions, next_states)
            
            # 記錄好奇心指標
            logger.set_step(t)
            logger.log_scalar("Curiosity/Forward_Loss", curiosity_stats['forward_loss'])
            logger.log_scalar("Curiosity/Inverse_Loss", curiosity_stats['inverse_loss'])
            logger.log_scalar("Curiosity/Avg_Intrinsic_Reward", curiosity_stats['avg_intrinsic_reward'])
        
        # 記錄訓練指標
        logger.set_step(t) 
        logger.log_scalar("Loss/Critic_Loss", critic_loss) 
        logger.log_scalar("Loss/Actor_Loss", actor_loss) 

    # =================================================================
    # 🔄 回合結束處理
    # =================================================================
    if done:
        episode_count += 1

        # 詳細記錄分解獎勵
        logger.log_scalar("Train/Episode_Total_Reward", episode_reward_sum, step=t)
        logger.log_scalar("Train/Episode_Extrinsic_Reward", episode_extrinsic_reward_sum, step=t)
        logger.log_scalar("Train/Episode_Intrinsic_Reward", episode_intrinsic_reward_sum, step=t)
        logger.log_scalar("Train/Episode_Steps", episode_steps, step=t)
        
        # 計算好奇心貢獻比例
        if episode_reward_sum != 0:
            curiosity_ratio = episode_intrinsic_reward_sum / abs(episode_reward_sum)
            logger.log_scalar("Train/Curiosity_Contribution_Ratio", curiosity_ratio, step=t)
        
        # 檢查最佳模型並自動保存到Google Drive
        if episode_reward_sum > best_reward:
            best_reward = episode_reward_sum
            
            # 保存模型狀態 (包含元數據)
            checkpoint = {
                'model_state_dict': ddpg_agent.state_dict(),
                'episode': episode_count + start_episode,
                'timestep': t,
                'best_reward': best_reward,
                'total_reward': episode_reward_sum,
                'intrinsic_reward': episode_intrinsic_reward_sum,
                'episode_steps': episode_steps
            }
            
            # 本地保存
            torch.save(checkpoint, best_model_path)
            
            # 自動保存到Google Drive
            metadata = {
                'episode': episode_count + start_episode,
                'timestep': t,
                'reward': episode_reward_sum,
                'intrinsic_reward': episode_intrinsic_reward_sum,
                'steps': episode_steps
            }
            gdrive_sync.save_model(checkpoint, f"best_{MODEL_NAME}", metadata)
            
            print(f"🏆 新最佳模型!")
            print(f"   總獎勵: {episode_reward_sum:.2f}")
            print(f"   原始獎勵: {episode_extrinsic_reward_sum:.2f}")
            print(f"   好奇心獎勵: {episode_intrinsic_reward_sum:.2f}")
            print(f"   回合步數: {episode_steps}")
            print(f"   訓練步數: {t}")
            print(f"   📤 已自動備份到Google Drive")
        
        # 定期進度報告
        if episode_count % 5 == 0:
            ratio = episode_intrinsic_reward_sum / max(abs(episode_extrinsic_reward_sum), 0.001)
            print(f"🧠 Episode {episode_count:3d} | "
                  f"總獎勵: {episode_reward_sum:6.2f} | "
                  f"原始: {episode_extrinsic_reward_sum:6.2f} | "
                  f"好奇心: {episode_intrinsic_reward_sum:5.2f} | "
                  f"步數: {episode_steps:3d} | "
                  f"比例: {ratio:.2f}")
        
        # 重置環境
        current_obs, info = env.reset()
        state = Preprocessor().modify_state(current_obs, info)[0]
        state = torch.tensor(state).float().to(device)
        
        # 重設變量
        episode_reward_sum = 0
        episode_intrinsic_reward_sum = 0
        episode_extrinsic_reward_sum = 0
        episode_steps = 0
    else:
        state = next_state
    
    # 大進度報告和定期備份
    if t % 100000 == 0:
        curiosity_stats = curiosity_explorer.get_statistics()
        print(f"\n🚀 === 訓練進度報告 (步數: {t}) ===")
        print(f"📊 回合總數: {episode_count}")
        print(f"💾 Buffer大小: {replay_buffer.size}")
        print(f"🏆 最佳總獎勵: {best_reward:.2f}")
        print(f"🧠 累計好奇心獎勵: {curiosity_stats['total_intrinsic_reward']:.2f}")
        print(f"📈 平均好奇心獎勵: {curiosity_stats['average_intrinsic_reward']:.4f}")
        print(f"🔄 好奇心更新次數: {curiosity_stats['update_count']}")
        
        # 定期自動備份到Google Drive
        checkpoint_name = f"checkpoint_{t//1000}k"
        checkpoint_data = {
            'model_state_dict': ddpg_agent.state_dict(),
            'episode': episode_count + start_episode,
            'timestep': t,
            'best_reward': best_reward
        }
        checkpoint_meta = {
            'episode': episode_count + start_episode,
            'timestep': t,
            'best_reward': best_reward,
            'checkpoint': True
        }
        
        if gdrive_sync.save_model(checkpoint_data, checkpoint_name, checkpoint_meta):
            print(f"📤 定期備份已保存到 Google Drive")
        
        print("=" * 50)

# =================================================================
# 5. 訓練完成和總結
# =================================================================
final_model_path = f"final_{MODEL_NAME}.pth"

# 保存最終模型 (包含完整狀態)
final_checkpoint = {
    'model_state_dict': ddpg_agent.state_dict(),
    'episode': episode_count + start_episode,
    'timestep': TOTAL_TIMESTEPS,
    'best_reward': best_reward,
    'final_training': True
}
torch.save(final_checkpoint, final_model_path)

# 自動保存最終模型到Google Drive
final_metadata = {
    'episode': episode_count + start_episode,
    'timestep': TOTAL_TIMESTEPS, 
    'best_reward': best_reward,
    'training_completed': True
}
gdrive_sync.save_model(final_checkpoint, f"final_{MODEL_NAME}", final_metadata)

curiosity_final_stats = curiosity_explorer.get_statistics()

print(f"\n🎉 純好奇心訓練完成！")
print(f"🏆 最佳回合獎勵: {best_reward:.2f}")
print(f"🧠 總好奇心獎勵: {curiosity_final_stats['total_intrinsic_reward']:.2f}")
print(f"📊 平均好奇心獎勵: {curiosity_final_stats['average_intrinsic_reward']:.4f}")
print(f"🔄 總回合數: {episode_count}")
print(f"💾 模型文件: {best_model_path}, {final_model_path}")

# 清理
env.close()
logger.close()
print("🏁 純好奇心實驗完成！")