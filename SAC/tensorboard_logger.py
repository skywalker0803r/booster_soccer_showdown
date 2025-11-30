import os
import time
from datetime import datetime
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter

class TensorBoardLogger:
    """TensorBoard 記錄器，用於訓練過程可視化"""
    
    def __init__(self, log_dir="runs", experiment_name=None, comment=""):
        """
        初始化 TensorBoard Logger
        
        Args:
            log_dir: 日誌保存目錄
            experiment_name: 實驗名稱
            comment: 額外註釋
        """
        if experiment_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            experiment_name = f"SAC_RND_{timestamp}"
        
        if comment:
            experiment_name += f"_{comment}"
        
        self.log_dir = os.path.join(log_dir, experiment_name)
        self.writer = SummaryWriter(log_dir=self.log_dir)
        
        # 記錄開始時間
        self.start_time = time.time()
        self.episode_count = 0
        self.step_count = 0
        
        print(f"TensorBoard Logger 初始化完成")
        print(f"日誌目錄: {self.log_dir}")
        print(f"啟動命令: tensorboard --logdir={log_dir}")
    
    def log_episode_metrics(self, episode, episode_reward, episode_steps, episode_time):
        """記錄回合級別的指標"""
        self.episode_count = episode
        
        # 基本回合指標
        self.writer.add_scalar('Episode/Reward', episode_reward, episode)
        self.writer.add_scalar('Episode/Steps', episode_steps, episode)
        self.writer.add_scalar('Episode/Time', episode_time, episode)
        self.writer.add_scalar('Episode/Steps_per_Second', episode_steps / episode_time, episode)
    
    def log_training_metrics(self, step, **kwargs):
        """記錄訓練級別的指標"""
        self.step_count = step
        
        for key, value in kwargs.items():
            if value is not None:
                self.writer.add_scalar(f'Training/{key}', value, step)
    
    def log_rnd_metrics(self, step, **kwargs):
        """記錄 RND 相關指標"""
        for key, value in kwargs.items():
            if value is not None:
                self.writer.add_scalar(f'RND/{key}', value, step)
    
    def log_buffer_metrics(self, step, buffer_size, buffer_capacity=None):
        """記錄 Buffer 相關指標"""
        self.writer.add_scalar('Buffer/Size', buffer_size, step)
        if buffer_capacity:
            usage_rate = buffer_size / buffer_capacity * 100
            self.writer.add_scalar('Buffer/Usage_Rate', usage_rate, step)
    
    def log_network_metrics(self, step, network_name, network):
        """記錄神經網絡參數統計"""
        for name, param in network.named_parameters():
            if param.grad is not None:
                # 參數統計
                self.writer.add_histogram(f'{network_name}/{name}', param.data.detach().cpu(), step)
                self.writer.add_scalar(f'{network_name}/{name}_mean', param.data.detach().mean().item(), step)
                self.writer.add_scalar(f'{network_name}/{name}_std', param.data.detach().std().item(), step)
                
                # 梯度統計
                self.writer.add_histogram(f'{network_name}/{name}_grad', param.grad.data.detach().cpu(), step)
                self.writer.add_scalar(f'{network_name}/{name}_grad_norm', param.grad.data.detach().norm().item(), step)
    
    def log_action_distribution(self, step, actions):
        """記錄動作分佈"""
        if isinstance(actions, torch.Tensor):
            actions = actions.detach().cpu().numpy()
        
        # 記錄每個動作維度的統計
        for i in range(actions.shape[-1]):
            self.writer.add_histogram(f'Actions/Action_{i}', actions[:, i], step)
            self.writer.add_scalar(f'Actions/Action_{i}_mean', np.mean(actions[:, i]), step)
            self.writer.add_scalar(f'Actions/Action_{i}_std', np.std(actions[:, i]), step)
    
    def log_reward_breakdown(self, step, external_reward, intrinsic_reward=None, total_reward=None):
        """記錄獎勵分解"""
        self.writer.add_scalar('Rewards/External', external_reward, step)
        
        if intrinsic_reward is not None:
            self.writer.add_scalar('Rewards/Intrinsic', intrinsic_reward, step)
        
        if total_reward is not None:
            self.writer.add_scalar('Rewards/Total', total_reward, step)
        
        if intrinsic_reward is not None and external_reward is not None:
            ratio = intrinsic_reward / (abs(external_reward) + 1e-8)
            self.writer.add_scalar('Rewards/Intrinsic_External_Ratio', ratio, step)
    
    def log_moving_averages(self, episode, rewards_history, steps_history, window_sizes=[10, 50, 100]):
        """記錄滑動平均"""
        for window in window_sizes:
            if len(rewards_history) >= window:
                avg_reward = np.mean(rewards_history[-window:])
                avg_steps = np.mean(steps_history[-window:])
                
                self.writer.add_scalar(f'MovingAvg/Reward_{window}', avg_reward, episode)
                self.writer.add_scalar(f'MovingAvg/Steps_{window}', avg_steps, episode)
    
    def log_hyperparameters(self, hparams_dict):
        """記錄超參數"""
        # 將所有數值轉換為 scalar
        processed_hparams = {}
        for key, value in hparams_dict.items():
            if isinstance(value, (int, float, bool)):
                processed_hparams[key] = value
            else:
                processed_hparams[key] = str(value)
        
        self.writer.add_hparams(processed_hparams, {})
    
    def log_text(self, tag, text, step=None):
        """記錄文本信息"""
        if step is None:
            step = self.step_count
        self.writer.add_text(tag, text, step)
    
    def log_model_graph(self, model, input_tensor):
        """記錄模型結構圖"""
        try:
            self.writer.add_graph(model, input_tensor)
        except Exception as e:
            print(f"無法記錄模型圖: {e}")
    
    def log_learning_curves(self, episode, metrics_dict):
        """記錄學習曲線"""
        for metric_name, value in metrics_dict.items():
            if value is not None:
                self.writer.add_scalar(f'LearningCurves/{metric_name}', value, episode)
    
    def log_exploration_metrics(self, step, state_visitation_count=None, unique_states_ratio=None):
        """記錄探索相關指標"""
        if state_visitation_count is not None:
            self.writer.add_scalar('Exploration/State_Visitation_Count', state_visitation_count, step)
        
        if unique_states_ratio is not None:
            self.writer.add_scalar('Exploration/Unique_States_Ratio', unique_states_ratio, step)
    
    def save_checkpoint_info(self, episode, checkpoint_path):
        """記錄檢查點信息"""
        self.log_text('Checkpoints', f'Episode {episode}: {checkpoint_path}', episode)
    
    def close(self):
        """關閉 TensorBoard Logger"""
        total_time = time.time() - self.start_time
        
        # 記錄總結信息
        summary_text = f"""
        訓練總結:
        - 總回合數: {self.episode_count}
        - 總步數: {self.step_count}
        - 總訓練時間: {total_time/3600:.2f} 小時
        - 平均每回合時間: {total_time/max(1, self.episode_count):.2f} 秒
        """
        
        self.log_text('Training_Summary', summary_text)
        self.writer.close()
        
        print(f"TensorBoard Logger 已關閉")
        print(f"總訓練時間: {total_time/3600:.2f} 小時")
        print(f"查看結果: tensorboard --logdir={os.path.dirname(self.log_dir)}")


class SAC_RND_TensorBoardLogger(TensorBoardLogger):
    """專門為 SAC + RND 設計的 TensorBoard Logger"""
    
    def __init__(self, log_dir="runs", experiment_name=None, comment=""):
        super().__init__(log_dir, experiment_name, comment)
        
        # SAC + RND 特定的統計
        self.actor_losses = []
        self.critic_losses = []
        self.rnd_losses = []
        self.intrinsic_rewards = []
    
    def log_sac_update(self, step, actor_loss=None, critic1_loss=None, critic2_loss=None, 
                       q1_value=None, q2_value=None, log_prob=None, alpha=None):
        """記錄 SAC 更新指標"""
        metrics = {
            'Actor_Loss': actor_loss,
            'Critic1_Loss': critic1_loss,
            'Critic2_Loss': critic2_loss,
            'Q1_Value_Mean': q1_value.detach().mean().item() if q1_value is not None else None,
            'Q2_Value_Mean': q2_value.detach().mean().item() if q2_value is not None else None,
            'Log_Prob_Mean': log_prob.detach().mean().item() if log_prob is not None else None,
            'Alpha': alpha
        }
        
        self.log_training_metrics(step, **metrics)
        
        # 記錄損失歷史
        if actor_loss is not None:
            self.actor_losses.append(actor_loss)
        if critic1_loss is not None:
            self.critic_losses.append(critic1_loss)
    
    def log_rnd_update(self, step, rnd_loss=None, mean_intrinsic_reward=None, 
                       std_intrinsic_reward=None, obs_count=None, rnd_buffer_size=None):
        """記錄 RND 更新指標"""
        metrics = {
            'Loss': rnd_loss,
            'Mean_Intrinsic_Reward': mean_intrinsic_reward,
            'Std_Intrinsic_Reward': std_intrinsic_reward,
            'Observation_Count': obs_count,
            'Buffer_Size': rnd_buffer_size
        }
        
        self.log_rnd_metrics(step, **metrics)
        
        # 記錄 RND 損失歷史
        if rnd_loss is not None:
            self.rnd_losses.append(rnd_loss)
        if mean_intrinsic_reward is not None:
            self.intrinsic_rewards.append(mean_intrinsic_reward)
    
    def log_episode_summary(self, episode, episode_reward, episode_steps, episode_time,
                           external_reward_sum=None, intrinsic_reward_sum=None):
        """記錄回合總結"""
        # 基本回合指標
        self.log_episode_metrics(episode, episode_reward, episode_steps, episode_time)
        
        # 獎勵分解
        if external_reward_sum is not None and intrinsic_reward_sum is not None:
            self.log_reward_breakdown(episode, external_reward_sum, intrinsic_reward_sum, episode_reward)


# 使用範例
if __name__ == "__main__":
    # 基本使用
    logger = SAC_RND_TensorBoardLogger(comment="test_run")
    
    # 記錄超參數
    hparams = {
        'learning_rate': 3e-4,
        'batch_size': 256,
        'gamma': 0.99,
        'tau': 0.005,
        'rnd_scale': 0.1
    }
    logger.log_hyperparameters(hparams)
    
    # 模擬記錄一些數據
    for episode in range(100):
        episode_reward = np.random.normal(10, 5)
        episode_steps = np.random.randint(50, 200)
        episode_time = episode_steps * 0.02
        
        logger.log_episode_summary(episode, episode_reward, episode_steps, episode_time)
        
        if episode % 10 == 0:
            logger.log_sac_update(
                step=episode,
                actor_loss=np.random.uniform(0.1, 1.0),
                critic1_loss=np.random.uniform(0.1, 1.0),
                critic2_loss=np.random.uniform(0.1, 1.0)
            )
            
            logger.log_rnd_update(
                step=episode,
                rnd_loss=np.random.uniform(0.01, 0.1),
                mean_intrinsic_reward=np.random.uniform(0.001, 0.01),
                obs_count=episode * 100
            )
    
    logger.close()
    print("TensorBoard 測試完成！")