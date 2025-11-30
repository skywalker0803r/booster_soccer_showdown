"""
SAC + RND 訓練配置文件
統一管理所有超參數，避免硬編碼
"""
from dataclasses import dataclass
from typing import Optional

@dataclass
class SACConfig:
    """SAC 演算法配置"""
    gamma: float = 0.99           # 折扣因子
    tau: float = 0.005            # 軟更新係數
    alpha: float = 0.2            # 溫度參數（熵調節）
    learning_rate: float = 3e-4   # 學習率
    batch_size: int = 256         # 批次大小
    buffer_size: int = 1_000_000  # 經驗回放緩衝區大小
    
    # 網絡架構
    actor_hidden_dim: int = 256   # Actor 網絡隱藏層維度
    critic_hidden_dim: int = 256  # Critic 網絡隱藏層維度
    
    def to_dict(self):
        """轉換為字典格式（用於記錄）"""
        return {
            'sac_gamma': self.gamma,
            'sac_tau': self.tau,
            'sac_alpha': self.alpha,
            'sac_learning_rate': self.learning_rate,
            'sac_batch_size': self.batch_size,
            'sac_buffer_size': self.buffer_size,
            'sac_actor_hidden_dim': self.actor_hidden_dim,
            'sac_critic_hidden_dim': self.critic_hidden_dim
        }

@dataclass
class RNDConfig:
    """RND 探索配置"""
    use_rnd: bool = True                    # 是否啟用 RND
    intrinsic_reward_scale: float = 0.1     # 內在獎勵縮放因子
    hidden_dim: int = 256                   # RND 網絡隱藏層維度
    learning_rate: float = 1e-4             # RND 學習率
    update_frequency: int = 10              # RND 更新頻率（每 N 步）
    
    def to_dict(self):
        """轉換為字典格式（用於記錄）"""
        return {
            'rnd_use_rnd': self.use_rnd,
            'rnd_intrinsic_reward_scale': self.intrinsic_reward_scale,
            'rnd_hidden_dim': self.hidden_dim,
            'rnd_learning_rate': self.learning_rate,
            'rnd_update_frequency': self.update_frequency
        }

@dataclass
class TrainingConfig:
    """訓練過程配置"""
    total_episodes: int = 1000              # 總訓練回合數
    save_interval: int = 100                # 檢查點保存間隔
    log_interval: int = 10                  # 日誌輸出間隔
    tensorboard_log_dir: str = "tensorboard_logs"  # TensorBoard 日誌目錄
    
    # TensorBoard 記錄頻率
    tb_action_log_freq: int = 50           # 動作分佈記錄頻率
    tb_network_log_freq: int = 100         # 網絡統計記錄頻率
    
    def to_dict(self):
        """轉換為字典格式（用於記錄）"""
        return {
            'training_total_episodes': self.total_episodes,
            'training_save_interval': self.save_interval,
            'training_log_interval': self.log_interval,
            'training_tensorboard_log_dir': self.tensorboard_log_dir,
            'training_tb_action_log_freq': self.tb_action_log_freq,
            'training_tb_network_log_freq': self.tb_network_log_freq
        }

@dataclass
class EnvironmentConfig:
    """環境配置"""
    env_name: str = "booster-soccer-showdown"  # 環境名稱
    api_key: str = "sai_LFcuaCZiqEkUbNVolQ3wbk5yU7H11jfv"  # API 密鑰
    
    def to_dict(self):
        """轉換為字典格式（用於記錄）"""
        return {
            'env_name': self.env_name,
            'env_api_key_length': len(self.api_key) if self.api_key else 0  # 不記錄完整密鑰
        }

class ExperimentConfig:
    """完整實驗配置"""
    
    def __init__(self, 
                 sac_config: Optional[SACConfig] = None,
                 rnd_config: Optional[RNDConfig] = None,
                 training_config: Optional[TrainingConfig] = None,
                 environment_config: Optional[EnvironmentConfig] = None,
                 experiment_name: str = "SAC_RND_Soccer",
                 device: str = "auto"):
        
        self.sac = sac_config or SACConfig()
        self.rnd = rnd_config or RNDConfig()
        self.training = training_config or TrainingConfig()
        self.environment = environment_config or EnvironmentConfig()
        self.experiment_name = experiment_name
        self.device = device
        
        # 自動推斷設備
        if self.device == "auto":
            import torch
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    def get_all_hyperparameters(self):
        """獲取所有超參數的統一字典"""
        all_params = {}
        all_params.update(self.sac.to_dict())
        all_params.update(self.rnd.to_dict())
        all_params.update(self.training.to_dict())
        all_params.update(self.environment.to_dict())
        all_params.update({
            'experiment_name': self.experiment_name,
            'device': self.device
        })
        return all_params
    
    def print_config(self):
        """打印配置摘要"""
        print(f"=== 實驗配置: {self.experiment_name} ===")
        print(f"設備: {self.device}")
        print(f"SAC 學習率: {self.sac.learning_rate}")
        print(f"SAC 批次大小: {self.sac.batch_size}")
        print(f"RND 啟用: {self.rnd.use_rnd}")
        print(f"RND 獎勵縮放: {self.rnd.intrinsic_reward_scale}")
        print(f"總訓練回合: {self.training.total_episodes}")
        print("=" * 50)

# 預設配置實例
DEFAULT_CONFIG = ExperimentConfig()

# 不同的實驗配置範例
def get_high_exploration_config():
    """高探索性配置"""
    return ExperimentConfig(
        rnd_config=RNDConfig(intrinsic_reward_scale=0.2),
        experiment_name="SAC_RND_HighExploration"
    )

def get_fast_training_config():
    """快速訓練配置"""
    return ExperimentConfig(
        sac_config=SACConfig(batch_size=128, learning_rate=5e-4),
        training_config=TrainingConfig(total_episodes=500),
        experiment_name="SAC_RND_FastTraining"
    )

def get_no_rnd_config():
    """純 SAC 配置（不使用 RND）"""
    return ExperimentConfig(
        rnd_config=RNDConfig(use_rnd=False),
        experiment_name="SAC_Only"
    )

# 使用範例
if __name__ == "__main__":
    # 測試配置系統
    config = DEFAULT_CONFIG
    config.print_config()
    
    # 測試高探索配置
    print("\n")
    high_exp_config = get_high_exploration_config()
    high_exp_config.print_config()
    
    # 打印所有超參數
    print("\n所有超參數:")
    for key, value in config.get_all_hyperparameters().items():
        print(f"  {key}: {value}")