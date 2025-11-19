# -*- coding: utf-8 -*-
# logger.py
import os
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

class TensorBoardLogger:
    """
    TensorBoard 紀錄器類，用於在訓練期間追蹤損失和獎勵。
    """
    def __init__(self, log_dir_name="runs", model_name="DDPG_Agent"):
        """
        初始化 SummaryWriter。
        log_dir_name: 儲存 logs 的主資料夾名稱 (通常是 'runs')。
        model_name: 模型的名稱，將用於創建子資料夾。
        """
        # 創建一個帶有時間戳的子資料夾，以保持每次運行的獨立性
        current_time = datetime.now().strftime('%Y%m%d-%H%M%S')
        log_dir = os.path.join(log_dir_name, f"{model_name}_{current_time}")
        
        self.writer = SummaryWriter(log_dir)
        print(f"TensorBoard 紀錄器已初始化，Log 目錄: {log_dir}")
        self.step = 0

    def log_scalar(self, tag, value, step=None):
        """
        紀錄單一純量值。
        tag: 標籤名稱 (例如 'Loss/Critic_Loss')
        value: 要紀錄的值
        step: 紀錄的全局步數，如果為 None 則使用內部步數計數器
        """
        if step is None:
            step = self.step
        self.writer.add_scalar(tag, value, step)

    def set_step(self, step):
        """
        設置當前的全局步數。
        """
        self.step = step
    
    def close(self):
        """
        關閉 Writer，確保所有資料寫入磁碟。
        """
        self.writer.close()