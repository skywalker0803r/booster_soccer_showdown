"""
SAI-Compatible DreamerV3 Model
根據官方文檔創建完全兼容的PyTorch模型
"""
import torch
import torch.nn as nn
import numpy as np

class SAICompatibleDreamerV3(nn.Module):
    """完全SAI兼容的DreamerV3包裝器"""
    
    def __init__(self, dreamer_model):
        super().__init__()
        # 將DreamerV3模型的所有參數移到CPU
        self.dreamer = dreamer_model.cpu()
        # 確保模型在評估模式
        self.dreamer.eval()
        
    def forward(self, x):
        """標準PyTorch forward方法，完全CPU兼容"""
        # 確保輸入是CPU上的float32 tensor
        if isinstance(x, np.ndarray):
            x = torch.FloatTensor(x)
        elif isinstance(x, torch.Tensor):
            x = x.float().cpu()  # 強制轉為CPU和float32
        
        # 處理批次維度
        if x.dim() == 1:
            x = x.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
        
        # 確保正確的觀察維度
        if x.shape[-1] != self.dreamer.obs_dim:
            # 如果維度不匹配，假設需要調整
            x = x.view(-1, self.dreamer.obs_dim)
        
        with torch.no_grad():
            # 簡化的前向傳播，直接使用編碼器和actor
            batch_size = x.shape[0]
            
            # 編碼觀察
            encoded = self.dreamer.encoder(x)
            
            # 創建虛擬的隨機狀態用於actor
            dummy_stoch = torch.zeros(batch_size, self.dreamer.stoch_dim)
            
            # 組合狀態
            actor_input = torch.cat([dummy_stoch, encoded], dim=-1)
            
            # 獲取動作
            action = self.dreamer.actor(actor_input)
            
            # 確保輸出是CPU上的tensor
            action = action.cpu()
            
            # 如果輸入是單個樣本，壓縮批次維度
            if squeeze_output:
                action = action.squeeze(0)
                
        return action