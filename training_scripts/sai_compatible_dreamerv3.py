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
            # 使用DreamerV3的編碼器和actor
            # 初始化狀態
            batch_size = x.shape[0]
            state = self.dreamer.rssm.init_state(batch_size)
            
            # 編碼觀察
            obs_embed = self.dreamer.rssm.encode_obs(x)
            
            # 獲取後驗狀態
            posterior_input = torch.cat([state['deter'], obs_embed], dim=-1)
            posterior_logits = self.dreamer.rssm.posterior_net(posterior_input)
            stoch = self.dreamer.rssm.get_stochastic_state(posterior_logits)
            
            # 創建完整狀態
            full_state = {
                'deter': state['deter'],
                'stoch': stoch
            }
            
            # 獲取動作
            action = self.dreamer.actor(full_state)
            
            # 確保輸出是CPU上的tensor
            action = action.cpu()
            
            # 如果輸入是單個樣本，壓縮批次維度
            if squeeze_output:
                action = action.squeeze(0)
                
        return action