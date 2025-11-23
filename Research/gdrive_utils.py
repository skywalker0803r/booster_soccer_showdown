# -*- coding: utf-8 -*-
# gdrive_utils.py
# 最小化Google Drive自動保存功能

import os
import shutil
import pickle
from datetime import datetime

class SimpleGDriveSync:
    """簡單的Google Drive同步工具 (基於本地掛載路徑)"""
    
    def __init__(self, gdrive_path=None):
        # Google Drive常見掛載路徑
        possible_paths = [
            "G:\\我的雲端硬碟",  # Windows
            "/content/drive/MyDrive",  # Colab
            os.path.expanduser("~/Google Drive"),  # Mac/Linux
            gdrive_path  # 自定義路徑
        ]
        
        self.gdrive_path = None
        for path in possible_paths:
            if path and os.path.exists(path):
                self.gdrive_path = path
                break
                
        if self.gdrive_path:
            self.models_dir = os.path.join(self.gdrive_path, "DDPG_Models")
            os.makedirs(self.models_dir, exist_ok=True)
            print(f"✅ Google Drive 已連接: {self.gdrive_path}")
        else:
            print("⚠️ Google Drive 未找到，將只保存到本地")
    
    def save_model(self, model_state, model_name, metadata=None, add_timestamp=True): # <--- Modified signature
        """保存模型到Google Drive"""
        if not self.gdrive_path:
            return False
            
        try:
            if add_timestamp: # <--- New logic
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"{model_name}_{timestamp}.pth"
            else:
                filename = f"{model_name}.pth" # <--- Fixed name
            
            # 本地臨時保存
            local_path = filename
            if isinstance(model_state, dict):
                import torch
                torch.save(model_state, local_path)
            else:
                # 如果已經是保存好的文件，直接複製
                if os.path.exists(str(model_state)):
                    shutil.copy2(str(model_state), local_path)
                else:
                    import torch
                    torch.save(model_state, local_path)
            
            # Google Drive保存
            gdrive_path = os.path.join(self.models_dir, filename)
            shutil.copy2(local_path, gdrive_path)
            
            # 清理臨時文件
            if os.path.exists(local_path) and local_path != str(model_state):
                os.remove(local_path)
            
            # 保存元數據
            if metadata:
                meta_path = os.path.join(self.models_dir, f"{model_name}_latest_metadata.pkl")
                with open(meta_path, 'wb') as f:
                    pickle.dump(metadata, f)
            
            print(f"💾 模型已保存到 Google Drive: {filename}")
            return True
            
        except Exception as e:
            print(f"❌ Google Drive 保存失敗: {e}")
            return False
    
    def list_saved_models(self, model_prefix=""):
        """列出已保存的模型"""
        if not self.gdrive_path:
            return []
            
        try:
            models = []
            for file in os.listdir(self.models_dir):
                if file.startswith(model_prefix) and file.endswith('.pth'):
                    file_path = os.path.join(self.models_dir, file)
                    stat = os.stat(file_path)
                    models.append({
                        'name': file,
                        'path': file_path,
                        'size': stat.st_size,
                        'modified': datetime.fromtimestamp(stat.st_mtime)
                    })
            
            # 按修改時間排序 (最新的在前)
            models.sort(key=lambda x: x['modified'], reverse=True)
            return models
            
        except Exception as e:
            print(f"❌ 讀取模型列表失敗: {e}")
            return []