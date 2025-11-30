import os
import json
import time
import zipfile
import tempfile
from datetime import datetime
from pathlib import Path

try:
    from pydrive2.auth import GoogleAuth
    from pydrive2.drive import GoogleDrive
    GDRIVE_AVAILABLE = True
except ImportError:
    GDRIVE_AVAILABLE = False
    print("警告: pydrive2 未安裝。請運行: pip install pydrive2")

import torch
import numpy as np

class GoogleDriveAutoSaver:
    """Google Drive 自動保存模組"""
    
    def __init__(self, 
                 credentials_file="credentials.json",
                 save_folder_name="SAC_RND_Models",
                 save_interval=100,
                 keep_best_n=5,
                 auto_auth=True):
        """
        初始化 Google Drive 自動保存模組
        
        Args:
            credentials_file: Google Drive API 憑證文件
            save_folder_name: Google Drive 中的保存資料夾名稱
            save_interval: 保存間隔（回合數）
            keep_best_n: 保留最佳模型數量
            auto_auth: 是否自動認證
        """
        self.credentials_file = credentials_file
        self.save_folder_name = save_folder_name
        self.save_interval = save_interval
        self.keep_best_n = keep_best_n
        self.auto_auth = auto_auth
        
        self.drive = None
        self.folder_id = None
        self.best_reward = float('-inf')
        self.saved_models = []  # 記錄已保存的模型信息
        self.last_save_episode = 0
        
        print(f"Google Drive 自動保存模組初始化")
        print(f"保存間隔: {save_interval} 回合")
        print(f"保留最佳模型數量: {keep_best_n}")
        
        if GDRIVE_AVAILABLE:
            if auto_auth:
                self._authenticate()
        else:
            print("❌ pydrive2 未安裝，Google Drive 功能不可用")
    
    def _authenticate(self):
        """認證 Google Drive"""
        try:
            # 檢查憑證文件
            if not os.path.exists(self.credentials_file):
                self._create_credentials_template()
                print(f"❌ 請配置 Google Drive 憑證文件: {self.credentials_file}")
                return False
            
            # 創建設置文件
            settings = {
                "client_config_backend": "file",
                "client_config_file": self.credentials_file,
                "save_credentials": True,
                "save_credentials_backend": "file",
                "save_credentials_file": "token.json",
                "oauth_scope": ["https://www.googleapis.com/auth/drive"]
            }
            
            # 保存設置到臨時文件
            with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
                import yaml
                yaml.dump(settings, f)
                settings_file = f.name
            
            try:
                gauth = GoogleAuth(settings_file)
                gauth.LocalWebserverAuth()  # 網頁認證
                self.drive = GoogleDrive(gauth)
                
                # 創建或找到保存資料夾
                self._setup_folder()
                
                print("✅ Google Drive 認證成功！")
                return True
                
            except Exception as e:
                print(f"❌ Google Drive 認證失敗: {e}")
                return False
            finally:
                # 清理臨時設置文件
                if os.path.exists(settings_file):
                    os.unlink(settings_file)
                    
        except ImportError:
            print("❌ 缺少依賴: pip install pyyaml")
            return False
        except Exception as e:
            print(f"❌ 認證過程出錯: {e}")
            return False
    
    def _create_credentials_template(self):
        """創建憑證文件模板"""
        template = {
            "installed": {
                "client_id": "YOUR_CLIENT_ID.googleusercontent.com",
                "project_id": "your-project-id",
                "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                "token_uri": "https://oauth2.googleapis.com/token",
                "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
                "client_secret": "YOUR_CLIENT_SECRET",
                "redirect_uris": ["http://localhost"]
            }
        }
        
        with open(self.credentials_file, 'w') as f:
            json.dump(template, f, indent=2)
        
        print(f"📝 已創建憑證模板: {self.credentials_file}")
        print("請到 Google Cloud Console 獲取真實的憑證信息")
        print("1. 前往: https://console.cloud.google.com/")
        print("2. 啟用 Google Drive API")
        print("3. 創建 OAuth 2.0 憑證")
        print("4. 下載 JSON 文件並替換模板內容")
    
    def _setup_folder(self):
        """設置或創建 Google Drive 資料夾"""
        if not self.drive:
            return
        
        try:
            # 查找現有資料夾
            file_list = self.drive.ListFile({
                'q': f"title='{self.save_folder_name}' and mimeType='application/vnd.google-apps.folder' and trashed=false"
            }).GetList()
            
            if file_list:
                self.folder_id = file_list[0]['id']
                print(f"✅ 找到現有資料夾: {self.save_folder_name}")
            else:
                # 創建新資料夾
                folder_metadata = {
                    'title': self.save_folder_name,
                    'mimeType': 'application/vnd.google-apps.folder'
                }
                folder = self.drive.CreateFile(folder_metadata)
                folder.Upload()
                self.folder_id = folder['id']
                print(f"✅ 已創建新資料夾: {self.save_folder_name}")
                
        except Exception as e:
            print(f"❌ 設置資料夾失敗: {e}")
            self.folder_id = None
    
    def should_save(self, episode, reward):
        """判斷是否應該保存模型"""
        if not self.drive or not self.folder_id:
            return False
        
        # 檢查保存間隔
        if episode - self.last_save_episode < self.save_interval:
            return False
        
        # 檢查是否是最佳模型
        if reward > self.best_reward:
            self.best_reward = reward
            return True
        
        # 定期保存（即使不是最佳）
        if episode % (self.save_interval * 2) == 0:
            return True
            
        return False
    
    def save_model(self, agent, episode, reward, metrics=None):
        """保存模型到 Google Drive"""
        if not self.should_save(episode, reward):
            return False
        
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_name = f"SAC_RND_ep{episode}_reward{reward:.2f}_{timestamp}"
            
            # 創建本地臨時目錄
            with tempfile.TemporaryDirectory() as temp_dir:
                model_dir = os.path.join(temp_dir, model_name)
                os.makedirs(model_dir)
                
                # 保存模型狀態
                model_files = self._save_local_model(agent, model_dir, episode, reward, metrics)
                
                # 創建壓縮文件
                zip_path = os.path.join(temp_dir, f"{model_name}.zip")
                self._create_zip(model_dir, zip_path)
                
                # 上傳到 Google Drive
                success = self._upload_to_drive(zip_path, f"{model_name}.zip")
                
                if success:
                    # 記錄保存的模型
                    model_info = {
                        'name': model_name,
                        'episode': episode,
                        'reward': reward,
                        'timestamp': timestamp,
                        'is_best': reward >= self.best_reward
                    }
                    self.saved_models.append(model_info)
                    self.last_save_episode = episode
                    
                    # 清理舊模型
                    self._cleanup_old_models()
                    
                    print(f"✅ 模型已保存到 Google Drive: {model_name}")
                    return True
                else:
                    print(f"❌ 模型上傳失敗: {model_name}")
                    return False
                    
        except Exception as e:
            print(f"❌ 保存模型時出錯: {e}")
            return False
    
    def _save_local_model(self, agent, model_dir, episode, reward, metrics):
        """保存模型到本地目錄"""
        files_created = []
        
        try:
            # 保存 SAC 網絡
            sac_state = {
                'actor_state_dict': agent.actor.state_dict(),
                'critic1_state_dict': agent.critic1.state_dict(),
                'critic2_state_dict': agent.critic2.state_dict(),
                'critic1_target_state_dict': agent.critic1_target.state_dict(),
                'critic2_target_state_dict': agent.critic2_target.state_dict(),
                'actor_optimizer_state_dict': agent.actor_opt.state_dict(),
                'critic1_optimizer_state_dict': agent.critic1_opt.state_dict(),
                'critic2_optimizer_state_dict': agent.critic2_opt.state_dict(),
                'episode': episode,
                'reward': reward,
                'obs_dim': agent.obs_dim,
                'act_dim': agent.act_dim
            }
            
            sac_path = os.path.join(model_dir, 'sac_model.pth')
            torch.save(sac_state, sac_path)
            files_created.append('sac_model.pth')
            
            # 保存 RND 模型（如果啟用）
            if agent.use_rnd and agent.rnd:
                rnd_path = os.path.join(model_dir, 'rnd_model.pth')
                agent.rnd.save(rnd_path)
                files_created.append('rnd_model.pth')
            
            # 保存訓練指標
            if metrics:
                metrics_path = os.path.join(model_dir, 'metrics.json')
                with open(metrics_path, 'w') as f:
                    json.dump(metrics, f, indent=2)
                files_created.append('metrics.json')
            
            # 保存模型信息
            model_info = {
                'episode': episode,
                'reward': reward,
                'timestamp': datetime.now().isoformat(),
                'obs_dim': agent.obs_dim,
                'act_dim': agent.act_dim,
                'use_rnd': agent.use_rnd,
                'files': files_created
            }
            
            info_path = os.path.join(model_dir, 'model_info.json')
            with open(info_path, 'w') as f:
                json.dump(model_info, f, indent=2)
            files_created.append('model_info.json')
            
            return files_created
            
        except Exception as e:
            print(f"❌ 保存本地模型失敗: {e}")
            return []
    
    def _create_zip(self, model_dir, zip_path):
        """創建模型壓縮文件"""
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(model_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, model_dir)
                    zipf.write(file_path, arcname)
    
    def _upload_to_drive(self, local_path, filename):
        """上傳文件到 Google Drive"""
        try:
            file_metadata = {
                'title': filename,
                'parents': [{'id': self.folder_id}]
            }
            
            file_obj = self.drive.CreateFile(file_metadata)
            file_obj.SetContentFile(local_path)
            file_obj.Upload()
            
            return True
            
        except Exception as e:
            print(f"❌ 上傳文件失敗: {e}")
            return False
    
    def _cleanup_old_models(self):
        """清理舊的模型文件"""
        if len(self.saved_models) <= self.keep_best_n:
            return
        
        try:
            # 按獎勵排序，保留最佳的 N 個模型
            self.saved_models.sort(key=lambda x: x['reward'], reverse=True)
            models_to_delete = self.saved_models[self.keep_best_n:]
            
            for model_info in models_to_delete:
                self._delete_from_drive(model_info['name'])
            
            # 更新保存的模型列表
            self.saved_models = self.saved_models[:self.keep_best_n]
            
            print(f"🧹 已清理舊模型，保留最佳 {self.keep_best_n} 個")
            
        except Exception as e:
            print(f"❌ 清理舊模型失敗: {e}")
    
    def _delete_from_drive(self, model_name):
        """從 Google Drive 刪除模型文件"""
        try:
            file_list = self.drive.ListFile({
                'q': f"title='{model_name}.zip' and '{self.folder_id}' in parents and trashed=false"
            }).GetList()
            
            for file_obj in file_list:
                file_obj.Delete()
                print(f"🗑️  已刪除舊模型: {model_name}")
                
        except Exception as e:
            print(f"❌ 刪除文件失敗: {e}")
    
    def get_saved_models_info(self):
        """獲取已保存模型的信息"""
        return sorted(self.saved_models, key=lambda x: x['reward'], reverse=True)
    
    def manual_save(self, agent, episode, reward, reason="manual"):
        """手動保存模型"""
        old_interval = self.save_interval
        self.save_interval = 0  # 臨時設置為0以強制保存
        
        result = self.save_model(agent, episode, reward, {'save_reason': reason})
        
        self.save_interval = old_interval  # 恢復原始設置
        return result
    
    def get_statistics(self):
        """獲取保存統計信息"""
        if not self.saved_models:
            return {
                'total_saved': 0,
                'best_reward': self.best_reward,
                'last_save_episode': self.last_save_episode
            }
        
        return {
            'total_saved': len(self.saved_models),
            'best_reward': max(model['reward'] for model in self.saved_models),
            'last_save_episode': self.last_save_episode,
            'average_reward': np.mean([model['reward'] for model in self.saved_models]),
            'latest_save': max(self.saved_models, key=lambda x: x['episode'])
        }


# 使用範例和測試
if __name__ == "__main__":
    # 測試 Google Drive 保存器
    saver = GoogleDriveAutoSaver(
        save_interval=10,
        keep_best_n=3,
        auto_auth=False  # 手動測試時設為 False
    )
    
    print("Google Drive 自動保存模組測試完成")
    print(f"保存間隔: {saver.save_interval} 回合")
    print(f"保留最佳模型數量: {saver.keep_best_n}")
    
    if GDRIVE_AVAILABLE:
        print("✅ pydrive2 已安裝，可以使用 Google Drive 功能")
    else:
        print("❌ pydrive2 未安裝，請運行: pip install pydrive2")