# 下載代碼庫
#git clone https://github.com/skywalker0803r/booster_soccer_showdown.git
#cd booster_soccer_showdown

# 安裝 conda（使用 Mambaforge：輕量且快速）
# -----------------------------------------------
# 下載 mambaforge (包含 conda)
wget https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-Linux-x86_64.sh
# 執行安裝腳本，使用 -b 批次模式，-p 指定安裝路徑到當前用戶的 /usr/local/
bash Mambaforge-Linux-x86_64.sh -b -p /usr/local
# 初始化 conda 環境
/usr/local/bin/conda init bash
# 重新載入 shell 設定，讓 conda 指令生效
source ~/.bashrc
# -----------------------------------------------

# 建立虛擬環境
conda create -n booster-ssl python=3.11 -y && conda activate booster-ssl

# 安裝依賴
pip install -r requirements.txt

# 訓練
python Research/main.py