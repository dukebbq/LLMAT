from huggingface_hub import snapshot_download
import os

# ✅ 你想保存到的本地目录（可自定义）
local_dir = "./all-MiniLM-L6-v2"

# 创建目录（如果不存在）
os.makedirs(local_dir, exist_ok=True)

# 下载模型（snapshot_download 会自动缓存，重复运行不会重复下载）
snapshot_download(
    repo_id="sentence-transformers/all-MiniLM-L6-v2",
    local_dir=local_dir,
    local_dir_use_symlinks=False  # Windows 下必须禁用 symlink
)

print(f"✅ 模型已下载到: {local_dir}")
