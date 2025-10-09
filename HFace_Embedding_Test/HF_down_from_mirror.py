# download_from_HF_mirror.py

import os
from huggingface_hub import snapshot_download
from sentence_transformers import SentenceTransformer

# ✅ 必须提前设置环境变量（在 import 之前最好）
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# 设置缓存目录
os.makedirs("./models", exist_ok=True)

# 要下载的模型名称
model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
local_model_path = f"./models/{model_name.replace('/', '_')}"

print(f"正在从镜像站下载模型到: {local_model_path}")
print(f"镜像地址: https://hf-mirror.com/{model_name}")

# 使用 snapshot_download 主动从镜像下载
try:
    snapshot_download(
        repo_id=model_name,
        local_dir=local_model_path,
        local_dir_use_symlinks=False,  # 直接复制文件，避免软链接问题
        endpoint="https://hf-mirror.com",  # 显式指定镜像站点
        max_workers=3
    )
    print("✅ 模型下载完成！")
except Exception as e:
    print("❌ 下载失败:", str(e))
    raise
