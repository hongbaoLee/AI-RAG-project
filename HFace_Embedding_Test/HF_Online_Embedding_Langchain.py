import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from langchain_huggingface import HuggingFaceEmbeddings
import torch

# 检查 GPU 可用性
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 初始化嵌入模型
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",  # 轻量级模型，适合国内网络
    model_kwargs={'device': device}
)

# 使用示例
text = "This is a test document for embedding."
query_result = embeddings.embed_query(text)
print(f"Embedding dimension: {len(query_result)}")