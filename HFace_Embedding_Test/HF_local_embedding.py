from sentence_transformers import SentenceTransformer
import os

# 指定本地模型路径
model_path = "./models/st_paraphrase-multilingual-MiniLM-L12-v2"
model = SentenceTransformer(model_path)

# 测试 embedding
sentences = ["你好，世界！", "Hello, world!"]
embeddings = model.encode(sentences)

print("Embedding 向量示例：")
print(embeddings[0][:10])  # 打印前10维