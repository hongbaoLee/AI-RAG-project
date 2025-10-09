from sentence_transformers import SentenceTransformer

model = SentenceTransformer("./models/st_paraphrase-multilingual-MiniLM-L12-v2")

sentences = ["你好，世界", "How are you?"]
embeddings = model.encode(sentences)

print(embeddings.shape)  # (2, 384)
