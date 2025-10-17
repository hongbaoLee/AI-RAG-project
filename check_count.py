# check_count.py
# This script checks the total number of documents in the "langchain" collection
# of a persistent ChromaDB database and prints the first 5 documents.
from chromadb import PersistentClient

client = PersistentClient(path="chroma_db_corp")
collection = client.get_collection("langchain")

print("📊 文档总数:", collection.count())

# 查看前5个文档内容
# results = collection.peek(5)
# for i, doc in enumerate(results["documents"]):
#     print(f"\n[{i+1}] {doc[:200]}...")