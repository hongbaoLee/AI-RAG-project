# list_collections.py
# This script lists all collections in a persistent ChromaDB database
# and prints their names along with the number of items in each collection.
from chromadb import PersistentClient

client = PersistentClient(path="chroma_db_corp")
collections = client.list_collections()

print("📊 当前数据库中的 collections:")
for col in collections:
    print(f" - {col.name} (count: {col.count()})")