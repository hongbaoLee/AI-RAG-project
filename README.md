by Kevin Lee
my AI project, to improve Enterprise's level in AI. It will use AI technologies including RAG, Embedding, Function Call, etc.

HFace_Embedding_test目录，测试在线和本地使用Huggingface进行Embedding的文件
models目录，下载到本地的Huggingface本地模型文件
chroma_db_company目录，chromaDB向量数据库
Enterprise_rules_Alli.py 使用阿里云内置的Embedding算法+pyPDF2+ChromaDB+Qwen3-max大模型，实现PDF问答内容的解析和智能问答
Enterprise_rules_HFace.py 使用下载到本地的HuggingFace模型进行Embedding+pyPDF2+ChromaDB+Qwen3-max大模型，实现PDF问答内容的解析和智能问答
  由于阿里云内置Embedding的向量维度为1536，而Huggingface的向量维度为384，所以二者并不兼容。ChromaDB不支持动态调整维度，向量数据库一旦创建，维度就固定了。所以必须删除旧数据库，用新的embedding模型重建。
assistan.py 基于streamlit开发的支持opensssl的用户问答客户端，文件中Line 2为关联的文件名称和要引用的函数。