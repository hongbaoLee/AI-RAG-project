by Kevin Lee
my AI project, to improve Enterprise's level in AI. It will use AI technologies including RAG, Embedding, Function Call, etc.

HFace_Embedding_test目录，测试在线和本地使用Huggingface进行Embedding的文件
models目录，下载到本地的Huggingface本地模型文件
chroma_db_company目录，chromaDB向量数据库
Enterprise_rules_Alli.py 使用阿里云内置的Embedding算法+pyPDF2+ChromaDB+Qwen3-max大模型，实现PDF问答内容的解析和智能问答
Enterprise_rules_HFace.py 使用下载到本地的HuggingFace模型进行Embedding+pyPDF2+ChromaDB+Qwen3-max大模型，实现PDF问答内容的解析和智能问答
  由于阿里云内置Embedding的向量维度为1536，而Huggingface的向量维度为384，所以二者并不兼容。ChromaDB不支持动态调整维度，向量数据库一旦创建，维度就固定了。所以必须删除旧数据库，用新的embedding模型重建。
assistan.py 基于streamlit开发的支持opensssl的用户问答客户端，文件中Line 2为关联的文件名称和要引用的函数。

从extract_text_from_PDF_with_tables.py的生成效果来看，chatGPT生成的代码质量比Qwen生成的代码质量好。

Enterprise_RAG_query_GPT.py ChatGPT辅助生成的企业RAG知识库查询系统，可以将word文档、excel表格、含图表的PDF文档内容向量化以后存入ChromaDB，通过Qwen-max模型提供知识库检索功能。员工可查询公司的规章制度。

主要可用文件：Enterprise_RAG_query_GPT.py

2025.10.29
增加了langchain开头的几个文件，用于langchain的学习，重点学习LangChain chains，Langchain Agent和Tools的构建
