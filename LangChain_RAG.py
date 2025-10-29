# LangChain_RAG.py
# 使用LangChain构建RAG（检索增强生成）系统的示例
# 支持从本地docx文件加载内容，进行向量化存储和检索，由于是Langchain学习文件，只处理简单的word文档，未进行扩展。
import os
# import docx                                                   #使用python-docx读取docx文件内容
# from langchain.schema import Document                         #使用python-docx读取docx文件内容
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import Docx2txtLoader 

EMBEDDING_MODEL_NAME = "./models/st_paraphrase-multilingual-MiniLM-L12-v2"   # 使用本地HuggingFace模型进行文本嵌入
LLM_MODEL = "qwen-max"    
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")

llm = ChatOpenAI(
    openai_api_key=DASHSCOPE_API_KEY,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    model=LLM_MODEL
)

# --- 1. 加载和处理文档 ---
#使用Docx2txtLoader加载docx文件
loader = Docx2txtLoader("./工作手册/第二部分同益保险公估车险工作手册.docx")
docs = loader.load()

#或者选择使用python-docx读取docx文件内容，包括段落和表格
# doc = docx.Document("./工作手册/第二部分同益保险公估车险工作手册.docx")
# text = "\n".join([para.text for para in doc.paragraphs if para.text.strip()])
# table_texts = []
# for table in doc.tables:
#     for row in table.rows:
#         row_text = " | ".join(cell.text.strip() for cell in row.cells)
#         table_texts.append(row_text)
# tables_str = "\n".join(table_texts)
# full_text = text + "\n" + tables_str
# docs = [Document(page_content=full_text, metadata={"source": "工作手册.docx"})]

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)

# --- 2. 创建向量存储和检索器 ---
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
retriever = vectorstore.as_retriever()

# --- 3. 构建 RAG Chain ---
template = """仅根据以下上下文来回答问题:
{context}

问题: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()} 
    | prompt
    | llm
    | StrOutputParser()
)

# --- 4. 用户输入问题并调用 RAG Chain ---
while True:
    question = input("请输入您的问题（输入 quit 或 exit 退出）：")
    if question.strip().lower() in ("quit", "exit"):
        print("程序已退出。")
        break
    response = rag_chain.invoke(question)
    print(response)
