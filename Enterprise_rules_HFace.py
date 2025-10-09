import os
from pathlib import Path
import json
from PyPDF2 import PdfReader
import chromadb
# 不再需要阿里云 embedding 函数
# from chromadb.utils import embedding_functions  # 不再使用
import dashscope
from dashscope import Generation  # 仍需用于 Qwen

# 新增：HuggingFace embedding 模型
from sentence_transformers import SentenceTransformer

# ==================== 配置区（请修改这里）====================
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")  # 仍用于 Qwen
CURRENT_DIR = Path(__file__).parent.absolute()
PDF_PATH = os.path.join(CURRENT_DIR, "exercise2024.pdf")
QWEN_MODEL = "qwen3-max"
# EMBEDDING_MODEL = "text-embedding-v1"  # ← 不再使用，已注释

# ⚠️ 请确保此路径指向你下载的 HuggingFace 模型文件夹
HF_MODEL_PATH = os.path.join(CURRENT_DIR, "./models/st_paraphrase-multilingual-MiniLM-L12-v2")
CHROMA_DB_PATH = "./chroma_db_company"
# ============================================================

# 设置 DashScope API Key（仅用于 Qwen）
dashscope.api_key = DASHSCOPE_API_KEY

# ================== Step 1: 提取 PDF 并分段 ==================
def extract_text_from_pdf(pdf_path):
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"找不到文件: {pdf_path}")
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

def split_text(text, chunk_size=300, overlap=50):
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start = end - overlap
    return [c.strip() for c in chunks if len(c.strip()) > 20]

# ================== Step 2: 使用本地 HuggingFace 模型获取 Embedding ==================
# 全局加载模型（只加载一次，提高效率）
print("🧠 正在加载本地 HuggingFace Embedding 模型...")
hf_embedding_model = SentenceTransformer(HF_MODEL_PATH)
print("✅ 模型加载完成！")

def get_embedding(texts):
    """
    使用本地 SentenceTransformer 模型生成 embedding
    输入: str 或 List[str]
    输出: 单个向量（list）或向量列表（List[list]）
    """
    was_single = False
    if isinstance(texts, str):
        texts = [texts]
        was_single = True

    # 生成 embedding，自动归一化（便于 cosine 相似度计算）
    embeddings = hf_embedding_model.encode(
        texts,
        convert_to_numpy=False,  # 返回 Python list 而非 numpy array（ChromaDB 兼容）
        normalize_embeddings=True  # ⭐ 关键：归一化后 dot = cosine 余弦相似度
    )

    # 转为普通 list（ChromaDB 接受 list[float]）
    embeddings = [emb.tolist() if hasattr(emb, 'tolist') else emb for emb in embeddings]

    return embeddings[0] if was_single else embeddings

# ================== Step 3: 初始化 ChromaDB（不再依赖远程 embedding）==================
client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

# 创建 collection，不指定 embedding_function（因为我们自己提供 embedding）
collection = client.get_or_create_collection(
    name="company_rules",
    metadata={"hnsw:space": "cosine"}  # cosine 距离，与 normalize_embeddings=True 匹配
)

# ================== Step 4: 将 PDF 内容写入向量库 ==================
def load_pdf_to_vector_db(pdf_path):
    print("📄 正在读取 PDF...")
    full_text = extract_text_from_pdf(pdf_path)
    chunks = split_text(full_text, chunk_size=300, overlap=50)
    print(f"✅ 切分为 {len(chunks)} 个段落")

    # 批量获取 embedding（本地模型，无需分批限制，但保留 batch_size 以防内存过大）
    batch_size = 32  # 可适当增大
    all_embeddings = []
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        print(f"🧠 正在处理第 {i//batch_size + 1} 批 Embedding（本地）...")
        embs = get_embedding(batch)
        all_embeddings.extend(embs)

    # 存入 ChromaDB
    collection.upsert(
        ids=[f"chunk_{i}" for i in range(len(chunks))],
        documents=chunks,
        embeddings=all_embeddings,
        metadatas=[{"source": os.path.basename(pdf_path), "id": f"chunk_{i}"} for i in range(len(chunks))]
    )
    print("💾 已将文档存入本地向量数据库！")

# ================== Step 5: 检索最相关的内容 ==================
def retrieve_relevant_context(question, n_results=2):
    print("🔍 正在检索知识库...")
    q_emb = get_embedding(question)  # 本地 embedding
    results = collection.query(
        query_embeddings=[q_emb],
        n_results=n_results
    )
    return results['documents'][0]  # 返回 top 2 段落

# ================== Step 6: 调用 Qwen 生成回答 ==================
def call_qwen(prompt):
    try:
        response = Generation.call(
            model=QWEN_MODEL,
            messages=[
                {'role': 'system', 'content': '你是一个企业制度顾问，请根据提供的资料准确回答问题。'},
                {'role': 'user', 'content': prompt}
            ],
            temperature=0.5,
            top_p=0.8
        )
        if response.status_code == 200:
            return response.output.choices[0].message.content
        else:
            return f"❌ 调用失败：{response.status_code} {response.message}"
    except Exception as e:
        return f"🚨 调用出错：{str(e)}"

# ================== Step 7: 主问答逻辑 ==================
def ask_company_rules(question):
    context = retrieve_relevant_context(question, n_results=2)
    prompt = f"""
你是一名公司制度专家，请根据以下真实资料回答问题。请回答简洁、准确、口语化。

【参考资料】
{''.join(f"- {c}\n" for c in context)}

【问题】
{question}

⚠️ 要求：
1. 如果资料中没有相关信息，请说“暂时无法找到相关信息”。
2. 不要编造内容。
3. 回答用中文。
"""
    print("☁️ 正在调用 Qwen 大模型...")
    answer = call_qwen(prompt)
    return answer

# ================== 主程序入口 ==================
if __name__ == "__main__":
    # 第一次运行时加载 PDF（只需一次）
    if collection.count() == 0:
        print("📥 检测到数据库为空，正在导入 PDF 数据...")
        load_pdf_to_vector_db(PDF_PATH)
    else:
        print(f"📊 已加载 {collection.count()} 条数据")

    print("\n💬 欢迎使用【公司制度智能助手】（输入 'quit' 退出）")
    print("💡 示例问题：")
    print("  • 年假怎么申请？")
    print("  • 加班有补贴吗？")
    print("  • 试用期多久？")

    while True:
        question = input("\n❓ 请输入你的问题：").strip()
        if question.lower() == 'quit':
            print("👋 感谢使用，再见！")
            break
        if not question:
            continue

        answer = ask_company_rules(question)
        print("\n" + "=" * 60)
        print("🤖 回答：")
        print(answer)
        print("=" * 60)
