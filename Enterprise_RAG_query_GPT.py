# Enterprise_RAG_query_GPT.py from chatgpt
# This script builds a RAG system for enterprise data using ChromaDB and Qwen-max LLM via DashScope API. 
# It extracts text, tables, and images from DOCX, XLSX, PDF files, but not MySQL databases, then saves them to ChromaDB.
# Finally, it performs RAG queries using the LLM.
# Required libraries: langchain, chromadb, transformers, dashscope, docx, openpyxl, pandas, pdfplumber
# Make sure to install them via pip if not already installed.
# Note: You need to have access to the Qwen-max model via DashScope API.
# Set your DashScope API key in the environment variable DASHSCOPE_API_KEY.
#
import os
import glob
import docx
import openpyxl
import pandas as pd
import dashscope
from dashscope import Generation
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.schema import Document
from typing import List
import pdfplumber
import pytesseract
from PIL import Image

# Configurations
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")  # 用于 Qwen
QWEN_MODEL = "qwen3-max"
CHROMA_DB_DIR = "./chroma_db_corp"
EMBEDDING_MODEL_NAME = "./models/st_paraphrase-multilingual-MiniLM-L12-v2"  # Use local path

# Initialize embedding model
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

# Text splitter
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

def extract_text_from_docx(file_path):
    doc = docx.Document(file_path)
    return "\n".join([para.text for para in doc.paragraphs if para.text.strip()])

def extract_tables_from_docx(file_path):
    doc = docx.Document(file_path)
    tables = []
    for table in doc.tables:
        data = []
        for row in table.rows:
            data.append([cell.text for cell in row.cells])
        tables.append(pd.DataFrame(data))
    return tables

def extract_text_from_xlsx(file_path):
    wb = openpyxl.load_workbook(file_path, data_only=True)
    texts = []
    for sheet in wb.worksheets:
        for row in sheet.iter_rows(values_only=True):
            row_text = " | ".join([str(cell) if cell is not None else "" for cell in row])
            if row_text.strip():
                texts.append(row_text)
    return "\n".join(texts)

def extract_tables_from_xlsx(file_path):
    wb = openpyxl.load_workbook(file_path, data_only=True)
    tables = []
    for sheet in wb.worksheets:
        data = []
        for row in sheet.iter_rows(values_only=True):
            data.append([str(cell) if cell is not None else "" for cell in row])
        tables.append(pd.DataFrame(data))
    return tables

def extract_text_tables_images_from_pdf(file_path):
    texts = []
    tables = []
    images = []
    ocr_texts = []
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            # Extract text
            page_text = page.extract_text()
            if page_text:
                texts.append(page_text)
            # Extract tables
            for table in page.extract_tables():
                df = pd.DataFrame(table)
                tables.append(df)
            # Extract images and perform OCR
            for img in page.images:
                cropped = page.within_bbox((img["x0"], img["top"], img["x1"], img["bottom"]))
                pil_img = cropped.to_image(resolution=300).original
                images.append(pil_img)
                # OCR on image
                ocr_result = pytesseract.image_to_string(pil_img, lang="chi_sim+eng")
                if ocr_result.strip():
                    ocr_texts.append(ocr_result.strip())
    # Combine normal text and OCR text
    all_text = "\n".join(texts + ocr_texts)
    return all_text, tables, images
 # MySQL part removed #
def save_to_chroma(docs: List[Document]):
    db = Chroma.from_documents(
        docs,
        embeddings,
        persist_directory=CHROMA_DB_DIR
    )

def build_corpus_and_save(file_dir):
    docs = []
    # DOCX
    for file in glob.glob(os.path.join(file_dir, "*.docx")):
        text = extract_text_from_docx(file)
        for chunk in splitter.split_text(text):
            docs.append(Document(page_content=chunk, metadata={"source": file}))
        for table in extract_tables_from_docx(file):
            docs.append(Document(page_content=table.to_csv(index=False), metadata={"source": file, "type": "table"}))
    # XLSX
    for file in glob.glob(os.path.join(file_dir, "*.xlsx")):
        text = extract_text_from_xlsx(file)
        for chunk in splitter.split_text(text):
            docs.append(Document(page_content=chunk, metadata={"source": file}))
        for table in extract_tables_from_xlsx(file):
            docs.append(Document(page_content=table.to_csv(index=False), metadata={"source": file, "type": "table"}))
    # PDF
    for file in glob.glob(os.path.join(file_dir, "*.pdf")):
        text, tables, images = extract_text_tables_images_from_pdf(file)
        for chunk in splitter.split_text(text):
            docs.append(Document(page_content=chunk, metadata={"source": file}))
        # Save each PDF table to ChromaDB as well
        for table in tables:
            docs.append(Document(page_content=table.to_csv(index=False), metadata={"source": file, "type": "table"}))
        # Images can be saved as base64 or file paths if needed
    # MySQL part removed #
    save_to_chroma(docs)

def rag_query(query: str, top_k=5):
    db = Chroma(
        persist_directory=CHROMA_DB_DIR,
        embedding_function=embeddings
    )
    docs = db.similarity_search(query, k=top_k)
    context = "\n".join([doc.page_content for doc in docs])
    # Use messages for Qwen3
    prompt = f"根据以下内容回答问题：\n\n{context}\n\n问题：{query}\n答案："
    messages = [
        {"role": "system", "content": "你是一个企业知识问答助手，回答要简洁准确。"},
        {"role": "user", "content": prompt}
    ]
    response = Generation.call(
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        model="qwen3-max",
        messages=messages,
        result_format="message",
        top_p=0.8,
        temperature=0.3,
        max_tokens=256
    )
    # Extract answer
    answer = response.output.choices[0].message.content
    return answer

if __name__ == "__main__":
    # Step 1: Build corpus and save to ChromaDB
    build_corpus_and_save("./")  # Set your directory

    # Step 2: Interactive RAG query loop
    while True:
        question = input("请输入您的问题（输入 quit 退出）：")
        if question.strip().lower() == "quit":
            print("程序已退出。")
            break
        answer = rag_query(question)
        print("Answer:", answer)