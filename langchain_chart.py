import os
import sqlite3
import numpy as np
import streamlit as st
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain.memory import ConversationBufferMemory
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# 1. 加载并分割 PDF 文档
def load_and_split_pdf(pdf_path):
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()

    # 分割文档为多个部分
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(pages)

    return chunks


# 2. 使用本地模型（例如 Sentence-Transformers）进行向量化存储到 SQLite
def create_vector_db(chunks, model_path, db_path="./vector_db.sqlite"):
    # 加载本地模型
    model = SentenceTransformer(model_path)

    # 连接 SQLite 数据库
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # 创建表（如果不存在）
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS embeddings (
        document_id INTEGER PRIMARY KEY,
        chunk_id INTEGER,
        embedding BLOB,
        content TEXT
    )
    """)

    # 插入嵌入（embedding）数据
    for doc_id, chunk in enumerate(chunks):
        embedding = model.encode(chunk.page_content)
        cursor.execute("""
        INSERT INTO embeddings (document_id, chunk_id, embedding, content)
        VALUES (?, ?, ?, ?)
        """, (doc_id, 0, embedding.tobytes(), chunk.page_content))

    conn.commit()
    conn.close()


# 3. 查询数据库中的嵌入（embedding）并返回最相似的结果
def query_vector_db(query, model_path, db_path="./vector_db.sqlite"):
    model = SentenceTransformer(model_path)
    query_embedding = model.encode(query)

    # 连接到数据库
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # 从数据库中取出所有嵌入数据
    cursor.execute("SELECT chunk_id, embedding, content FROM embeddings")
    rows = cursor.fetchall()

    # 计算余弦相似度并返回最相似的文档
    similarities = []
    for row in rows:
        stored_embedding = np.frombuffer(row[1], dtype=np.float32)
        sim = cosine_similarity([query_embedding], [stored_embedding])
        similarities.append((row[2], sim[0][0]))

    # 按照相似度排序并返回最相关的文档
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[0][0]  # 返回最相似的文档


# 4. 初始化问答链
def create_qa_chain():
    # 使用 BM25 和 SQLite 向量数据库创建 EnsembleRetriever
    bm25_retriever = BM25Retriever.from_documents(chunks)
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever],
        weights=[1.0]
    )

    # 增加内存功能，支持多轮对话
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    # 自定义问答链
    class LocalRetrievalQA(RetrievalQA):
        def __init__(self, retriever, memory):
            super().__init__(retriever=retriever, memory=memory)

        def run(self, query):
            # 执行检索
            context = query_vector_db(query, model_path="./models/all-MiniLM-L6-v2")
            return context  # 返回检索到的上下文

    # 构建问答链
    qa_chain = LocalRetrievalQA(
        retriever=ensemble_retriever,
        memory=memory
    )

    return qa_chain


# 5. Streamlit 界面
st.set_page_config(page_title="PDF 智能助手")
st.header("📄 本地 PDF 问答系统")

# 上传 PDF 文件
uploaded_file = st.file_uploader("上传 PDF 文件", type="pdf")
if uploaded_file:
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())

    # 触发文档处理逻辑
    chunks = load_and_split_pdf("temp.pdf")

    # 输入本地模型路径
    model_path = st.text_input("请输入本地模型路径：", "./models/all-MiniLM-L6-v2")
    if model_path:
        create_vector_db(chunks, model_path)
        st.success("文档处理完成！")

# 提问与回答
user_question = st.text_input("输入您的问题：")
if user_question:
    response = query_vector_db(user_question, model_path="./models/all-MiniLM-L6-v2")
    st.write("回答：", response)
