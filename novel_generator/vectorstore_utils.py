#novel_generator/vectorstore_utils.py
# -*- coding: utf-8 -*-
"""
向量库操作模块
功能：
1. 向量库初始化和加载
2. 向量库更新和添加新内容
3. 相似度检索和上下文获取
4. 文本分段处理
5. 向量库清空和管理
"""

import os  # 文件和目录操作
import logging  # 日志记录
import traceback  # 异常追踪
import nltk  # 自然语言处理工具包
import numpy as np  # 数值计算
import re  # 正则表达式处理
import ssl  # SSL安全连接
import requests  # HTTP请求
import warnings  # 警告控制
from langchain_chroma import Chroma  # Chroma向量数据库
logging.basicConfig(
    filename='app.log',      # 日志文件名
    filemode='a',            # 追加模式（'w' 会覆盖）
    level=logging.INFO,      # 记录 INFO 及以上级别的日志
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
# 禁用特定的Torch警告
warnings.filterwarnings('ignore', message='.*Torch was not compiled with flash attention.*')
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # 禁用tokenizer并行警告

from chromadb.config import Settings  # Chroma配置
from langchain.docstore.document import Document  # 文档对象
from sklearn.metrics.pairwise import cosine_similarity  # 余弦相似度计算
from .common import call_with_retry

def get_vectorstore_dir(filepath: str) -> str:
    """
    获取向量库存储路径
    
    参数:
        filepath: 项目文件路径
    
    返回:
        str: 向量库目录的完整路径
    """
    return os.path.join(filepath, "vectorstore")

def clear_vector_store(filepath: str) -> bool:
    """
    清空向量库
    
    参数:
        filepath: 项目文件路径
    
    返回:
        bool: 清空成功返回True，失败返回False
    
    功能:
        删除整个向量库目录，包括所有存储的向量数据
        用于重置向量库或释放存储空间
    """
    import shutil  # 文件和目录操作
    store_dir = get_vectorstore_dir(filepath)
    # 检查向量库是否存在
    if not os.path.exists(store_dir):
        logging.info("No vector store found to clear.")
        return False
    try:
        # 删除向量库目录及其所有内容
        shutil.rmtree(store_dir)
        logging.info(f"Vector store directory '{store_dir}' removed.")
        return True
    except Exception as e:
        logging.error(f"无法删除向量库文件夹，请关闭程序后手动删除 {store_dir}。\n {str(e)}")
        traceback.print_exc()
        return False

def init_vector_store(embedding_adapter, texts, filepath: str):
    """
    初始化向量库
    
    参数:
        embedding_adapter: Embedding适配器实例
        texts: 要添加到向量库的文本列表
        filepath: 项目文件路径
    
    返回:
        Chroma向量库实例，失败时返回None
    
    功能:
        1. 在指定路径下创建向量库目录
        2. 将文本转换为Document对象
        3. 使用Embedding适配器将文本向量化
        4. 创建并返回Chroma向量库实例
        
    注意:
        如果Embedding失败，返回None而不中断任务
    """
    from langchain.embeddings.base import Embeddings as LCEmbeddings  # LangChain Embeddings基类

    # 获取并创建向量库目录
    store_dir = get_vectorstore_dir(filepath)
    os.makedirs(store_dir, exist_ok=True)
    # 将文本转换为Document对象
    documents = [Document(page_content=str(t)) for t in texts]

    try:
        # 创建Embedding包装类，实现LangChain的Embeddings接口
        class LCEmbeddingWrapper(LCEmbeddings):
            # 文档向量化方法
            def embed_documents(self, texts):
                return call_with_retry(
                    func=embedding_adapter.embed_documents,
                    max_retries=3,  # 最多重试3次
                    fallback_return=[],  # 失败时返回空列表
                    texts=texts
                )
            # 查询向量化方法
            def embed_query(self, query: str):
                res = call_with_retry(
                    func=embedding_adapter.embed_query,
                    max_retries=3,  # 最多重试3次
                    fallback_return=[],  # 失败时返回空列表
                    query=query
                )
                return res

        chroma_embedding = LCEmbeddingWrapper()
        vectorstore = Chroma.from_documents(
            documents,
            embedding=chroma_embedding,
            persist_directory=store_dir,
            client_settings=Settings(anonymized_telemetry=False),
            collection_name="novel_collection"
        )
        return vectorstore
    except Exception as e:
        logging.warning(f"Init vector store failed: {e}")
        traceback.print_exc()
        return None

def load_vector_store(embedding_adapter, filepath: str):
    """
    加载已存在的向量库
    
    参数:
        embedding_adapter: Embedding适配器实例
        filepath: 项目文件路径
    
    返回:
        Chroma向量库实例，不存在或加载失败时返回None
    
    功能:
        从磁盘加载已存在的Chroma向量库
        用于后续的相似度检索和查询操作
    """
    from langchain.embeddings.base import Embeddings as LCEmbeddings  # LangChain Embeddings基类
    store_dir = get_vectorstore_dir(filepath)
    # 检查向量库目录是否存在
    if not os.path.exists(store_dir):
        logging.info("Vector store not found. Will return None.")
        return None

    try:
        class LCEmbeddingWrapper(LCEmbeddings):
            def embed_documents(self, texts):
                return call_with_retry(
                    func=embedding_adapter.embed_documents,
                    max_retries=3,
                    fallback_return=[],
                    texts=texts
                )
            def embed_query(self, query: str):
                res = call_with_retry(
                    func=embedding_adapter.embed_query,
                    max_retries=3,
                    fallback_return=[],
                    query=query
                )
                return res

        chroma_embedding = LCEmbeddingWrapper()
        return Chroma(
            persist_directory=store_dir,
            embedding_function=chroma_embedding,
            client_settings=Settings(anonymized_telemetry=False),
            collection_name="novel_collection"
        )
    except Exception as e:
        logging.warning(f"Failed to load vector store: {e}")
        traceback.print_exc()
        return None

def split_by_length(text: str, max_length: int = 500):
    """
    按固定长度切分文本
    
    参数:
        text: 要切分的文本
        max_length: 每段的最大长度（默认500字符）
    
    返回:
        list: 切分后的文本段列表
    
    功能:
        将文本按照固定长度切分成多个段落
        用于处理长文本，避免超出向量库的长度限制
    """
    segments = []  # 存储切分后的文本段
    start_idx = 0  # 起始索引
    # 循环切分文本
    while start_idx < len(text):
        end_idx = min(start_idx + max_length, len(text))  # 计算结束索引
        segment = text[start_idx:end_idx]  # 提取文本段
        segments.append(segment.strip())  # 添加到列表
        start_idx = end_idx  # 移动到下一段
    return segments

def split_text_for_vectorstore(chapter_text: str, max_length: int = 500, similarity_threshold: float = 0.7):
    """
    智能切分文本用于向量库
    
    参数:
        chapter_text: 章节文本
        max_length: 每段的最大长度（默认500字符）
        similarity_threshold: 相似度阈值（默认0.7）
    
    返回:
        list: 切分后的文本段列表
    
    功能:
        1. 使用NLTK进行句子分割
        2. 根据语义相似度合并相邻句子
        3. 确保每段不超过最大长度
        4. 保持语义完整性，避免在句子中间切分
    """
    if not chapter_text.strip():
        return []
    
    # nltk.download('punkt', quiet=True)
    # nltk.download('punkt_tab', quiet=True)
    sentences = nltk.sent_tokenize(chapter_text)
    if not sentences:
        return []
    
    # 直接按长度分段,不做相似度合并
    final_segments = []
    current_segment = []
    current_length = 0
    
    for sentence in sentences:
        sentence_length = len(sentence)
        if current_length + sentence_length > max_length:
            if current_segment:
                final_segments.append(" ".join(current_segment))
            current_segment = [sentence]
            current_length = sentence_length
        else:
            current_segment.append(sentence)
            current_length += sentence_length
    
    if current_segment:
        final_segments.append(" ".join(current_segment))
    
    return final_segments

def update_vector_store(embedding_adapter, new_chapter: str, filepath: str):
    """
    将最新章节文本插入到向量库中。
    若库不存在则初始化；若初始化/更新失败，则跳过。
    """
    from utils import read_file, clear_file_content, save_string_to_txt
    splitted_texts = split_text_for_vectorstore(new_chapter)
    if not splitted_texts:
        logging.warning("No valid text to insert into vector store. Skipping.")
        return

    store = load_vector_store(embedding_adapter, filepath)
    if not store:
        logging.info("Vector store does not exist or failed to load. Initializing a new one for new chapter...")
        store = init_vector_store(embedding_adapter, splitted_texts, filepath)
        if not store:
            logging.warning("Init vector store failed, skip embedding.")
        else:
            logging.info("New vector store created successfully.")
        return

    try:
        docs = [Document(page_content=str(t)) for t in splitted_texts]
        store.add_documents(docs)
        logging.info("Vector store updated with the new chapter splitted segments.")
    except Exception as e:
        logging.warning(f"Failed to update vector store: {e}")
        traceback.print_exc()

def get_relevant_context_from_vector_store(embedding_adapter, query: str, filepath: str, k: int = 2) -> str:
    """
    从向量库中检索与 query 最相关的 k 条文本，拼接后返回。
    如果向量库加载/检索失败，则返回空字符串。
    最终只返回最多2000字符的检索片段。
    """
    store = load_vector_store(embedding_adapter, filepath)
    if not store:
        logging.info("No vector store found or load failed. Returning empty context.")
        return ""

    try:
        docs = store.similarity_search(query, k=k)
        if not docs:
            logging.info(f"No relevant documents found for query '{query}'. Returning empty context.")
            return ""
        combined = "\n".join([d.page_content for d in docs])
        if len(combined) > 2000:
            combined = combined[:2000]
        return combined
    except Exception as e:
        logging.warning(f"Similarity search failed: {e}")
        traceback.print_exc()
        return ""

def _get_sentence_transformer(model_name: str = 'paraphrase-MiniLM-L6-v2'):
    """获取sentence transformer模型，处理SSL问题"""
    try:
        # 设置torch环境变量
        os.environ["TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"] = "0"
        os.environ["TORCH_CUDNN_V8_API_ENABLED"] = "0"
        
        # 禁用SSL验证
        ssl._create_default_https_context = ssl._create_unverified_context
        
        # ...existing code...
    except Exception as e:
        logging.error(f"Failed to load sentence transformer model: {e}")
        traceback.print_exc()
        return None
