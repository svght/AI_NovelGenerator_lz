#novel_generator/knowledge.py
# -*- coding: utf-8 -*-
"""
知识库管理模块
功能：
1. 知识文件内容分段处理
2. 将知识内容导入向量库
3. 支持智能分段，保持语义完整性
"""

import os  # 文件和目录操作
import logging  # 日志记录
import re  # 正则表达式处理
import traceback  # 异常追踪
import nltk  # 自然语言处理工具包
import warnings  # 警告控制

from utils import read_file  # 读取文件内容
from novel_generator.vectorstore_utils import (
    load_vector_store,  # 加载向量库
    init_vector_store  # 初始化向量库
)
from langchain.docstore.document import Document  # 文档对象

# 禁用特定的Torch警告
warnings.filterwarnings('ignore', message='.*Torch was not compiled with flash attention.*')
os.environ["TOKENIZERS_PARALLELISM"] = "false"
logging.basicConfig(
    filename='app.log',      # 日志文件名
    filemode='a',            # 追加模式（'w' 会覆盖）
    level=logging.INFO,      # 记录 INFO 及以上级别的日志
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
def advanced_split_content(content: str, similarity_threshold: float = 0.7, max_length: int = 500) -> list:
    """
    高级文本分段
    
    参数:
        content: 要分段的内容
        similarity_threshold: 相似度阈值（默认0.7）
        max_length: 每段最大长度（默认500字符）
    
    返回:
        list: 分段后的文本列表
    
    功能:
        使用NLTK进行句子分割
        根据语义相似度合并相邻句子
        保持语义完整性
    """
    # nltk.download('punkt', quiet=True)
    # nltk.download('punkt_tab', quiet=True)
    sentences = nltk.sent_tokenize(content)
    if not sentences:
        return []

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

def import_knowledge_file(
    embedding_api_key: str,
    embedding_url: str,
    embedding_interface_format: str,
    embedding_model_name: str,
    file_path: str,
    filepath: str
):
    """
    导入知识文件到向量库
    
    参数:
        embedding_api_key: Embedding服务的API密钥
        embedding_url: Embedding服务的基础URL
        embedding_interface_format: Embedding接口格式
        embedding_model_name: 使用的Embedding模型名称
        file_path: 知识库文件路径
        filepath: 项目文件路径
    
    功能:
        1. 读取知识库文件内容
        2. 对内容进行分段处理
        3. 创建或加载向量库
        4. 将分段后的内容添加到向量库
    """
    logging.info(f"开始导入知识库文件: {file_path}, 接口格式: {embedding_interface_format}, 模型: {embedding_model_name}")
    # 检查知识库文件是否存在
    if not os.path.exists(file_path):
        logging.warning(f"知识库文件不存在: {file_path}")
        return
    # 读取知识库文件内容
    content = read_file(file_path)
    # 检查内容是否为空
    if not content.strip():
        logging.warning("知识库文件内容为空。")
        return
    paragraphs = advanced_split_content(content)
    from embedding_adapters import create_embedding_adapter
    embedding_adapter = create_embedding_adapter(
        embedding_interface_format,
        embedding_api_key,
        embedding_url if embedding_url else "http://localhost:11434/api",
        embedding_model_name
    )
    store = load_vector_store(embedding_adapter, filepath)
    if not store:
        logging.info("Vector store does not exist or load failed. Initializing a new one for knowledge import...")
        store = init_vector_store(embedding_adapter, paragraphs, filepath)
        if store:
            logging.info("知识库文件已成功导入至向量库(新初始化)。")
        else:
            logging.warning("知识库导入失败，跳过。")
    else:
        try:
            docs = [Document(page_content=str(p)) for p in paragraphs]
            store.add_documents(docs)
            logging.info("知识库文件已成功导入至向量库(追加模式)。")
        except Exception as e:
            logging.warning(f"知识库导入失败: {e}")
            traceback.print_exc()
