#novel_generator/finalization.py
# -*- coding: utf-8 -*-
"""
章节定稿和扩写模块
功能：
1. 章节定稿处理：更新全局摘要和角色状态
2. 将章节内容存入向量库
3. 章节扩写功能：丰富章节内容
"""

import os  # 文件和目录操作
import logging  # 日志记录

from llm_adapters import create_llm_adapter  # LLM适配器创建
from embedding_adapters import create_embedding_adapter  # Embedding适配器创建
from prompt_definitions import (
    summary_prompt,  # 摘要生成提示词
    update_character_state_prompt  # 角色状态更新提示词
)
from novel_generator.common import invoke_with_cleaning  # LLM调用通用函数
from utils import (
    read_file,  # 读取文件内容
    clear_file_content,  # 清空文件内容
    save_string_to_txt  # 保存字符串到文本文件
)
from novel_generator.vectorstore_utils import update_vector_store  # 更新向量库
logging.basicConfig(
    filename='app.log',      # 日志文件名
    filemode='a',            # 追加模式（'w' 会覆盖）
    level=logging.INFO,      # 记录 INFO 及以上级别的日志
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
def finalize_chapter(
    novel_number: int,
    word_number: int,
    api_key: str,
    base_url: str,
    model_name: str,
    temperature: float,
    filepath: str,
    embedding_api_key: str,
    embedding_url: str,
    embedding_interface_format: str,
    embedding_model_name: str,
    interface_format: str,
    max_tokens: int,
    timeout: int = 600
):
    """
    章节定稿处理
    
    参数:
        novel_number: 章节编号
        word_number: 目标字数
        api_key: LLM服务的API密钥
        base_url: LLM服务的基础URL
        model_name: 使用的LLM模型名称
        temperature: 生成温度参数
        filepath: 项目文件路径
        embedding_api_key: Embedding服务的API密钥
        embedding_url: Embedding服务的基础URL
        embedding_interface_format: Embedding接口格式
        embedding_model_name: 使用的Embedding模型名称
        interface_format: LLM接口格式
        max_tokens: 最大生成token数
        timeout: 请求超时时间（秒）
    
    功能:
        1. 读取章节文本
        2. 生成章节摘要并更新全局摘要
        3. 更新角色状态
        4. 将章节内容存入向量库
        
    注意:
        默认不做扩写操作，如需扩写应在外部先调用enrich_chapter_text
    """
    # 构造章节目录和文件路径
    chapters_dir = os.path.join(filepath, "chapters")
    chapter_file = os.path.join(chapters_dir, f"chapter_{novel_number}.txt")
    # 读取章节文本
    chapter_text = read_file(chapter_file).strip()
    # 检查章节内容是否为空
    if not chapter_text:
        logging.warning(f"Chapter {novel_number} is empty, cannot finalize.")
        return

    # 读取全局摘要文件
    global_summary_file = os.path.join(filepath, "global_summary.txt")
    old_global_summary = read_file(global_summary_file)
    # 读取角色状态文件
    character_state_file = os.path.join(filepath, "character_state.txt")
    old_character_state = read_file(character_state_file)

    # 创建LLM适配器
    llm_adapter = create_llm_adapter(
        interface_format=interface_format,
        base_url=base_url,
        model_name=model_name,
        api_key=api_key,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout
    )

    # 生成新的全局摘要
    prompt_summary = summary_prompt.format(
        chapter_text=chapter_text,
        global_summary=old_global_summary
    )
    new_global_summary = invoke_with_cleaning(llm_adapter, prompt_summary)
    # 如果生成失败，使用旧摘要
    if not new_global_summary.strip():
        new_global_summary = old_global_summary

    # 生成新的角色状态
    prompt_char_state = update_character_state_prompt.format(
        chapter_text=chapter_text,
        old_state=old_character_state
    )
    new_char_state = invoke_with_cleaning(llm_adapter, prompt_char_state)
    # 如果生成失败，使用旧状态
    if not new_char_state.strip():
        new_char_state = old_character_state

    # 保存新的全局摘要
    clear_file_content(global_summary_file)
    save_string_to_txt(new_global_summary, global_summary_file)
    # 保存新的角色状态
    clear_file_content(character_state_file)
    save_string_to_txt(new_char_state, character_state_file)

    # 更新向量库，将新章节内容存入
    update_vector_store(
        embedding_adapter=create_embedding_adapter(
            embedding_interface_format,
            embedding_api_key,
            embedding_url,
            embedding_model_name
        ),
        new_chapter=chapter_text,
        filepath=filepath
    )

    logging.info(f"Chapter {novel_number} has been finalized.")

def enrich_chapter_text(
    chapter_text: str,
    word_number: int,
    api_key: str,
    base_url: str,
    model_name: str,
    temperature: float,
    interface_format: str,
    max_tokens: int,
    timeout: int=600
) -> str:
    """
    章节文本扩写
    
    参数:
        chapter_text: 原始章节文本
        word_number: 目标字数
        api_key: LLM服务的API密钥
        base_url: LLM服务的基础URL
        model_name: 使用的LLM模型名称
        temperature: 生成温度参数
        interface_format: LLM接口格式
        max_tokens: 最大生成token数
        timeout: 请求超时时间（秒）
    
    返回:
        str: 扩写后的章节文本
    
    功能:
        对章节文本进行扩写，使其更接近目标字数
        同时保持剧情连贯和内容一致性
        通过添加细节描写、对话、场景描述等方式丰富内容
    """
    llm_adapter = create_llm_adapter(
        interface_format=interface_format,
        base_url=base_url,
        model_name=model_name,
        api_key=api_key,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout
    )
    prompt = f"""以下章节文本较短，请在保持剧情连贯的前提下进行扩写，使其更充实，接近 {word_number} 字左右，仅给出最终文本，不要解释任何内容。：
原内容：
{chapter_text}
"""
    enriched_text = invoke_with_cleaning(llm_adapter, prompt)
    return enriched_text if enriched_text else chapter_text
