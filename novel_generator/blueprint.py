#novel_generator/blueprint.py
# -*- coding: utf-8 -*-
"""
章节蓝图生成模块
功能：
1. 根据小说架构生成完整的章节目录
2. 支持分块生成，避免单次生成内容过长
3. 支持断点续传，从已有目录继续生成
4. 每章包含标题、定位、作用、人物、道具、场景等详细信息
"""

import os  # 文件和目录操作
import re  # 正则表达式处理
import logging  # 日志记录

from novel_generator.common import invoke_with_cleaning  # LLM调用通用函数
from llm_adapters import create_llm_adapter  # LLM适配器创建
from prompt_definitions import (
    chapter_blueprint_prompt,  # 单次生成章节目录的提示词
    chunked_chapter_blueprint_prompt  # 分块生成章节目录的提示词
)
from utils import (
    read_file,  # 读取文件内容
    clear_file_content,  # 清空文件内容
    save_string_to_txt  # 保存字符串到文本文件
)
logging.basicConfig(
    filename='app.log',      # 日志文件名
    filemode='a',            # 追加模式（'w' 会覆盖）
    level=logging.INFO,      # 记录 INFO 及以上级别的日志
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
def compute_chunk_size(number_of_chapters: int, max_tokens: int) -> int:
    """
    基于“每章约100 tokens”的粗略估算，
    再结合当前max_tokens，计算分块大小：
      chunk_size = (floor(max_tokens/100/10)*10) - 10
    并确保 chunk_size 不会小于1或大于实际章节数。
    """
    tokens_per_chapter = 200.0
    ratio = max_tokens / tokens_per_chapter
    ratio_rounded_to_10 = int(ratio // 10) * 10
    chunk_size = ratio_rounded_to_10 - 10
    if chunk_size < 1:
        chunk_size = 1
    if chunk_size > number_of_chapters:
        chunk_size = number_of_chapters
    return chunk_size

def limit_chapter_blueprint(blueprint_text: str, limit_chapters: int = 100) -> str:
    """
    从已有章节目录中只取最近的 limit_chapters 章，以避免 prompt 超长。
    """
    pattern = r"(第\s*\d+\s*章.*?)(?=第\s*\d+\s*章|$)"
    chapters = re.findall(pattern, blueprint_text, flags=re.DOTALL)
    if not chapters:
        return blueprint_text
    if len(chapters) <= limit_chapters:
        return blueprint_text
    selected = chapters[-limit_chapters:]
    return "\n\n".join(selected).strip()

def Chapter_blueprint_generate(
    interface_format: str,
    api_key: str,
    base_url: str,
    llm_model: str,
    filepath: str,
    number_of_chapters: int,
    user_guidance: str = "",  # 用户提供的额外指导内容
    temperature: float = 0.7,
    max_tokens: int = 4096,
    timeout: int = 600
) -> None:
    """
    生成章节蓝图（章节目录）
    
    参数:
        interface_format: LLM接口格式（如OpenAI、DeepSeek等）
        api_key: LLM服务的API密钥
        base_url: LLM服务的基础URL
        llm_model: 使用的LLM模型名称
        filepath: 文件保存路径
        number_of_chapters: 总章节数
        user_guidance: 用户提供的额外指导内容
        temperature: 生成温度参数
        max_tokens: 最大生成token数
        timeout: 请求超时时间（秒）
    
    返回:
        None
    
    功能:
        1. 检查是否存在已有章节目录（Novel_directory.txt）
        2. 如果存在且非空：
           - 解析已有的章节数
           - 从下一章节继续分块生成
           - 传入时仅保留最近100章目录，避免prompt过长
        3. 如果不存在：
           - 若章节数 <= chunk_size，直接一次性生成
           - 若章节数 > chunk_size，进行分块生成
        4. 生成完成后输出到Novel_directory.txt
        
    分块生成策略:
        - 计算合适的chunk_size（基于max_tokens）
        - 每次生成chunk_size章
        - 将已有目录（限制为最近100章）作为上下文
        - 循环生成直到完成所有章节
    """
    # 检查架构文件是否存在
    arch_file = os.path.join(filepath, "Novel_architecture.txt")
    if not os.path.exists(arch_file):
        logging.warning("Novel_architecture.txt not found. Please generate architecture first.")
        return
    
    # 读取架构内容
    architecture_text = read_file(arch_file).strip()
    if not architecture_text:
        logging.warning("Novel_architecture.txt is empty.")
        return

    # 创建LLM适配器
    llm_adapter = create_llm_adapter(
        interface_format=interface_format,
        base_url=base_url,
        model_name=llm_model,
        api_key=api_key,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout
    )

    # 准备目录文件
    filename_dir = os.path.join(filepath, "Novel_directory.txt")
    if not os.path.exists(filename_dir):
        open(filename_dir, "w", encoding="utf-8").close()

    # 读取已有的目录内容（用于断点续传）
    existing_blueprint = read_file(filename_dir).strip()
    # 计算分块大小
    chunk_size = compute_chunk_size(number_of_chapters, max_tokens)
    logging.info(f"Number of chapters = {number_of_chapters}, computed chunk_size = {chunk_size}.")

    # ============ 断点续传逻辑 ============
    if existing_blueprint:
        logging.info("Detected existing blueprint content. Will resume chunked generation from that point.")
        # 使用正则表达式提取已有的章节编号
        pattern = r"第\s*(\d+)\s*章"
        existing_chapter_numbers = re.findall(pattern, existing_blueprint)
        # 转换为整数并过滤无效值
        existing_chapter_numbers = [int(x) for x in existing_chapter_numbers if x.isdigit()]
        # 找到最大章节号
        max_existing_chap = max(existing_chapter_numbers) if existing_chapter_numbers else 0
        logging.info(f"Existing blueprint indicates up to chapter {max_existing_chap} has been generated.")
        # 从已有内容继续
        final_blueprint = existing_blueprint
        current_start = max_existing_chap + 1
        # 分块生成剩余章节
        while current_start <= number_of_chapters:
            # 计算当前块的结束章节
            current_end = min(current_start + chunk_size - 1, number_of_chapters)
            # 限制已有目录为最近100章，避免prompt过长
            limited_blueprint = limit_chapter_blueprint(final_blueprint, 100)
            # 构造分块生成提示词
            chunk_prompt = chunked_chapter_blueprint_prompt.format(
                novel_architecture=architecture_text,
                chapter_list=limited_blueprint,
                number_of_chapters=number_of_chapters,
                n=current_start,
                m=current_end,
                user_guidance=user_guidance  # 添加用户指导内容
            )
            logging.info(f"Generating chapters [{current_start}..{current_end}] in a chunk...")
            # 调用LLM生成当前块
            chunk_result = invoke_with_cleaning(llm_adapter, chunk_prompt)
            # 检查生成结果
            if not chunk_result.strip():
                logging.warning(f"Chunk generation for chapters [{current_start}..{current_end}] is empty.")
                # 保存当前进度后退出
                clear_file_content(filename_dir)
                save_string_to_txt(final_blueprint.strip(), filename_dir)
                return
            # 追加新内容到目录
            final_blueprint += "\n\n" + chunk_result.strip()
            # 保存更新后的目录
            clear_file_content(filename_dir)
            save_string_to_txt(final_blueprint.strip(), filename_dir)
            # 移动到下一块
            current_start = current_end + 1

        logging.info("All chapters blueprint have been generated (resumed chunked).")
        return

    # ============ 单次生成（章节数较少时） ============
    if chunk_size >= number_of_chapters:
        # 构造单次生成提示词
        prompt = chapter_blueprint_prompt.format(
            novel_architecture=architecture_text,
            number_of_chapters=number_of_chapters,
            user_guidance=user_guidance  # 添加用户指导内容
        )
        # 调用LLM生成完整目录
        blueprint_text = invoke_with_cleaning(llm_adapter, prompt)
        # 检查生成结果
        if not blueprint_text.strip():
            logging.warning("Chapter blueprint generation result is empty.")
            return
        # 保存生成的目录
        clear_file_content(filename_dir)
        save_string_to_txt(blueprint_text, filename_dir)
        logging.info("Novel_directory.txt (chapter blueprint) has been generated successfully (single-shot).")
        return

    # ============ 分块生成（章节数较多时） ============
    logging.info("Will generate chapter blueprint in chunked mode from scratch.")
    final_blueprint = ""
    current_start = 1
    # 循环分块生成所有章节
    while current_start <= number_of_chapters:
        # 计算当前块的结束章节
        current_end = min(current_start + chunk_size - 1, number_of_chapters)
        # 限制已有目录为最近100章
        limited_blueprint = limit_chapter_blueprint(final_blueprint, 100)
        # 构造分块生成提示词
        chunk_prompt = chunked_chapter_blueprint_prompt.format(
            novel_architecture=architecture_text,
            chapter_list=limited_blueprint,
            number_of_chapters=number_of_chapters,
            n=current_start,
            m=current_end,
            user_guidance=user_guidance  # 添加用户指导内容
        )
        logging.info(f"Generating chapters [{current_start}..{current_end}] in a chunk...")
        # 调用LLM生成当前块
        chunk_result = invoke_with_cleaning(llm_adapter, chunk_prompt)
        # 检查生成结果
        if not chunk_result.strip():
            logging.warning(f"Chunk generation for chapters [{current_start}..{current_end}] is empty.")
            # 保存当前进度后退出
            clear_file_content(filename_dir)
            save_string_to_txt(final_blueprint.strip(), filename_dir)
            return
        # 追加新内容到目录
        if final_blueprint.strip():
            final_blueprint += "\n\n" + chunk_result.strip()
        else:
            final_blueprint = chunk_result.strip()
        # 保存更新后的目录
        clear_file_content(filename_dir)
        save_string_to_txt(final_blueprint.strip(), filename_dir)
        # 移动到下一块
        current_start = current_end + 1

    logging.info("Novel_directory.txt (chapter blueprint) has been generated successfully (chunked).")
