#novel_generator/architecture.py
# -*- coding: utf-8 -*-
"""
小说总体架构生成模块
功能：
1. 生成小说的核心设定（主题、类型、篇幅等）
2. 创建角色动力学体系
3. 构建世界观设定
4. 设计三幕式情节架构
5. 初始化角色状态表
"""

import os  # 文件和目录操作
import json  # JSON数据处理
import logging  # 日志记录
import traceback  # 异常追踪

from novel_generator.common import invoke_with_cleaning  # 调用LLM的通用函数
from llm_adapters import create_llm_adapter  # 创建LLM适配器
from prompt_definitions import (
    core_seed_prompt,  # 核心种子生成提示词
    character_dynamics_prompt,  # 角色动力学生成提示词
    world_building_prompt,  # 世界观生成提示词
    plot_architecture_prompt,  # 情节架构生成提示词
    create_character_state_prompt  # 角色状态创建提示词
)
logging.basicConfig(
    filename='app.log',      # 日志文件名
    filemode='a',            # 追加模式（'w' 会覆盖）
    level=logging.INFO,      # 记录 INFO 及以上级别的日志
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
from utils import clear_file_content, save_string_to_txt

def load_partial_architecture_data(filepath: str) -> dict:
    """
    从指定路径加载部分架构数据
    
    参数:
        filepath: 项目文件路径
    
    返回:
        dict: 已保存的部分架构数据，如果文件不存在或解析失败则返回空字典
    
    功能:
        读取partial_architecture.json文件，用于断点续传
        允许在生成过程中断后从上次停止的位置继续
    """
    partial_file = os.path.join(filepath, "partial_architecture.json")
    if not os.path.exists(partial_file):
        return {}
    try:
        with open(partial_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data
    except Exception as e:
        logging.warning(f"Failed to load partial_architecture.json: {e}")
        return {}

def save_partial_architecture_data(filepath: str, data: dict):
    """
    保存部分架构数据到文件
    
    参数:
        filepath: 项目文件路径
        data: 要保存的架构数据字典
    
    功能:
        将当前生成的架构数据保存到partial_architecture.json
        用于支持断点续传功能
    """
    partial_file = os.path.join(filepath, "partial_architecture.json")
    try:
        with open(partial_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logging.warning(f"Failed to save partial_architecture.json: {e}")

def Novel_architecture_generate(
    interface_format: str,
    api_key: str,
    base_url: str,
    llm_model: str,
    topic: str,
    genre: str,
    number_of_chapters: int,
    word_number: int,
    filepath: str,
    user_guidance: str = "",  # 用户提供的额外指导内容
    temperature: float = 0.7,
    max_tokens: int = 2048,
    timeout: int = 600
) -> None:
    """
    生成小说总体架构
    
    参数:
        interface_format: LLM接口格式（如OpenAI、DeepSeek等）
        api_key: LLM服务的API密钥
        base_url: LLM服务的基础URL
        llm_model: 使用的LLM模型名称
        topic: 小说主题
        genre: 小说类型
        number_of_chapters: 总章节数
        word_number: 每章目标字数
        filepath: 文件保存路径
        user_guidance: 用户提供的额外指导内容
        temperature: 生成温度参数（0-1，越高越有创造性）
        max_tokens: 最大生成token数
        timeout: 请求超时时间（秒）
    
    返回:
        None
    
    功能:
        依次调用四个步骤生成小说架构:
        1. core_seed_prompt - 生成核心种子（主题、类型、篇幅）
        2. character_dynamics_prompt - 创建角色动力学体系
        3. world_building_prompt - 构建世界观设定
        4. plot_architecture_prompt - 设计三幕式情节架构
        
        支持断点续传：
        - 若在中间任何一步报错，将已生成内容保存到partial_architecture.json
        - 下次调用时从该步骤继续
        
        生成初始角色状态:
        - 在完成角色动力学后，生成初始角色状态表
        - 保存到character_state.txt，供后续章节更新使用
        
        最终输出:
        - Novel_architecture.txt - 包含完整架构的文件
    """
    # 创建输出目录（如果不存在）
    os.makedirs(filepath, exist_ok=True)
    
    # 加载已有的部分架构数据（用于断点续传）
    partial_data = load_partial_architecture_data(filepath)
    
    # 创建LLM适配器实例
    llm_adapter = create_llm_adapter(
        interface_format=interface_format,
        base_url=base_url,
        model_name=llm_model,
        api_key=api_key,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout
    )
    
    # ============ Step1: 生成核心种子 ============
    if "core_seed_result" not in partial_data:
        logging.info("Step1: Generating core_seed_prompt (核心种子) ...")
        # 格式化提示词，填入用户提供的参数
        prompt_core = core_seed_prompt.format(
            topic=topic,
            genre=genre,
            number_of_chapters=number_of_chapters,
            word_number=word_number,
            user_guidance=user_guidance  # 添加用户指导内容
        )
        # 调用LLM生成核心种子
        core_seed_result = invoke_with_cleaning(llm_adapter, prompt_core)
        # 检查生成结果是否有效
        if not core_seed_result.strip():
            logging.warning("core_seed_prompt generation failed and returned empty.")
            # 保存当前进度后退出
            save_partial_architecture_data(filepath, partial_data)
            return
        # 保存生成结果
        partial_data["core_seed_result"] = core_seed_result
        save_partial_architecture_data(filepath, partial_data)
    else:
        logging.info("Step1 already done. Skipping...")
    
    # ============ Step2: 生成角色动力学 ============
    if "character_dynamics_result" not in partial_data:
        logging.info("Step2: Generating character_dynamics_prompt ...")
        # 基于核心种子生成角色体系
        prompt_character = character_dynamics_prompt.format(
            core_seed=partial_data["core_seed_result"].strip(),
            user_guidance=user_guidance
        )
        # 调用LLM生成角色动力学
        character_dynamics_result = invoke_with_cleaning(llm_adapter, prompt_character)
        # 检查生成结果
        if not character_dynamics_result.strip():
            logging.warning("character_dynamics_prompt generation failed.")
            save_partial_architecture_data(filepath, partial_data)
            return
        # 保存生成结果
        partial_data["character_dynamics_result"] = character_dynamics_result
        save_partial_architecture_data(filepath, partial_data)
    else:
        logging.info("Step2 already done. Skipping...")
    # 生成初始角色状态
    if "character_dynamics_result" in partial_data and "character_state_result" not in partial_data:
        logging.info("Generating initial character state from character dynamics ...")
        prompt_char_state_init = create_character_state_prompt.format(
            character_dynamics=partial_data["character_dynamics_result"].strip()
        )
        character_state_init = invoke_with_cleaning(llm_adapter, prompt_char_state_init)
        if not character_state_init.strip():
            logging.warning("create_character_state_prompt generation failed.")
            save_partial_architecture_data(filepath, partial_data)
            return
        partial_data["character_state_result"] = character_state_init
        character_state_file = os.path.join(filepath, "character_state.txt")
        clear_file_content(character_state_file)
        save_string_to_txt(character_state_init, character_state_file)
        save_partial_architecture_data(filepath, partial_data)
        logging.info("Initial character state created and saved.")
    # Step3: 世界观
    if "world_building_result" not in partial_data:
        logging.info("Step3: Generating world_building_prompt ...")
        prompt_world = world_building_prompt.format(
            core_seed=partial_data["core_seed_result"].strip(),
            user_guidance=user_guidance  # 修复：添加用户指导
        )
        world_building_result = invoke_with_cleaning(llm_adapter, prompt_world)
        if not world_building_result.strip():
            logging.warning("world_building_prompt generation failed.")
            save_partial_architecture_data(filepath, partial_data)
            return
        partial_data["world_building_result"] = world_building_result
        save_partial_architecture_data(filepath, partial_data)
    else:
        logging.info("Step3 already done. Skipping...")
    # Step4: 三幕式情节
    if "plot_arch_result" not in partial_data:
        logging.info("Step4: Generating plot_architecture_prompt ...")
        prompt_plot = plot_architecture_prompt.format(
            core_seed=partial_data["core_seed_result"].strip(),
            character_dynamics=partial_data["character_dynamics_result"].strip(),
            world_building=partial_data["world_building_result"].strip(),
            user_guidance=user_guidance  # 修复：添加用户指导
        )
        plot_arch_result = invoke_with_cleaning(llm_adapter, prompt_plot)
        if not plot_arch_result.strip():
            logging.warning("plot_architecture_prompt generation failed.")
            save_partial_architecture_data(filepath, partial_data)
            return
        partial_data["plot_arch_result"] = plot_arch_result
        save_partial_architecture_data(filepath, partial_data)
    else:
        logging.info("Step4 already done. Skipping...")

    # 从partial_data中提取所有生成结果
    core_seed_result = partial_data["core_seed_result"]
    character_dynamics_result = partial_data["character_dynamics_result"]
    world_building_result = partial_data["world_building_result"]
    plot_arch_result = partial_data["plot_arch_result"]

    # 组合所有内容为最终架构文本
    final_content = (
        "#=== 0) 小说设定 ===\n"
        f"主题：{topic},类型：{genre},篇幅：约{number_of_chapters}章（每章{word_number}字）\n\n"
        "#=== 1) 核心种子 ===\n"
        f"{core_seed_result}\n\n"
        "#=== 2) 角色动力学 ===\n"
        f"{character_dynamics_result}\n\n"
        "#=== 3) 世界观 ===\n"
        f"{world_building_result}\n\n"
        "#=== 4) 三幕式情节架构 ===\n"
        f"{plot_arch_result}\n"
    )

    # 保存完整架构到文件
    arch_file = os.path.join(filepath, "Novel_architecture.txt")
    clear_file_content(arch_file)
    save_string_to_txt(final_content, arch_file)
    logging.info("Novel_architecture.txt has been generated successfully.")

    # 删除临时文件（所有步骤已完成）
    partial_arch_file = os.path.join(filepath, "partial_architecture.json")
    if os.path.exists(partial_arch_file):
        os.remove(partial_arch_file)
        logging.info("partial_architecture.json removed (all steps completed).")
