# main.py
# -*- coding: utf-8 -*-
"""
程序主入口文件
功能：初始化GUI应用程序并启动主循环
"""

import customtkinter as ctk  # 导入customtkinter库用于创建现代化GUI
from ui import NovelGeneratorGUI  # 导入主GUI类

def main():
    """
    主函数
    创建应用程序实例、初始化GUI界面并启动主事件循环
    """
    app = ctk.CTk()  # 创建CTk主窗口实例
    gui = NovelGeneratorGUI(app)  # 初始化GUI界面
    app.mainloop()  # 启动主事件循环，保持窗口运行

if __name__ == "__main__":
    main()  # 当直接运行此文件时，执行main函数
