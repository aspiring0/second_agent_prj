# src/agent/tools_dir/__init__.py
"""工具自动发现和加载"""

import importlib
from pathlib import Path
from langchain_core.tools import BaseTool
from src.utils.logger import setup_logger

logger = setup_logger("Tools_Loader")


def discover_tools() -> list:
    """自动扫描 tools_dir/ 目录下的所有模块，收集工具"""
    all_tools = []
    tools_dir = Path(__file__).parent

    for py_file in tools_dir.glob("*.py"):
        if py_file.name.startswith("_"):
            continue
        module_name = f"src.agent.tools_dir.{py_file.stem}"
        try:
            module = importlib.import_module(module_name)
            if hasattr(module, 'get_tools'):
                all_tools.extend(module.get_tools())
                logger.debug(f"加载工具模块 {module_name}: {len(module.get_tools())} 个工具")
        except ImportError as e:
            logger.warning(f"加载工具模块 {module_name} 失败: {e}")

    return all_tools


def get_all_tools() -> list:
    """获取所有可用工具"""
    return discover_tools()
