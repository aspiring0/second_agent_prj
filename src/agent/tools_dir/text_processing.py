# src/agent/tools_dir/text_processing.py
"""文本处理工具：summarize_text, translate_text, analyze_code"""

from langchain_core.tools import tool, BaseTool
from langgraph.config import RunnableConfig
from typing import List

from ._common import logger, get_general_llm


@tool
def summarize_text(text: str, config: RunnableConfig) -> str:
    """
    文本总结工具。将长文本总结成简洁的摘要。
    当用户要求"总结"、"概括"、"提炼要点"时使用。

    参数:
        text: 需要总结的文本内容
    """
    try:
        logger.info(f"文本总结，长度: {len(text)}")
        prompt = f"""请将以下文本总结成简洁的摘要，保留关键信息：

{text}

摘要："""
        response = get_general_llm().invoke(prompt)
        return response.content
    except Exception as e:
        logger.error(f"总结失败: {e}")
        return f"总结时出错: {str(e)}"


@tool
def translate_text(text: str, target_language: str = "中文", config: RunnableConfig = None) -> str:
    """
    翻译工具。将文本翻译成目标语言。

    参数:
        text: 需要翻译的文本
        target_language: 目标语言，如"中文"、"英文"、"日文"等，默认中文
    """
    try:
        logger.info(f"翻译到 {target_language}")
        prompt = f"""请将以下文本翻译成{target_language}，只输出翻译结果：

{text}"""
        response = get_general_llm().invoke(prompt)
        return response.content
    except Exception as e:
        logger.error(f"翻译失败: {e}")
        return f"翻译时出错: {str(e)}"


@tool
def analyze_code(code: str, language: str = "auto", config: RunnableConfig = None) -> str:
    """
    代码分析工具。分析代码的功能、潜在问题、优化建议等。

    参数:
        code: 需要分析的代码
        language: 编程语言，如"Python"、"JavaScript"等，默认自动检测
    """
    try:
        logger.info(f"代码分析，语言: {language}")
        prompt = f"""请分析以下{'代码' if language == 'auto' else language + '代码'}：

```
{code}
```

请从以下几个方面分析：
1. 代码功能说明
2. 潜在问题或bug
3. 优化建议
4. 代码质量评分（1-10分）

分析结果："""
        response = get_general_llm().invoke(prompt)
        return response.content
    except Exception as e:
        logger.error(f"代码分析失败: {e}")
        return f"分析代码时出错: {str(e)}"


def get_tools() -> List[BaseTool]:
    """返回文本处理工具"""
    return [summarize_text, translate_text, analyze_code]
