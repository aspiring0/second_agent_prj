# src/agent/tools_dir/general.py
"""通用工具：general_qa, get_current_time, calculate_expression"""

import datetime
from langchain_core.tools import tool, BaseTool
from langgraph.config import RunnableConfig
from typing import List

from ._common import logger, get_general_llm


@tool
def general_qa(question: str, config: RunnableConfig) -> str:
    """
    通用问答工具 - 处理不需要知识库的问题

    【核心功能】
    使用大模型的通用知识回答各类问题，不依赖知识库文档。

    【适用场景】
    - 编程问题：代码语法、框架使用、调试技巧
    - 概念解释：技术概念、术语解释、原理说明
    - 一般建议：学习路径、最佳实践、方案选择
    - 逻辑推理：数学问题、逻辑分析、因果关系
    - 创意生成：文案撰写、头脑风暴、方案设计

    参数:
        question: 用户的完整问题，保持原意传递
    """
    try:
        logger.info(f"通用问答: {question}")
        response = get_general_llm().invoke(question)
        return response.content
    except Exception as e:
        logger.error(f"通用问答失败: {e}")
        return f"回答问题时出错: {str(e)}"


@tool
def get_current_time(config: RunnableConfig) -> str:
    """
    获取当前时间工具。返回当前的日期和时间。
    当用户问"现在几点"、"今天日期"等时间相关问题时使用。
    """
    now = datetime.datetime.now()
    return f"当前时间：{now.strftime('%Y年%m月%d日 %H:%M:%S')} ({now.strftime('%A')})"


@tool
def calculate_expression(expression: str, config: RunnableConfig = None) -> str:
    """
    计算器工具。执行数学计算和表达式求值。
    支持基本运算、百分比等。

    参数:
        expression: 数学表达式，如"2+3*4"、"100*0.15"、"(10+5)*2"
    """
    try:
        logger.info(f"计算: {expression}")
        allowed_chars = set("0123456789+-*/.()% ")
        if not all(c in allowed_chars for c in expression):
            logger.info("表达式包含非法字符，交给 LLM 处理")
            prompt = f"请计算以下数学问题，只输出数字结果：\n{expression}"
            response = get_general_llm().invoke(prompt)
            return response.content

        result = _safe_math_eval(expression)
        return f"计算结果：{expression} = {result}"
    except Exception as e:
        logger.error(f"计算失败: {e}")
        return f"计算时出错: {str(e)}。请检查表达式格式。"


def _safe_math_eval(expr: str):
    """
    安全的数学表达式解析器
    仅支持数字和 + - * / . % ( ) 和空格，不使用 eval()
    使用递归下降解析器实现
    """
    tokens = _tokenize_math(expr)
    pos = [0]

    def peek():
        if pos[0] < len(tokens):
            return tokens[pos[0]]
        return None

    def consume():
        tok = tokens[pos[0]]
        pos[0] += 1
        return tok

    def parse_number():
        tok = consume()
        try:
            return float(tok)
        except ValueError:
            raise ValueError(f"无效的数字: {tok}")

    def parse_factor():
        tok = peek()
        if tok is None:
            raise ValueError("表达式不完整")
        if tok == '(':
            consume()
            val = parse_expr()
            if peek() != ')':
                raise ValueError("缺少右括号")
            consume()
            return val
        elif tok == '-':
            consume()
            return -parse_factor()
        elif tok == '+':
            consume()
            return parse_factor()
        else:
            return parse_number()

    def parse_term():
        left = parse_factor()
        while peek() in ('*', '/', '%'):
            op = consume()
            right = parse_factor()
            if op == '*':
                left = left * right
            elif op == '/':
                if right == 0:
                    raise ValueError("除数不能为零")
                left = left / right
            elif op == '%':
                left = left % right
        return left

    def parse_expr():
        left = parse_term()
        while peek() in ('+', '-'):
            op = consume()
            right = parse_term()
            if op == '+':
                left = left + right
            else:
                left = left - right
        return left

    result = parse_expr()
    if pos[0] != len(tokens):
        raise ValueError("表达式格式错误，存在多余字符")
    if isinstance(result, float) and result == int(result):
        return int(result)
    return result


def _tokenize_math(expr: str):
    """将数学表达式分词为 token 列表"""
    tokens = []
    i = 0
    while i < len(expr):
        ch = expr[i]
        if ch.isspace():
            i += 1
            continue
        if ch in '+-*/%()':
            tokens.append(ch)
            i += 1
        elif ch.isdigit() or ch == '.':
            start = i
            while i < len(expr) and (expr[i].isdigit() or expr[i] == '.'):
                i += 1
            tokens.append(expr[start:i])
        else:
            raise ValueError(f"非法字符: '{ch}'")
    return tokens


def get_tools() -> List[BaseTool]:
    """返回通用工具"""
    return [general_qa, get_current_time, calculate_expression]
