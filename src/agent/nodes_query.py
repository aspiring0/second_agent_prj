# src/agent/nodes_query.py
"""
查询改写节点 - HyDE 假设性答案 & Multi-Query 多角度改写
通过改写用户查询提升检索召回率
"""

from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser

from src.agent.state import AgentState
from src.utils.logger import setup_logger
from src.utils.model_manager import model_manager
from config.settings import settings

logger = setup_logger("Query_Rewriter")

# 简单查询的关键词列表（这些查询不需要改写）
TRIVIAL_PATTERNS = ["几点", "时间", "日期", "今天", "你好", "谢谢", "再见", "hi", "hello"]


def _is_trivial_query(query: str) -> bool:
    """
    判断是否为简单/闲聊类查询，不需要改写

    Args:
        query: 用户查询文本

    Returns:
        True 表示是简单查询，不需要改写
    """
    query_lower = query.strip().lower()
    if len(query_lower) < 5:
        return True
    for pattern in TRIVIAL_PATTERNS:
        if pattern in query_lower:
            return True
    return False


def _hyde_rewrite(llm, query: str) -> str:
    """
    HyDE（Hypothetical Document Embeddings）改写策略
    让 LLM 生成假设性答案，用该答案替代原查询进行检索

    Args:
        llm: 语言模型实例
        query: 用户查询

    Returns:
        改写后的查询文本（假设性答案）
    """
    prompt = (
        "请回答以下问题。你的回答将被用于语义检索，"
        "所以请写一个详细、信息丰富的答案，"
        "即使你不确定也要基于常识给出合理回答。\n\n"
        f"问题：{query}\n\n答案："
    )
    chain = llm | StrOutputParser()
    result = chain.invoke(prompt)
    logger.info(f"HyDE 改写完成，原始查询: {query[:50]}... → 改写后: {result[:80]}...")
    return result


def _multi_query_rewrite(llm, query: str) -> str:
    """
    Multi-Query 多角度改写策略
    生成多个不同角度的查询，合并后用于检索

    Args:
        llm: 语言模型实例
        query: 用户查询

    Returns:
        合并后的多角度查询文本
    """
    prompt = (
        "你是一个AI助手。请针对用户的提问，生成3个不同角度的搜索查询，"
        "用于从知识库中检索相关信息。每行一个查询，不要编号。\n\n"
        f"原始问题：{query}\n\n改写后的查询："
    )
    chain = llm | StrOutputParser()
    result = chain.invoke(prompt)
    logger.info(f"Multi-Query 改写完成: {result[:100]}...")
    return result


def query_rewriter_node(state: AgentState) -> AgentState:
    """
    查询改写节点

    策略选择基于配置和查询类型：
    - 简单/闲聊查询 → 直接透传，不做改写
    - hyde 策略 → HyDE（生成假设性答案用于检索）
    - multi 策略 → Multi-Query（多角度改写）
    - auto 策略 → 根据查询复杂度自动选择

    Args:
        state: 当前智能体状态

    Returns:
        更新后的状态（包含改写后的查询）
    """
    # 检查是否启用查询改写
    if not settings.ENABLE_QUERY_REWRITE:
        logger.info("查询改写已禁用，直接透传")
        return {"messages": []}

    # 获取用户查询
    messages = state.get("messages", [])
    if not messages:
        return {"messages": []}

    query = messages[-1].content if hasattr(messages[-1], "content") else str(messages[-1])

    # 简单查询直接透传
    if _is_trivial_query(query):
        logger.info(f"简单查询，直接透传: {query}")
        return {"messages": []}

    # 获取 LLM 实例
    llm = model_manager.get_chat_model(temperature=0.0)

    # 选择改写策略
    strategy = settings.QUERY_REWRITE_STRATEGY

    if strategy == "hyde":
        rewritten = _hyde_rewrite(llm, query)
    elif strategy == "multi":
        rewritten = _multi_query_rewrite(llm, query)
    elif strategy == "auto":
        # auto 模式：根据查询长度和复杂度选择策略
        if len(query) > 30 or "和" in query or "与" in query or "以及" in query:
            rewritten = _multi_query_rewrite(llm, query)
        else:
            rewritten = _hyde_rewrite(llm, query)
    else:
        logger.warning(f"未知改写策略: {strategy}，直接透传")
        return {"messages": []}

    return {
        "messages": [HumanMessage(content=rewritten)],
        "rewritten_queries": [rewritten],
    }
