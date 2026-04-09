# src/agent/state.py
"""
智能体状态定义 - Sprint 3 增强版
定义智能体之间传递的数据结构
"""

import operator
from typing import Annotated, List, TypedDict
from langchain_core.messages import BaseMessage


class AgentState(TypedDict):
    """
    智能体之间传递的数据结构
    Annotated[List[BaseMessage], operator.add] 的意思是：
    当有新消息来的时候，不是覆盖旧消息，而是追加 (add) 到列表后面。
    """
    messages: Annotated[List[BaseMessage], operator.add]

    # Sprint 1: 查询改写相关字段
    rewritten_queries: List[str]
    """改写后的查询列表"""

    query_type: str
    """查询类型：simple | complex | comparative | multi_hop"""

    original_query: str
    """原始用户查询（不变）"""

    # Sprint 3: 检索状态（由 researcher_node / retrieval_evaluator_node 填充）
    retrieval_quality: str
    """检索质量：sufficient | insufficient | irrelevant"""

    retrieval_attempts: int
    """检索尝试次数"""

    max_retrieval_attempts: int
    """最大检索次数（防死循环）"""

    # Sprint 3: 生成状态（由 answer_verifier_node 填充）
    verification_result: str
    """验证结果：pass | fail"""

    verification_issues: List[str]
    """审核发现的问题"""

    confidence_score: float
    """置信度 0-1"""

    verification_attempts: int
    """验证尝试次数"""

    # 兼容旧代码
    next_step: str
