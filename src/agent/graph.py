# src/agent/graph.py
"""
LangGraph workflow definition.
Sprint 3: 5-node workflow with evaluation loops.

START → query_rewriter → researcher → [tools] → researcher → retrieval_evaluator
                              ↑                       ↓
                              └── insufficient ───────┘
                                              sufficient
                                                  ↓
                                        writer → answer_verifier → END
                                                    ↓ fail
                                              writer（重写，最多2次）
"""

from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

from config.settings import settings
from src.agent.state import AgentState
from src.agent.nodes import researcher_node, writer_node, all_tools
from src.agent.nodes_eval import retrieval_evaluator_node, answer_verifier_node
from src.utils.logger import setup_logger

logger = setup_logger("Agent_Graph")

# 懒加载 query_rewriter（Sprint 1 的模块，可能未启用）
if settings.ENABLE_QUERY_REWRITE:
    from src.agent.nodes_query import query_rewriter_node

# 1. 创建工作流
workflow = StateGraph(AgentState)

# 2. 添加节点
# 状态初始化节点 — 设置所有字段默认值，防止 None 引发异常
def initializer_node(state: AgentState) -> dict:
    """初始化 AgentState 的默认值"""
    return {
        "retrieval_quality": "sufficient",
        "retrieval_attempts": 0,
        "max_retrieval_attempts": 3,
        "verification_result": "pass",
        "verification_issues": [],
        "confidence_score": 0.0,
        "verification_attempts": 0,
        "rewritten_queries": [],
        "query_type": "simple",
        "original_query": "",
        "next_step": "",
    }

workflow.add_node("initializer", initializer_node)

if settings.ENABLE_QUERY_REWRITE:
    workflow.add_node("query_rewriter", query_rewriter_node)

workflow.add_node("researcher", researcher_node)
workflow.add_node("writer", writer_node)
workflow.add_node("tools", ToolNode(all_tools))
workflow.add_node("retrieval_evaluator", retrieval_evaluator_node)
workflow.add_node("answer_verifier", answer_verifier_node)

# 3. 入口点 — 统一从 initializer 开始
workflow.set_entry_point("initializer")
workflow.add_edge("initializer", "query_rewriter" if settings.ENABLE_QUERY_REWRITE else "researcher")

if settings.ENABLE_QUERY_REWRITE:
    workflow.add_edge("query_rewriter", "researcher")

# 4. researcher → tools 或 writer
def should_continue(state: AgentState):
    """researcher 调用工具还是直接到 writer"""
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tools"
    return "writer"

workflow.add_conditional_edges(
    "researcher",
    should_continue,
    {
        "tools": "tools",
        "writer": "writer"
    }
)

# tools → retrieval_evaluator（Sprint 3: 检索后评估）
workflow.add_edge("tools", "retrieval_evaluator")

# 5. retrieval_evaluator → researcher（重试）或 writer
def route_after_retrieval_eval(state: AgentState) -> str:
    """检索评估后的路由"""
    quality = state.get("retrieval_quality", "sufficient")
    attempts = state.get("retrieval_attempts", 0)
    max_attempts = state.get("max_retrieval_attempts", 3)

    if quality == "sufficient" or attempts >= max_attempts:
        logger.info(f"检索评估路由: → writer (quality={quality}, attempts={attempts})")
        return "writer"
    elif quality == "irrelevant":
        logger.info(f"检索评估路由: → writer (irrelevant, 交给writer告知用户)")
        return "writer"
    else:
        logger.info(f"检索评估路由: → researcher (insufficient, 重试 {attempts}/{max_attempts})")
        return "researcher"

workflow.add_conditional_edges(
    "retrieval_evaluator",
    route_after_retrieval_eval,
    {
        "researcher": "researcher",
        "writer": "writer"
    }
)

# 6. writer → answer_verifier（Sprint 3: 答案验证）
workflow.add_edge("writer", "answer_verifier")

# 7. answer_verifier → END 或 writer（重写）
def route_after_verification(state: AgentState) -> str:
    """答案验证后的路由"""
    result = state.get("verification_result", "pass")
    attempts = state.get("verification_attempts", 0)
    if result == "pass" or attempts >= 2:
        logger.info(f"答案验证路由: → END (result={result}, attempts={attempts})")
        return "end"
    logger.info(f"答案验证路由: → writer (重写, attempts={attempts})")
    return "writer"

workflow.add_conditional_edges(
    "answer_verifier",
    route_after_verification,
    {
        "end": END,
        "writer": "writer"
    }
)

# 8. 编译
app = workflow.compile()
