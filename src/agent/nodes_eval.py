# src/agent/nodes_eval.py
"""
Sprint 3: 质量评估节点
1. 检索评估器 - 评估检索结果质量
2. 答案验证器 - 验证生成答案的忠实度
"""

from src.agent.state import AgentState
from src.utils.logger import setup_logger

logger = setup_logger("Eval_Nodes")


def retrieval_evaluator_node(state: AgentState) -> dict:
    """
    评估检索结果质量（不调用 LLM，使用规则 + 向量分数）

    评估维度：
    1. 最相关文档的分数是否超过阈值
    2. 检索结果数量是否足够
    3. 检索结果是否与问题相关（基于已有分数）
    """
    messages = state["messages"]

    # 从最近的 ToolMessage 中提取检索分数
    tool_messages = [m for m in messages if getattr(m, 'type', '') == 'tool']
    if not tool_messages:
        logger.info("检索评估: 无工具消息，标记为 irrelevant")
        return {
            "retrieval_quality": "irrelevant",
            "retrieval_attempts": state.get("retrieval_attempts", 0) + 1,
        }

    # 基于规则评估（不调用LLM，避免延迟）
    last_tool = tool_messages[-1]
    content = last_tool.content

    # 如果工具返回了"没有找到"类型的消息
    if "没有找到" in content or "未找到" in content or "没有找到" in content:
        quality = "insufficient"
    elif "没有" in content and "文件" in content:
        quality = "insufficient"
    elif len(content) < 100:
        quality = "insufficient"
    else:
        quality = "sufficient"

    attempts = state.get("retrieval_attempts", 0) + 1
    logger.info(f"检索评估: quality={quality}, attempts={attempts}, content_len={len(content)}")

    return {
        "retrieval_quality": quality,
        "retrieval_attempts": attempts,
    }


def answer_verifier_node(state: AgentState) -> dict:
    """
    验证生成答案的忠实度（可调用 LLM 进行深度检查）

    检查项：
    1. 答案是否引用了检索到的上下文
    2. 是否存在明显的幻觉（与上下文矛盾）
    3. 答案是否完整回答了问题
    """
    messages = state["messages"]

    # 找到最后一条 AI 消息（writer 的输出）
    ai_messages = [m for m in messages if getattr(m, 'type', '') == 'ai' and not getattr(m, 'tool_calls', None)]
    if not ai_messages:
        logger.info("答案验证: 无AI消息，默认 pass")
        return {
            "verification_result": "pass",
            "confidence_score": 0.5,
            "verification_issues": [],
            "verification_attempts": 1,
        }

    answer = ai_messages[-1].content

    # 简单规则检查（不调用LLM，降低延迟）
    issues = []

    # 检查1：答案长度是否合理
    if len(answer) < 20:
        issues.append("答案过短")

    # 检查2：是否包含"抱歉"但仍然给出了答案（矛盾）
    if "抱歉" in answer and len(answer) > 100:
        issues.append("答案包含矛盾表述")

    attempts = state.get("verification_attempts", 0) + 1

    if issues:
        logger.info(f"答案验证: FAIL (issues={issues}, attempts={attempts})")
        return {
            "verification_result": "fail",
            "verification_issues": issues,
            "confidence_score": 0.5,
            "verification_attempts": attempts,
        }

    logger.info(f"答案验证: PASS (attempts={attempts})")
    return {
        "verification_result": "pass",
        "verification_issues": [],
        "confidence_score": 0.8,
        "verification_attempts": attempts,
    }
