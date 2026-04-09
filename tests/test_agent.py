# tests/test_agent.py
"""
Agent 工作流测试 — 验证 LangGraph 节点路由逻辑
Mock LLM 调用，不需要 API Key
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from unittest.mock import MagicMock, patch, PropertyMock


# ==================== 路由函数测试 ====================

class TestRetrievalEvalRouting:
    """测试检索评估后的路由逻辑"""

    def _import_route(self):
        # 从 graph 模块中获取路由函数
        from src.agent.graph import route_after_retrieval_eval
        return route_after_retrieval_eval

    def test_sufficient_goes_to_writer(self):
        """检索质量 sufficient → 走 writer"""
        route = self._import_route()
        state = {"retrieval_quality": "sufficient", "retrieval_attempts": 1, "max_retrieval_attempts": 3}
        assert route(state) == "writer"

    def test_insufficient_goes_to_researcher(self):
        """检索质量 insufficient → 回 researcher 重试"""
        route = self._import_route()
        state = {"retrieval_quality": "insufficient", "retrieval_attempts": 1, "max_retrieval_attempts": 3}
        assert route(state) == "researcher"

    def test_irrelevant_goes_to_writer(self):
        """检索质量 irrelevant → 交给 writer 告知用户"""
        route = self._import_route()
        state = {"retrieval_quality": "irrelevant", "retrieval_attempts": 1, "max_retrieval_attempts": 3}
        assert route(state) == "writer"

    def test_max_attempts_forces_writer(self):
        """超过最大尝试次数 → 强制走 writer"""
        route = self._import_route()
        state = {"retrieval_quality": "insufficient", "retrieval_attempts": 3, "max_retrieval_attempts": 3}
        assert route(state) == "writer"

    def test_default_quality_is_sufficient(self):
        """未设置 quality → 默认 sufficient"""
        route = self._import_route()
        state = {"retrieval_attempts": 0, "max_retrieval_attempts": 3}
        assert route(state) == "writer"


class TestVerificationRouting:
    """测试答案验证后的路由逻辑"""

    def _import_route(self):
        from src.agent.graph import route_after_verification
        return route_after_verification

    def test_pass_goes_to_end(self):
        """验证通过 → END"""
        route = self._import_route()
        state = {"verification_result": "pass", "verification_attempts": 1}
        assert route(state) == "end"

    def test_fail_goes_to_writer(self):
        """验证失败 → 回 writer 重写"""
        route = self._import_route()
        state = {"verification_result": "fail", "verification_attempts": 1}
        assert route(state) == "writer"

    def test_max_attempts_forces_end(self):
        """超过最大验证次数 → 强制 END"""
        route = self._import_route()
        state = {"verification_result": "fail", "verification_attempts": 2}
        assert route(state) == "end"

    def test_default_result_is_pass(self):
        """未设置 result → 默认 pass"""
        route = self._import_route()
        state = {"verification_attempts": 0}
        assert route(state) == "end"


# ==================== 节点函数测试 ====================

class TestRetrievalEvaluatorNode:
    """测试检索评估节点的状态输出"""

    def test_increments_attempts(self):
        """每次评估增加 retrieval_attempts"""
        from src.agent.nodes_eval import retrieval_evaluator_node
        tool_msg = MagicMock()
        tool_msg.type = "tool"
        tool_msg.content = "找到了相关内容" + "x" * 100
        state = {"messages": [tool_msg], "retrieval_attempts": 2}
        result = retrieval_evaluator_node(state)
        assert result["retrieval_attempts"] == 3

    def test_state_output_has_required_keys(self):
        """输出包含必需的键"""
        from src.agent.nodes_eval import retrieval_evaluator_node
        tool_msg = MagicMock()
        tool_msg.type = "tool"
        tool_msg.content = "内容" + "x" * 100
        state = {"messages": [tool_msg], "retrieval_attempts": 0}
        result = retrieval_evaluator_node(state)
        assert "retrieval_quality" in result
        assert "retrieval_attempts" in result


class TestAnswerVerifierNode:
    """测试答案验证节点的状态输出"""

    def test_output_has_required_keys(self):
        """输出包含所有必需键"""
        from src.agent.nodes_eval import answer_verifier_node
        ai_msg = MagicMock()
        ai_msg.type = "ai"
        ai_msg.tool_calls = None
        ai_msg.content = "这是一个符合要求的答案内容，长度适中，不包含矛盾表述。"
        state = {"messages": [ai_msg], "verification_attempts": 0}
        result = answer_verifier_node(state)
        assert "verification_result" in result
        assert "verification_issues" in result
        assert "confidence_score" in result
        assert "verification_attempts" in result

    def test_increments_verification_attempts(self):
        """每次验证增加 verification_attempts"""
        from src.agent.nodes_eval import answer_verifier_node
        ai_msg = MagicMock()
        ai_msg.type = "ai"
        ai_msg.tool_calls = None
        ai_msg.content = "正常答案" + "，" + "内容丰富" * 20
        state = {"messages": [ai_msg], "verification_attempts": 1}
        result = answer_verifier_node(state)
        assert result["verification_attempts"] == 2


# ==================== AgentState 结构测试 ====================

class TestAgentState:
    """测试 AgentState 结构完整性"""

    def test_state_has_all_required_fields(self):
        """AgentState 应包含所有 Sprint 3 新增字段"""
        from src.agent.state import AgentState
        annotations = AgentState.__annotations__
        required_fields = [
            "messages", "rewritten_queries", "query_type", "original_query",
            "retrieval_quality", "retrieval_attempts", "max_retrieval_attempts",
            "verification_result", "verification_issues", "confidence_score",
            "verification_attempts"
        ]
        for field in required_fields:
            assert field in annotations, f"AgentState 缺少字段: {field}"


# ==================== 工具发现测试 ====================

class TestToolDiscovery:
    """测试工具目录自动发现"""

    def test_discovers_all_tools(self):
        """应发现所有 9 个工具"""
        from src.agent.tools_dir import get_all_tools
        tools = get_all_tools()
        assert len(tools) == 9

    def test_tool_names(self):
        """所有工具应有正确的名称"""
        from src.agent.tools_dir import get_all_tools
        tools = get_all_tools()
        names = {t.name for t in tools}
        expected = {
            "ask_knowledge_base", "list_knowledge_base_files", "search_by_filename",
            "general_qa", "get_current_time", "calculate_expression",
            "summarize_text", "translate_text", "analyze_code"
        }
        assert names == expected

    def test_each_module_exports_get_tools(self):
        """每个子模块都应有 get_tools 函数"""
        import importlib
        modules = ["knowledge_base", "general", "text_processing"]
        for mod_name in modules:
            mod = importlib.import_module(f"src.agent.tools_dir.{mod_name}")
            assert hasattr(mod, "get_tools"), f"{mod_name} 缺少 get_tools()"
            tools = mod.get_tools()
            assert len(tools) > 0, f"{mod_name} 没有返回任何工具"
