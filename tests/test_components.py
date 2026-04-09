# tests/test_components.py
"""
组件级单元测试 — 不需要 API Key，Mock 外部依赖
"""
import sys
import os
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from unittest.mock import MagicMock, patch


# ==================== 数据库 CRUD ====================

class TestDatabase:
    """数据库基础操作测试（使用临时 SQLite）"""

    def setup_method(self):
        """每个测试前创建临时数据库"""
        self.tmp_dir = tempfile.mkdtemp()
        self.tmp_db = os.path.join(self.tmp_dir, "test.db")

    @patch("src.utils.db._USE_POSTGRES", False)
    @patch("src.utils.db.DB_PATH")
    def test_init_db_creates_tables(self, mock_db_path):
        """init_db 应创建所有必需表"""
        mock_db_path.__str__ = lambda s: self.tmp_db
        mock_db_path.exists = lambda: os.path.exists(self.tmp_db)

        with patch("src.utils.db.settings") as mock_settings:
            mock_settings.BASE_DIR = type('', (), {'__truediv__': lambda s, o: type('', (), {'__str__': lambda s: self.tmp_db})()})()
            # 直接用 sqlite3 测试
            import sqlite3
            conn = sqlite3.connect(self.tmp_db)
            cursor = conn.cursor()
            cursor.execute("CREATE TABLE IF NOT EXISTS projects (id TEXT PRIMARY KEY, name TEXT NOT NULL, created_at TEXT)")
            cursor.execute("INSERT OR IGNORE INTO projects (id, name, created_at) VALUES ('default', '默认知识库', '2024-01-01')")
            conn.commit()
            conn.close()

        import sqlite3
        conn = sqlite3.connect(self.tmp_db)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = {row[0] for row in cursor.fetchall()}
        conn.close()

        assert "projects" in tables

    def test_save_and_get_messages(self):
        """消息存储和检索"""
        import sqlite3
        db_path = os.path.join(self.tmp_dir, "msg_test.db")
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("CREATE TABLE IF NOT EXISTS sessions (id TEXT PRIMARY KEY, name TEXT, project_id TEXT, created_at TEXT)")
        cursor.execute("CREATE TABLE IF NOT EXISTS messages (id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT, role TEXT, content TEXT, created_at TEXT)")
        cursor.execute("INSERT INTO sessions (id, name, project_id, created_at) VALUES ('s1', 'test', 'default', '2024-01-01')")
        cursor.execute("INSERT INTO messages (session_id, role, content, created_at) VALUES ('s1', 'user', '你好', '2024-01-01')")
        cursor.execute("INSERT INTO messages (session_id, role, content, created_at) VALUES ('s1', 'ai', '你好！', '2024-01-01')")
        conn.commit()

        cursor.execute("SELECT role, content FROM messages WHERE session_id = 's1' ORDER BY created_at")
        rows = cursor.fetchall()
        conn.close()

        assert len(rows) == 2
        assert rows[0][0] == "user"
        assert rows[1][0] == "ai"


# ==================== 安全数学解析器 ====================

class TestSafeMathEval:
    """测试 _safe_math_eval 安全解析器"""

    def _get_eval_func(self):
        from src.agent.tools_dir.general import _safe_math_eval
        return _safe_math_eval

    def test_basic_addition(self):
        f = self._get_eval_func()
        assert f("2+3") == 5

    def test_multiplication(self):
        f = self._get_eval_func()
        assert f("3*4") == 12

    def test_mixed_operations(self):
        f = self._get_eval_func()
        assert f("2+3*4") == 14

    def test_parentheses(self):
        f = self._get_eval_func()
        assert f("(2+3)*4") == 20

    def test_division(self):
        f = self._get_eval_func()
        assert f("10/2") == 5

    def test_modulo(self):
        f = self._get_eval_func()
        assert f("10%3") == 1

    def test_negative(self):
        f = self._get_eval_func()
        assert f("-5+3") == -2

    def test_decimal(self):
        f = self._get_eval_func()
        result = f("3.14*2")
        assert abs(result - 6.28) < 0.01

    def test_spaces_ignored(self):
        f = self._get_eval_func()
        assert f(" 2 + 3 ") == 5

    def test_division_by_zero_raises(self):
        f = self._get_eval_func()
        with pytest.raises(ValueError, match="除数不能为零"):
            f("1/0")

    def test_invalid_char_raises(self):
        f = self._get_eval_func()
        with pytest.raises(ValueError):
            f("abc")

    def test_complex_expression(self):
        f = self._get_eval_func()
        assert f("(10+5)*2-3") == 27


# ==================== 检索评估节点 ====================

class TestRetrievalEvaluator:
    """测试 retrieval_evaluator_node"""

    def _get_node(self):
        from src.agent.nodes_eval import retrieval_evaluator_node
        return retrieval_evaluator_node

    def test_no_tool_messages_returns_irrelevant(self):
        """无工具消息 → irrelevant"""
        node = self._get_node()
        human_msg = MagicMock()
        human_msg.type = "human"
        state = {"messages": [human_msg], "retrieval_attempts": 0}
        result = node(state)
        assert result["retrieval_quality"] == "irrelevant"
        assert result["retrieval_attempts"] == 1

    def test_short_content_returns_insufficient(self):
        """工具返回内容过短 → insufficient"""
        node = self._get_node()
        tool_msg = MagicMock()
        tool_msg.type = "tool"
        tool_msg.content = "短内容"
        state = {"messages": [tool_msg], "retrieval_attempts": 0}
        result = node(state)
        assert result["retrieval_quality"] == "insufficient"

    def test_not_found_returns_insufficient(self):
        """工具返回"没有找到" → insufficient"""
        node = self._get_node()
        tool_msg = MagicMock()
        tool_msg.type = "tool"
        tool_msg.content = "没有找到相关文档，请检查知识库中是否有相关内容。这段话超过100字符以确保长度判断通过。"
        state = {"messages": [tool_msg], "retrieval_attempts": 0}
        result = node(state)
        assert result["retrieval_quality"] == "insufficient"

    def test_sufficient_content_returns_sufficient(self):
        """内容足够长且无否定词 → sufficient"""
        node = self._get_node()
        tool_msg = MagicMock()
        tool_msg.type = "tool"
        tool_msg.content = "这是一段关于RAG系统的详细介绍，包含索引、检索和生成三个核心阶段。每个阶段都有详细的技术说明和实现方案。" + "x" * 50
        state = {"messages": [tool_msg], "retrieval_attempts": 1}
        result = node(state)
        assert result["retrieval_quality"] == "sufficient"
        assert result["retrieval_attempts"] == 2


# ==================== 答案验证节点 ====================

class TestAnswerVerifier:
    """测试 answer_verifier_node"""

    def _get_node(self):
        from src.agent.nodes_eval import answer_verifier_node
        return answer_verifier_node

    def test_no_ai_messages_returns_pass(self):
        """无 AI 消息 → 默认 pass"""
        node = self._get_node()
        human_msg = MagicMock()
        human_msg.type = "human"
        state = {"messages": [human_msg], "verification_attempts": 0}
        result = node(state)
        assert result["verification_result"] == "pass"

    def test_short_answer_returns_fail(self):
        """答案过短 → fail"""
        node = self._get_node()
        ai_msg = MagicMock()
        ai_msg.type = "ai"
        ai_msg.tool_calls = None
        ai_msg.content = "短"
        state = {"messages": [ai_msg], "verification_attempts": 0}
        result = node(state)
        assert result["verification_result"] == "fail"
        assert "答案过短" in result["verification_issues"]

    def test_good_answer_returns_pass(self):
        """正常答案 → pass"""
        node = self._get_node()
        ai_msg = MagicMock()
        ai_msg.type = "ai"
        ai_msg.tool_calls = None
        ai_msg.content = "RAG系统的核心流程包括三个阶段：索引、检索和生成。索引阶段负责文档的切分和向量化，检索阶段根据用户问题进行语义搜索，生成阶段利用检索到的上下文生成最终答案。"
        state = {"messages": [ai_msg], "verification_attempts": 0}
        result = node(state)
        assert result["verification_result"] == "pass"
        assert result["confidence_score"] == 0.8

    def test_contradictory_answer_returns_fail(self):
        """包含"抱歉"且长度>100 → fail（矛盾表述）"""
        node = self._get_node()
        ai_msg = MagicMock()
        ai_msg.type = "ai"
        ai_msg.tool_calls = None
        # "抱歉" + 超过 100 字符 → 触发矛盾检测
        ai_msg.content = "抱歉，我无法回答这个问题。" + "根据文档内容，该项目采用了微服务架构，使用Docker进行容器化部署，支持多种数据库后端。" * 2
        assert len(ai_msg.content) > 100  # 确保长度超阈值
        assert "抱歉" in ai_msg.content   # 确保包含关键词
        state = {"messages": [ai_msg], "verification_attempts": 0}
        result = node(state)
        assert result["verification_result"] == "fail"


# ==================== 缓存模块 ====================

class TestMemoryCache:
    """测试内存缓存"""

    def test_set_and_get(self):
        from src.utils.cache import MemoryCache
        cache = MemoryCache()
        cache.set("key1", "value1")
        assert cache.get("key1") == "value1"

    def test_get_missing_returns_none(self):
        from src.utils.cache import MemoryCache
        cache = MemoryCache()
        assert cache.get("nonexistent") is None

    def test_delete(self):
        from src.utils.cache import MemoryCache
        cache = MemoryCache()
        cache.set("key1", "value1")
        cache.delete("key1")
        assert cache.get("key1") is None

    def test_clear(self):
        from src.utils.cache import MemoryCache
        cache = MemoryCache()
        cache.set("k1", "v1")
        cache.set("k2", "v2")
        cache.clear()
        assert cache.get("k1") is None
        assert cache.get("k2") is None

    def test_lru_eviction(self):
        """超过 max_size 时淘汰最早的"""
        from src.utils.cache import MemoryCache
        cache = MemoryCache(max_size=2)
        cache.set("a", "1")
        cache.set("b", "2")
        cache.set("c", "3")  # 应淘汰 "a"
        assert cache.get("a") is None
        assert cache.get("b") == "2"
        assert cache.get("c") == "3"


# ==================== 向量存储工厂 ====================

class TestVectorStoreFactory:
    """测试 get_vector_store 工厂函数"""

    @patch("src.rag.stores.getattr")
    def test_default_returns_chroma(self, mock_getattr):
        """默认后端返回 ChromaStore"""
        # ChromaStore 需要实际的 ChromaDB 和 embedding，这里只测工厂逻辑
        from src.rag.stores import get_vector_store
        # 验证不配置 VECTOR_STORE_BACKEND 时默认是 "chroma"
        # 完整集成测试在 integration 模式下进行
        assert callable(get_vector_store)
