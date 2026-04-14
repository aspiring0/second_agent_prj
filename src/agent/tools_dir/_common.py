# src/agent/tools_dir/_common.py
"""工具模块共享状态 — 延迟初始化，支持 UI 动态更新 API Key"""

from config.settings import settings
from src.utils.logger import setup_logger

logger = setup_logger("Agent_Tools")

# 懒加载单例
_rag_engine = None
_general_llm = None


def get_rag_engine():
    """延迟获取 RAG 引擎（每次创建新实例，确保用最新 Key）"""
    from src.rag.generator import RAGGenerator
    return RAGGenerator()


def get_general_llm():
    """延迟获取通用 LLM（每次从 model_manager 获取，确保用最新 Key）"""
    from src.utils.model_manager import model_manager
    return model_manager.get_chat_model(temperature=0.7)


def get_chroma_db():
    """获取向量数据库连接（通过抽象层）"""
    from src.rag.stores import get_vector_store
    store = get_vector_store()
    if hasattr(store, 'raw_client'):
        return store.raw_client
    return store
