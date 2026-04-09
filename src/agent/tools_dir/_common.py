# src/agent/tools_dir/_common.py
"""工具模块共享状态"""

from langchain_openai import ChatOpenAI
from src.rag.generator import RAGGenerator
from config.settings import settings
from src.utils.logger import setup_logger

logger = setup_logger("Agent_Tools")

# RAG 引擎（全局单例）
rag_engine = RAGGenerator()

# 通用 LLM（用于通用问答和文本处理）
general_llm = ChatOpenAI(
    model=settings.CHAT_MODEL,
    temperature=0.7,
    openai_api_key=settings.OPENAI_API_KEY,
    openai_api_base=settings.OPENAI_BASE_URL
)


def get_chroma_db():
    """获取向量数据库连接（通过抽象层）"""
    from src.rag.stores import get_vector_store
    store = get_vector_store()
    if hasattr(store, 'raw_client'):
        return store.raw_client
    return store
