# src/agent/tools.py
#python装饰器，它的作用是将一个普通的函数转换为一个工具（Tool），
# 使其可以被智能体（Agent）调用和使用。
from langchain_core.tools import tool
from langgraph.config import RunnableConfig
#导入RAG生成器
from src.rag.generator import RAGGenerator

# 初始化 RAG 引擎 (只会初始化一次)
# 这里实例化 RAGGenerator，连接数据库和 LLM
rag_engine = RAGGenerator()

# --- 3. 定义工具函数 ---
# @tool 装饰器是关键。它会自动解析函数的【名称】、【参数】和【文档注释(Docstring)】。
# Agent 就是靠阅读这些文档注释来决定“什么时候用这个工具”以及“怎么用”。
@tool
def ask_knowledge_base(query: str, config: RunnableConfig) -> str:
    """
    企业内部知识库查询工具（基于 project 知识库，session 仅用于对话历史）。
    """
    cfg = config.get("configurable", {}) or {}
    session_id = cfg.get("session_id")
    project_id = cfg.get("project_id", "default")

    return rag_engine.get_answer(query, session_id=session_id, project_id=project_id)