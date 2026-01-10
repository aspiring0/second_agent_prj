# src/agent/tools.py
#python装饰器，它的作用是将一个普通的函数转换为一个工具（Tool），
# 使其可以被智能体（Agent）调用和使用。
from langchain_core.tools import tool

#导入RAG生成器
from src.rag.generator import RAGGenerator

# 初始化 RAG 引擎 (只会初始化一次)
# 这里实例化 RAGGenerator，连接数据库和 LLM
rag_engine = RAGGenerator()

# --- 3. 定义工具函数 ---
# @tool 装饰器是关键。它会自动解析函数的【名称】、【参数】和【文档注释(Docstring)】。
# Agent 就是靠阅读这些文档注释来决定“什么时候用这个工具”以及“怎么用”。
@tool
def ask_knowledge_base(query: str) -> str:
    """
    这是一个企业内部知识库查询工具。
    当用户询问关于【公司制度】、【技术文档】、【Python介绍】等
    需要依靠事实依据的问题时，必须使用此工具。
    不要用于回答“你好”等闲聊问题。
    
    Args:
        query: 用户的具体问题字符串
    """
    # 这里直接调用我们 Step 4 写的核心逻辑
    return rag_engine.get_answer(query)