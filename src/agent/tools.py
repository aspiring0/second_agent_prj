# src/agent/tools.py
#python装饰器，它的作用是将一个普通的函数转换为一个工具（Tool），
# 使其可以被智能体（Agent）调用和使用。
from langchain_core.tools import tool
from langgraph.config import RunnableConfig
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
#导入RAG生成器
from src.rag.generator import RAGGenerator
from config.settings import settings
from src.utils.logger import setup_logger
import datetime
import json

logger = setup_logger("Agent_Tools")

# 初始化 RAG 引擎 (只会初始化一次)
# 这里实例化 RAGGenerator，连接数据库和 LLM
rag_engine = RAGGenerator()

# 初始化通用 LLM（用于通用问答）
general_llm = ChatOpenAI(
    model=settings.CHAT_MODEL,
    temperature=0.7,
    openai_api_key=settings.OPENAI_API_KEY,
    openai_api_base=settings.OPENAI_BASE_URL
)

# 初始化向量数据库连接（用于元数据查询）
def get_chroma_db():
    """获取 Chroma 数据库连接"""
    return Chroma(
        persist_directory=str(settings.DB_DIR),
        embedding_function=OpenAIEmbeddings(
            model=settings.EMBEDDING_MODEL,
            openai_api_key=settings.OPENAI_API_KEY,
            openai_api_base=settings.OPENAI_BASE_URL
        )
    )

# ============== 通用能力工具 ==============

@tool
def general_qa(question: str, config: RunnableConfig) -> str:
    """
    通用问答工具。用于回答常识性问题、编程问题、概念解释等不需要知识库的问题。
    当用户问题与知识库无关时使用此工具，如：
    - 编程问题："Python如何读取文件"
    - 概念解释："什么是机器学习"
    - 一般建议："如何提高编程能力"
    - 数学计算、逻辑推理等
    
    参数:
        question: 用户的问题
    """
    try:
        logger.info(f"🔧 通用问答: {question}")
        response = general_llm.invoke(question)
        return response.content
    except Exception as e:
        logger.error(f"通用问答失败: {e}")
        return f"回答问题时出错: {str(e)}"

@tool  
def summarize_text(text: str, config: RunnableConfig) -> str:
    """
    文本总结工具。将长文本总结成简洁的摘要。
    当用户要求"总结"、"概括"、"提炼要点"时使用。
    
    参数:
        text: 需要总结的文本内容
    """
    try:
        logger.info(f"🔧 文本总结，长度: {len(text)}")
        prompt = f"""请将以下文本总结成简洁的摘要，保留关键信息：

{text}

摘要："""
        response = general_llm.invoke(prompt)
        return response.content
    except Exception as e:
        logger.error(f"总结失败: {e}")
        return f"总结时出错: {str(e)}"

@tool
def translate_text(text: str, target_language: str = "中文", config: RunnableConfig = None) -> str:
    """
    翻译工具。将文本翻译成目标语言。
    
    参数:
        text: 需要翻译的文本
        target_language: 目标语言，如"中文"、"英文"、"日文"等，默认中文
    """
    try:
        logger.info(f"🔧 翻译到 {target_language}")
        prompt = f"""请将以下文本翻译成{target_language}，只输出翻译结果：

{text}"""
        response = general_llm.invoke(prompt)
        return response.content
    except Exception as e:
        logger.error(f"翻译失败: {e}")
        return f"翻译时出错: {str(e)}"

@tool
def analyze_code(code: str, language: str = "auto", config: RunnableConfig = None) -> str:
    """
    代码分析工具。分析代码的功能、潜在问题、优化建议等。
    
    参数:
        code: 需要分析的代码
        language: 编程语言，如"Python"、"JavaScript"等，默认自动检测
    """
    try:
        logger.info(f"🔧 代码分析，语言: {language}")
        prompt = f"""请分析以下{'代码' if language == 'auto' else language + '代码'}：

```
{code}
```

请从以下几个方面分析：
1. 代码功能说明
2. 潜在问题或bug
3. 优化建议
4. 代码质量评分（1-10分）

分析结果："""
        response = general_llm.invoke(prompt)
        return response.content
    except Exception as e:
        logger.error(f"代码分析失败: {e}")
        return f"分析代码时出错: {str(e)}"

@tool
def get_current_time(config: RunnableConfig) -> str:
    """
    获取当前时间工具。返回当前的日期和时间。
    当用户问"现在几点"、"今天日期"等时间相关问题时使用。
    """
    now = datetime.datetime.now()
    return f"当前时间：{now.strftime('%Y年%m月%d日 %H:%M:%S')} ({now.strftime('%A')})"

@tool
def calculate_expression(expression: str, config: RunnableConfig = None) -> str:
    """
    计算器工具。执行数学计算和表达式求值。
    支持基本运算、百分比、幂运算等。
    
    参数:
        expression: 数学表达式，如"2+3*4"、"100*0.15"、"2**10"
    """
    try:
        logger.info(f"🔧 计算: {expression}")
        # 安全的数学计算
        allowed_chars = set("0123456789+-*/.()% ")
        if not all(c in allowed_chars for c in expression):
            # 使用 LLM 处理复杂的数学问题
            prompt = f"请计算以下数学问题，只输出数字结果：\n{expression}"
            response = general_llm.invoke(prompt)
            return response.content
        
        result = eval(expression)
        return f"计算结果：{expression} = {result}"
    except Exception as e:
        logger.error(f"计算失败: {e}")
        return f"计算时出错: {str(e)}。请检查表达式格式。"

@tool
def list_knowledge_base_files(config: RunnableConfig) -> str:
    """
    列出知识库中所有的文件名和文件类型。
    当用户问"有哪些文件"、"有什么PDF"、"知识库里有什么"时使用此工具。
    返回文件列表，帮助用户了解知识库内容。
    """
    cfg = config.get("configurable", {}) or {}
    project_id = cfg.get("project_id", "default")
    
    try:
        db = get_chroma_db()
        # 获取所有文档的元数据
        results = db.get(include=["metadatas"])
        
        if not results or not results.get("metadatas"):
            return "知识库中暂时没有任何文件。"
        
        # 统计文件来源
        files = {}
        for meta in results["metadatas"]:
            if meta.get("project_id") == project_id or project_id == "default":
                source = meta.get("source", "未知来源")
                file_type = source.split(".")[-1].upper() if "." in source else "未知"
                if source not in files:
                    files[source] = {"type": file_type, "count": 0}
                files[source]["count"] += 1
        
        if not files:
            return f"项目 {project_id} 下没有找到任何文件。"
        
        # 格式化输出
        output_lines = [f"📚 知识库中共有 {len(files)} 个文件：\n"]
        for source, info in files.items():
            output_lines.append(f"  • {source} ({info['type']} 文件, {info['count']} 个片段)")
        
        return "\n".join(output_lines)
        
    except Exception as e:
        logger.error(f"列出文件失败: {e}")
        return f"获取文件列表失败: {str(e)}"

@tool
def search_by_filename(filename: str, config: RunnableConfig) -> str:
    """
    按文件名搜索知识库内容。
    当用户提到具体文件名（如"PDF文件"、"那个py文件"、"xxx.pdf"）时使用此工具。
    会返回该文件的所有相关内容片段。
    
    参数:
        filename: 文件名或文件类型关键词，如"pdf"、"算法.pdf"、"py"
    """
    cfg = config.get("configurable", {}) or {}
    project_id = cfg.get("project_id", "default")
    
    try:
        db = get_chroma_db()
        results = db.get(include=["metadatas", "documents"])
        
        if not results or not results.get("metadatas"):
            return "知识库中没有找到任何内容。"
        
        # 筛选匹配的文件
        matched_content = []
        filename_lower = filename.lower()
        
        for i, meta in enumerate(results["metadatas"]):
            source = meta.get("source", "").lower()
            # 匹配文件名或文件类型
            if filename_lower in source or source.endswith(f".{filename_lower}"):
                doc_content = results["documents"][i] if i < len(results["documents"]) else ""
                if project_id == "default" or meta.get("project_id") == project_id:
                    matched_content.append(f"【来源: {meta.get('source')}】\n{doc_content}")
        
        if not matched_content:
            # 尝试模糊匹配文件类型
            type_hints = {
                "pdf": [".pdf"],
                "py": [".py"],
                "txt": [".txt"],
                "word": [".doc", ".docx"],
                "md": [".md"],
            }
            
            for hint_type, extensions in type_hints.items():
                if hint_type in filename_lower:
                    for i, meta in enumerate(results["metadatas"]):
                        source = meta.get("source", "")
                        for ext in extensions:
                            if source.lower().endswith(ext):
                                doc_content = results["documents"][i] if i < len(results["documents"]) else ""
                                if project_id == "default" or meta.get("project_id") == project_id:
                                    matched_content.append(f"【来源: {source}】\n{doc_content}")
                    break
        
        if not matched_content:
            return f"没有找到与 '{filename}' 相关的文件内容。\n提示：可以使用 list_knowledge_base_files 工具查看所有可用文件。"
        
        # 合并内容
        total_content = "\n\n---\n\n".join(matched_content)
        logger.info(f"按文件名 '{filename}' 搜索到 {len(matched_content)} 个片段")
        
        return f"找到 {len(matched_content)} 个与 '{filename}' 相关的内容片段：\n\n{total_content}"
        
    except Exception as e:
        logger.error(f"按文件名搜索失败: {e}")
        return f"搜索失败: {str(e)}"

@tool
def ask_knowledge_base(query: str, config: RunnableConfig) -> str:
    """
    企业内部知识库语义搜索工具。
    根据用户问题进行语义检索，返回最相关的内容。
    当用户有具体问题时使用此工具，如"Python是什么"、"如何配置环境"。
    
    参数:
        query: 用户的问题或搜索关键词
    """
    cfg = config.get("configurable", {}) or {}
    session_id = cfg.get("session_id")
    project_id = cfg.get("project_id", "default")

    return rag_engine.get_answer(query, session_id=session_id, project_id=project_id)
