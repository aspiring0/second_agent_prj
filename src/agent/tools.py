# src/agent/tools.py
#python装饰器，它的作用是将一个普通的函数转换为一个工具（Tool），
# 使其可以被智能体（Agent）调用和使用。
from langchain_core.tools import tool
from langgraph.config import RunnableConfig
from langchain_openai import ChatOpenAI
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
    """获取向量数据库连接（通过抽象层）"""
    from src.rag.stores import get_vector_store
    store = get_vector_store()
    if hasattr(store, 'raw_client'):
        return store.raw_client
    return store

# ============== 通用能力工具 ==============

@tool
def general_qa(question: str, config: RunnableConfig) -> str:
    """
    通用问答工具 - 处理不需要知识库的问题
    
    【核心功能】
    使用大模型的通用知识回答各类问题，不依赖知识库文档。
    
    【适用场景】
    - 编程问题：代码语法、框架使用、调试技巧
    - 概念解释：技术概念、术语解释、原理说明
    - 一般建议：学习路径、最佳实践、方案选择
    - 逻辑推理：数学问题、逻辑分析、因果关系
    - 创意生成：文案撰写、头脑风暴、方案设计
    
    【不适用场景】
    - 需要查询用户上传的文档 → 使用 ask_knowledge_base
    - 需要查找特定文件 → 使用 search_by_filename
    - 涉及企业内部知识 → 使用知识库相关工具
    
    【使用示例】
    - 用户问："Python如何读取JSON文件？" → 调用此工具
    - 用户问："什么是RAG技术？" → 调用此工具
    - 用户问："这个项目的架构是什么？" → 使用 ask_knowledge_base
    
    参数:
        question: 用户的完整问题，保持原意传递
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
    支持基本运算、百分比等。

    参数:
        expression: 数学表达式，如"2+3*4"、"100*0.15"、"(10+5)*2"
    """
    try:
        logger.info(f"🔧 计算: {expression}")
        # 安全检查：只允许数字、运算符和括号
        allowed_chars = set("0123456789+-*/.()% ")
        if not all(c in allowed_chars for c in expression):
            # 使用 LLM 处理复杂的数学问题
            logger.info("表达式包含非法字符，交给 LLM 处理")
            prompt = f"请计算以下数学问题，只输出数字结果：\n{expression}"
            response = general_llm.invoke(prompt)
            return response.content

        # 安全的数学解析器：逐字符解析，支持 + - * / % ( )
        result = _safe_math_eval(expression)
        return f"计算结果：{expression} = {result}"
    except Exception as e:
        logger.error(f"计算失败: {e}")
        return f"计算时出错: {str(e)}。请检查表达式格式。"


def _safe_math_eval(expr: str):
    """
    安全的数学表达式解析器
    仅支持数字和 + - * / . % ( ) 和空格，不使用 eval()
    使用递归下降解析器实现
    """
    import operator as _op

    # 预处理：去除空格
    tokens = _tokenize_math(expr)
    pos = [0]  # 使用列表包装以便在闭包中修改

    def peek():
        if pos[0] < len(tokens):
            return tokens[pos[0]]
        return None

    def consume():
        tok = tokens[pos[0]]
        pos[0] += 1
        return tok

    def parse_number():
        """解析数字（整数或小数）"""
        tok = consume()
        try:
            return float(tok)
        except ValueError:
            raise ValueError(f"无效的数字: {tok}")

    def parse_factor():
        """解析因子：数字 或 (表达式) 或 一元正负号"""
        tok = peek()
        if tok is None:
            raise ValueError("表达式不完整")
        if tok == '(':
            consume()  # 吃掉 '('
            val = parse_expr()
            if peek() != ')':
                raise ValueError("缺少右括号")
            consume()  # 吃掉 ')'
            return val
        elif tok == '-':
            consume()
            return -parse_factor()
        elif tok == '+':
            consume()
            return parse_factor()
        else:
            return parse_number()

    def parse_term():
        """解析项：处理 * / % 运算"""
        left = parse_factor()
        while peek() in ('*', '/', '%'):
            op = consume()
            right = parse_factor()
            if op == '*':
                left = left * right
            elif op == '/':
                if right == 0:
                    raise ValueError("除数不能为零")
                left = left / right
            elif op == '%':
                left = left % right
        return left

    def parse_expr():
        """解析表达式：处理 + - 运算"""
        left = parse_term()
        while peek() in ('+', '-'):
            op = consume()
            right = parse_term()
            if op == '+':
                left = left + right
            else:
                left = left - right
        return left

    result = parse_expr()
    if pos[0] != len(tokens):
        raise ValueError("表达式格式错误，存在多余字符")
    # 如果结果是整数，返回整数形式
    if isinstance(result, float) and result == int(result):
        return int(result)
    return result


def _tokenize_math(expr: str):
    """
    将数学表达式分词为 token 列表
    支持：数字（含小数）、运算符 + - * / %、括号 ( )
    """
    tokens = []
    i = 0
    while i < len(expr):
        ch = expr[i]
        if ch.isspace():
            i += 1
            continue
        if ch in '+-*/%()':
            tokens.append(ch)
            i += 1
        elif ch.isdigit() or ch == '.':
            # 收集完整数字（含小数点）
            start = i
            while i < len(expr) and (expr[i].isdigit() or expr[i] == '.'):
                i += 1
            tokens.append(expr[start:i])
        else:
            raise ValueError(f"非法字符: '{ch}'")
    return tokens

@tool
def list_knowledge_base_files(config: RunnableConfig) -> str:
    """
    知识库文件列表工具 - 查看知识库中有哪些文件
    
    【核心功能】
    列出当前知识库中所有已上传的文件，包括文件名、类型和片段数量。
    
    【适用场景】
    - 用户想了解知识库内容："知识库里有什么？"
    - 用户不确定有哪些文件："有哪些文档？"
    - 用户想确认文件是否上传成功："我的PDF上传了吗？"
    - 作为搜索前的探索步骤
    
    【不适用场景】
    - 用户有具体问题需要回答 → 使用 ask_knowledge_base
    - 用户要查找特定文件内容 → 使用 search_by_filename
    
    【使用示例】
    - "知识库里有什么？" → 直接调用
    - "我上传了哪些文件？" → 直接调用
    - "有没有PDF文档？" → 直接调用（会显示文件类型）
    
    【返回信息】
    - 文件名列表
    - 每个文件的类型（PDF、TXT、PY等）
    - 每个文件的切片数量
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
    文件名搜索工具 - 按文件名或类型搜索知识库内容
    
    【核心功能】
    根据文件名或文件类型，从知识库中检索对应文件的全部内容。
    
    【适用场景】
    - 用户提到具体文件名："看一下test.py的内容"
    - 用户想查看某类文件："PDF文件里讲了什么"
    - 用户想找特定格式的内容："所有代码文件"
    
    【参数说明】
    filename 支持多种格式：
    - 文件类型：`pdf`、`py`、`md`、`txt`、`doc`、`docx`
    - 完整文件名：`report.pdf`、`main.py`
    - 部分文件名：`test`、`report`、`算法`
    
    【使用示例】
    - "PDF文件有哪些内容？" → search_by_filename("pdf")
    - "看一下README.md" → search_by_filename("README.md")
    - "那个算法文件" → search_by_filename("算法")
    - "所有Python代码" → search_by_filename("py")
    
    【不适用场景】
    - 用户有具体问题需要语义搜索 → 使用 ask_knowledge_base
    - 用户只想看有哪些文件 → 使用 list_knowledge_base_files
    
    【返回信息】
    - 匹配文件的内容片段
    - 每个片段的来源文件名
    - 片段总数
    
    参数:
        filename: 文件名或文件类型关键词
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
    知识库语义搜索工具 - 智能检索知识库内容
    
    【核心功能】
    使用语义理解技术，从知识库中检索与用户问题最相关的内容。
    这是最常用的知识库查询工具。
    
    【适用场景】
    - 用户有具体问题需要从知识库找答案
    - 问题涉及已上传文档的内容
    - 需要跨多个文档进行语义搜索
    
    【使用示例】
    - "这个项目的架构是什么？" → ask_knowledge_base("项目架构")
    - "如何配置环境？" → ask_knowledge_base("如何配置环境")
    - "RAG系统的工作原理" → ask_knowledge_base("RAG系统工作原理")
    - "有什么使用说明？" → ask_knowledge_base("使用说明")
    
    【不适用场景】
    - 用户只想看有哪些文件 → 使用 list_knowledge_base_files
    - 用户要找特定文件名的文件 → 使用 search_by_filename
    - 通用知识问题（不涉及知识库）→ 使用 general_qa
    
    【搜索特点】
    - 基于语义相似度，而非关键词匹配
    - 支持中文自然语言查询
    - 返回最相关的Top-K结果
    - 自动进行相关性判断
    
    【返回信息】
    - 基于检索内容生成的回答
    - 如果没有相关内容，会返回友好提示
    
    参数:
        query: 用户的自然语言问题，建议保持原意传递
    """
    cfg = config.get("configurable", {}) or {}
    session_id = cfg.get("session_id")
    project_id = cfg.get("project_id", "default")

    return rag_engine.get_answer(query, session_id=session_id, project_id=project_id)
