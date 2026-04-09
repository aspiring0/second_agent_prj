# src/agent/tools_dir/knowledge_base.py
"""知识库工具：ask_knowledge_base, search_by_filename, list_knowledge_base_files"""

from langchain_core.tools import tool, BaseTool
from langgraph.config import RunnableConfig
from typing import List

from ._common import logger, rag_engine, get_chroma_db


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

    参数:
        query: 用户的自然语言问题，建议保持原意传递
    """
    cfg = config.get("configurable", {}) or {}
    session_id = cfg.get("session_id")
    project_id = cfg.get("project_id", "default")

    return rag_engine.get_answer(query, session_id=session_id, project_id=project_id)


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

    【返回信息】
    - 文件名列表
    - 每个文件的类型（PDF、TXT、PY等）
    - 每个文件的切片数量
    """
    cfg = config.get("configurable", {}) or {}
    project_id = cfg.get("project_id", "default")

    try:
        db = get_chroma_db()
        results = db.get(include=["metadatas"])

        if not results or not results.get("metadatas"):
            return "知识库中暂时没有任何文件。"

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

        output_lines = [f"知识库中共有 {len(files)} 个文件：\n"]
        for source, info in files.items():
            output_lines.append(f"  - {source} ({info['type']} 文件, {info['count']} 个片段)")

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

        matched_content = []
        filename_lower = filename.lower()

        for i, meta in enumerate(results["metadatas"]):
            source = meta.get("source", "").lower()
            if filename_lower in source or source.endswith(f".{filename_lower}"):
                doc_content = results["documents"][i] if i < len(results["documents"]) else ""
                if project_id == "default" or meta.get("project_id") == project_id:
                    matched_content.append(f"【来源: {meta.get('source')}】\n{doc_content}")

        if not matched_content:
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

        total_content = "\n\n---\n\n".join(matched_content)
        logger.info(f"按文件名 '{filename}' 搜索到 {len(matched_content)} 个片段")

        return f"找到 {len(matched_content)} 个与 '{filename}' 相关的内容片段：\n\n{total_content}"

    except Exception as e:
        logger.error(f"按文件名搜索失败: {e}")
        return f"搜索失败: {str(e)}"


def get_tools() -> List[BaseTool]:
    """返回知识库相关工具"""
    return [ask_knowledge_base, list_knowledge_base_files, search_by_filename]
