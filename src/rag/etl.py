# src/rag/etl.py
import os
import tempfile
from pathlib import Path
from typing import List, Optional

from streamlit.runtime.uploaded_file_manager import UploadedFile

from langchain_community.document_loaders import (
    TextLoader, 
    PyPDFLoader, 
    Docx2txtLoader, 
    UnstructuredMarkdownLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config.settings import settings
from src.utils.logger import setup_logger

logger = setup_logger("RAG_ETL")

# 针对中文文档的分隔符（按优先级排序）
CHINESE_SEPARATORS = [
    "\n\n",    # 段落分隔
    "\n",      # 行分隔
    "。",      # 中文句号
    "！",      # 中文感叹号
    "？",      # 中文问号
    "；",      # 中文分号
    "，",      # 中文逗号
    "：",      # 中文冒号
    "」",      # 中文右引号
    "」",      # 中文右角引号
    "『",      # 中文左双引号
    "』",      # 中文右双引号
    "(", ")",  # 英文括号
    "（", "）", # 中文括号
    " ",       # 空格
    ""         # 最后按字符切分
]

# 针对 Markdown 文档的分隔符
MARKDOWN_SEPARATORS = [
    "\n\n",    # 段落分隔
    "\n",      # 行分隔
    "```",     # 代码块
    "## ",     # 二级标题
    "### ",    # 三级标题
    "#### ",   # 四级标题
    "- ",      # 列表项
    "* ",      # 列表项
    "。",      # 中文句号
    "！",      # 中文感叹号
    "？",      # 中文问号
    "；",      # 中文分号
    "，",      # 中文逗号
    " ",       # 空格
    ""         # 最后按字符切分
]


class ContentProcessor:
    def __init__(self):
        self.chunk_size = settings.CHUNK_SIZE
        self.chunk_overlap = settings.CHUNK_OVERLAP

    def load_uploaded_files(self, uploaded_files: List[UploadedFile]) -> List:
        """
        直接处理内存中的文件对象，不持久化保存到磁盘。
        使用临时文件技术适配 LangChain Loader。
        """
        documents = []
        
        for up_file in uploaded_files:
            tmp_path = None
            try:
                # 1. 创建临时文件
                suffix = Path(up_file.name).suffix
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
                    tmp_file.write(up_file.getvalue())
                    tmp_path = tmp_file.name
                
                # 2. 选择加载器并读取
                logger.info(f"📄 正在处理: {up_file.name}")
                loader = self._select_loader(tmp_path, up_file.name)
                
                if loader:
                    docs = loader.load()
                    
                    # 3. 元数据修复：将 source 改回原始文件名
                    for doc in docs:
                        doc.metadata["source"] = up_file.name
                        # 记录文件类型，便于后续选择切分策略
                        doc.metadata["file_type"] = suffix.lower()
                        
                    documents.extend(docs)
                    logger.info(f"   ✅ 成功解析 {len(docs)} 个文档片段")
                
            except Exception as e:
                logger.error(f"❌ 处理文件 {up_file.name} 失败: {e}")
            finally:
                # 4. 确保临时文件被清理
                if tmp_path and os.path.exists(tmp_path):
                    try:
                        os.remove(tmp_path)
                    except Exception as cleanup_error:
                        logger.warning(f"⚠️ 清理临时文件失败: {cleanup_error}")

        logger.info(f"✅ 加载完成: 共解析 {len(documents)} 个文档片段")
        return documents

    def _select_loader(self, file_path: str, original_name: str) -> Optional[object]:
        """根据文件后缀选择加载器"""
        suffix = Path(original_name).suffix.lower()
        
        try:
            if suffix == ".txt":
                # 尝试多种编码，增强容错性
                return self._create_text_loader_with_fallback(file_path)
            elif suffix == ".md":
                return UnstructuredMarkdownLoader(file_path)
            elif suffix == ".pdf":
                return PyPDFLoader(file_path)
            elif suffix == ".docx":
                return Docx2txtLoader(file_path)
            elif suffix in [".py", ".js", ".java", ".c", ".cpp", ".ts", ".go", ".rs"]:
                # 代码文件使用 TextLoader
                return TextLoader(file_path, encoding="utf-8")
            else:
                logger.warning(f"⚠️ 暂不支持格式: {suffix}")
                return None
        except Exception as e:
            logger.error(f"❌ 创建加载器失败 ({suffix}): {e}")
            return None

    def _create_text_loader_with_fallback(self, file_path: str) -> TextLoader:
        """创建带有编码容错的 TextLoader"""
        # 优先尝试 UTF-8
        try:
            loader = TextLoader(file_path, encoding="utf-8")
            # 尝试读取一小段来验证编码
            with open(file_path, 'r', encoding='utf-8') as f:
                f.read(1024)
            return loader
        except UnicodeDecodeError:
            # 回退到 GBK（常见于 Windows 中文环境）
            logger.info(f"   📝 检测到非 UTF-8 编码，尝试 GBK...")
            return TextLoader(file_path, encoding="gbk", errors="ignore")

    def split_documents(self, documents: List) -> List:
        """
        智能切分文档：
        - 根据文档类型选择不同的分隔符策略
        - 非代码文件使用优化的中文分隔符
        """
        if not documents:
            return []
        
        all_chunks = []
        
        # 按文件类型分组处理
        for doc in documents:
            file_type = doc.metadata.get("file_type", "").lower()
            
            # 根据文件类型选择分隔符
            if file_type == ".md":
                separators = MARKDOWN_SEPARATORS
            elif file_type in [".py", ".js", ".java", ".c", ".cpp", ".ts", ".go", ".rs"]:
                # 代码文件保持原有逻辑（暂不优化）
                separators = ["\n\nclass ", "\n\ndef ", "\n\nclass ", "\n\ndef ",
                             "\nclass ", "\ndef ", "\n\n", "\n", " ", ""]
            else:
                # txt、pdf、docx 等普通文档使用中文分隔符
                separators = CHINESE_SEPARATORS
            
            # 创建切分器
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                separators=separators,
                length_function=len,
                is_separator_regex=False
            )
            
            # 切分单个文档
            chunks = text_splitter.split_documents([doc])
            all_chunks.extend(chunks)
        
        logger.info(f"✅ 切分完成: {len(documents)} 个文档 → {len(all_chunks)} 个片段")
        return all_chunks