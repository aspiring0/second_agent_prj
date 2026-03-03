# src/rag/etl.py
import os
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

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

# 针对 Python 代码的分隔符（优化版 - 保持类和函数完整）
PYTHON_SEPARATORS = [
    "\n\nclass ",     # 类定义前（双换行）
    "\n\ndef ",       # 函数定义前（双换行）
    "\n\nasync def ", # 异步函数定义前
    "\n    def ",     # 类方法
    "\nclass ",       # 类定义前（单换行）
    "\ndef ",         # 函数定义前（单换行）
    "\n\n",           # 空行
    "\n",             # 换行
    ";",              # 分号
    " ",              # 空格
    ""                # 字符
]

# 针对 Java 代码的分隔符
JAVA_SEPARATORS = [
    "\n\npublic class ",   # 公共类
    "\n\nclass ",          # 普通类
    "\n\npublic ",         # 公共方法
    "\n\nprivate ",        # 私有方法
    "\n\nprotected ",      # 保护方法
    "\n    public ",       # 类内公共方法
    "\n    private ",      # 类内私有方法
    "\n    protected ",    # 类内保护方法
    "\npublic ",           # 公共声明
    "\nprivate ",          # 私有声明
    "\n\n",                # 空行
    "\n",                  # 换行
    ";",                   # 分号
    " ",                   # 空格
    ""                     # 字符
]

# 针对 JavaScript/TypeScript 代码的分隔符
JS_SEPARATORS = [
    "\n\nfunction ",       # 函数
    "\n\nconst ",          # const声明
    "\n\nlet ",            # let声明
    "\n\nclass ",          # 类
    "\n\nexport ",         # 导出
    "\n    function ",     # 方法
    "\nfunction ",         # 函数（单换行）
    "\nconst ",            # const
    "\nlet ",              # let
    "\n\n",                # 空行
    "\n",                  # 换行
    ";",                   # 分号
    " ",                   # 空格
    ""                     # 字符
]

# 针对 Go 代码的分隔符
GO_SEPARATORS = [
    "\n\nfunc ",           # 函数
    "\n\nfunc (",          # 方法（带接收器）
    "\n\ntype ",           # 类型定义
    "\n\nstruct ",         # 结构体
    "\n\ninterface ",      # 接口
    "\nfunc ",             # 函数（单换行）
    "\ntype ",             # 类型
    "\n\n",                # 空行
    "\n",                  # 换行
    "{",                   # 左花括号
    "}",                   # 右花括号
    " ",                   # 空格
    ""                     # 字符
]

# 针对 C 代码的分隔符
C_SEPARATORS = [
    "\n\ntypedef ",        # 类型定义
    "\n\nstruct ",         # 结构体
    "\n\nenum ",           # 枚举
    "\n\nint ",            # 整型函数
    "\n\nvoid ",           # 无返回值函数
    "\n\nchar ",           # 字符函数
    "\n\nstatic ",         # 静态函数
    "\nint ",              # 整型声明
    "\nvoid ",             # void声明
    "\n\n",                # 空行
    "\n",                  # 换行
    "{",                   # 左花括号
    "}",                   # 右花括号
    ";",                   # 分号
    " ",                   # 空格
    ""                     # 字符
]

# 针对 Rust 代码的分隔符
RUST_SEPARATORS = [
    "\n\nfn ",             # 函数
    "\n\npub fn ",         # 公共函数
    "\n\nasync fn ",       # 异步函数
    "\n\nstruct ",         # 结构体
    "\n\nenum ",           # 枚举
    "\n\nimpl ",           # 实现
    "\n\ntrait ",          # trait
    "\nfn ",               # 函数（单换行）
    "\nstruct ",           # 结构体
    "\n\n",                # 空行
    "\n",                  # 换行
    "{",                   # 左花括号
    "}",                   # 右花括号
    " ",                   # 空格
    ""                     # 字符
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
        - 支持中文文档、Markdown、多种代码语言
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
            elif file_type == ".py":
                separators = PYTHON_SEPARATORS
            elif file_type == ".java":
                separators = JAVA_SEPARATORS
            elif file_type in [".js", ".ts"]:
                separators = JS_SEPARATORS
            elif file_type in [".c", ".cpp", ".go", ".rs"]:
                # C/C++/Go/Rust 使用类似的代码分隔符
                separators = [
                    "\n\nfunc ", "\n\nfunc (", "\n\nfn ", "\n\nstruct ",
                    "\nfunc ", "\nfn ", "\nstruct ",
                    "\n\n", "\n", ";", " ", ""
                ]
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
    
    def get_supported_file_types(self) -> Dict[str, str]:
        """返回支持的文件类型及描述"""
        return {
            ".txt": "纯文本文件（支持UTF-8/GBK编码）",
            ".md": "Markdown文档",
            ".pdf": "PDF文档",
            ".docx": "Word文档",
            ".py": "Python代码",
            ".js": "JavaScript代码",
            ".ts": "TypeScript代码",
            ".java": "Java代码",
            ".c": "C语言代码",
            ".cpp": "C++代码",
            ".go": "Go语言代码",
            ".rs": "Rust代码"
        }
