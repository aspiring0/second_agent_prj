# src/rag/etl.py
import os
import tempfile
from pathlib import Path
from typing import List

# 引入 Streamlit 的文件对象类型提示 (可选，为了代码规范)
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

class ContentProcessor:
    def __init__(self):
        self.chunk_size = settings.CHUNK_SIZE
        self.chunk_overlap = settings.CHUNK_OVERLAP

    def load_uploaded_files(self, uploaded_files: List[UploadedFile]):
        """
        直接处理内存中的文件对象，不持久化保存到磁盘。
        使用临时文件技术适配 LangChain Loader。
        """
        documents = []
        
        for up_file in uploaded_files:
            # up_file.name 是文件名 (e.g., "report.pdf")
            # up_file.getvalue() 是二进制内容
            
            try:
                # 1. 创建临时文件 (TempFile)
                # delete=False 是为了兼容 Windows，必须先关闭文件才能让 Loader 去再次打开读取
                suffix = Path(up_file.name).suffix
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
                    tmp_file.write(up_file.getvalue())
                    tmp_path = tmp_file.name  # 获取临时文件的绝对路径
                
                # 2. 选择加载器并读取
                logger.info(f"📄 正在内存处理: {up_file.name}")
                loader = self._select_loader(Path(tmp_path))
                
                if loader:
                    docs = loader.load()
                    
                    # 3. 关键修正：元数据修复
                    # 加载器读的是临时路径 (如 /tmp/tmp8x9s.pdf)，
                    # 我们要把 source 改回原始文件名 (report.pdf)，否则引用会乱码
                    for doc in docs:
                        doc.metadata["source"] = up_file.name
                        
                    documents.extend(docs)
                
                # 4. 清理现场：删除临时文件
                os.remove(tmp_path)
                
            except Exception as e:
                logger.error(f"❌ 处理文件 {up_file.name} 失败: {e}")
                # 确保发生错误也删除临时文件
                if 'tmp_path' in locals() and os.path.exists(tmp_path):
                    os.remove(tmp_path)

        logger.info(f"✅ 内存加载完成: 共解析 {len(documents)} 份文档")
        return documents

    def _select_loader(self, file_path: Path):
        """根据临时文件的后缀选择加载器"""
        suffix = file_path.suffix.lower()
        
        if suffix == ".txt":
            return TextLoader(str(file_path), encoding="utf-8")
        elif suffix == ".md":
            return UnstructuredMarkdownLoader(str(file_path))
        elif suffix == ".pdf":
            return PyPDFLoader(str(file_path))
        elif suffix == ".docx":
            return Docx2txtLoader(str(file_path))
        elif suffix in [".py", ".js", ".java", ".c", ".cpp"]:
            return TextLoader(str(file_path), encoding="utf-8")
        else:
            logger.warning(f"⚠️ 暂不支持格式: {suffix}")
            return None

    def split_documents(self, documents):
        if not documents:
            return []
            
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", "。", "！", "class ", "def ", " ", ""]
        )
        chunks = text_splitter.split_documents(documents)
        return chunks