# src/service/document_service.py
"""
文档服务 - Document Service
负责文档上传、处理和入库
"""
from typing import List, Tuple
from dataclasses import dataclass
from collections import Counter
from pathlib import Path

from streamlit.runtime.uploaded_file_manager import UploadedFile

from src.rag.etl import ContentProcessor
from src.rag.vectorstore import VectorDBManager
from src.utils.db import add_project_file_record
from src.utils.logger import setup_logger

logger = setup_logger("DOC_SERVICE")


@dataclass
class IngestResult:
    """入库结果"""
    success: bool
    message: str
    total_chunks: int
    file_details: List[dict]


@dataclass
class ProcessedFile:
    """已处理的文件信息"""
    filename: str
    file_type: str
    chunks_count: int


class DocumentService:
    """文档处理服务"""
    
    def __init__(self):
        self.processor = ContentProcessor()
        self.vector_db = VectorDBManager()
        logger.info("✅ 文档服务初始化完成")
    
    def process_and_ingest(
        self, 
        uploaded_files: List[UploadedFile], 
        project_id: str
    ) -> IngestResult:
        """
        处理文件并入库
        
        Args:
            uploaded_files: 上传的文件列表
            project_id: 知识库 ID
            
        Returns:
            IngestResult: 入库结果
        """
        if not uploaded_files:
            return IngestResult(
                success=False,
                message="请先上传文件",
                total_chunks=0,
                file_details=[]
            )
        
        try:
            # 1. 加载文档
            logger.info(f"📄 开始处理 {len(uploaded_files)} 个文件...")
            docs = self.processor.load_uploaded_files(uploaded_files)
            
            if not docs:
                return IngestResult(
                    success=False,
                    message="未解析出有效内容",
                    total_chunks=0,
                    file_details=[]
                )
            
            # 2. 切分文档
            logger.info(f"✂️ 切分文档...")
            chunks = self.processor.split_documents(docs)
            
            # 3. 写入向量库
            logger.info(f"📥 写入向量库 (project_id={project_id})...")
            self.vector_db.create_vector_db(chunks, project_id=project_id)
            
            # 4. 统计并写入目录记录
            src_counter = Counter()
            for c in chunks:
                src = (c.metadata or {}).get("source", "unknown")
                src_counter[src] += 1
            
            file_details = []
            for f in uploaded_files:
                suffix = Path(f.name).suffix.lower().lstrip(".")
                chunks_count = src_counter.get(f.name, 0)
                
                add_project_file_record(
                    project_id=project_id,
                    source=f.name,
                    file_type=suffix,
                    chunks_count=chunks_count
                )
                
                file_details.append({
                    "filename": f.name,
                    "type": suffix,
                    "chunks": chunks_count
                })
            
            logger.info(f"✅ 入库完成: {len(chunks)} chunks, {len(file_details)} files")
            
            return IngestResult(
                success=True,
                message=f"入库成功：新增 {len(chunks)} 个片段，{len(file_details)} 个文件",
                total_chunks=len(chunks),
                file_details=file_details
            )
            
        except Exception as e:
            logger.error(f"❌ 入库失败: {e}")
            return IngestResult(
                success=False,
                message=f"入库失败: {str(e)}",
                total_chunks=0,
                file_details=[]
            )
    
    def get_supported_formats(self) -> List[str]:
        """获取支持的文件格式"""
        return [".txt", ".md", ".pdf", ".docx", ".py", ".js", ".java", ".c", ".cpp", ".ts", ".go", ".rs"]
    
    def is_supported(self, filename: str) -> bool:
        """检查文件是否支持"""
        suffix = Path(filename).suffix.lower()
        return suffix in self.get_supported_formats()
    
    def filter_supported_files(self, files: List[UploadedFile]) -> Tuple[List[UploadedFile], List[str]]:
        """过滤出支持的文件，返回 (支持的文件列表, 不支持的文件名列表)"""
        supported = []
        unsupported = []
        
        for f in files:
            if self.is_supported(f.name):
                supported.append(f)
            else:
                unsupported.append(f.name)
        
        return supported, unsupported