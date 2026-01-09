# src/rag/etl.py
from langchain_community.document_loaders import DirectoryLoader, TextLoader, UnstructuredMarkdownLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config.settings import settings
from src.utils.logger import setup_logger

logger = setup_logger("RAG_ETL")

class ContentProcessor:
    def __init__(self):
        self.data_path = settings.DATA_DIR
        self.chunk_size = settings.CHUNK_SIZE
        self.chunk_overlap = settings.CHUNK_OVERLAP

    def load_documents(self):
        """加载 data/raw 下的所有支持文档"""
        documents = []
        
        # 1. 加载 TXT
        # glob="**/*.txt" 表示递归查找子文件夹里的所有 txt
        if list(self.data_path.glob("**/*.txt")):
            logger.info("正在加载 TXT 文件...")
            loader = DirectoryLoader(
                self.data_path, glob="**/*.txt",
                loader_cls=TextLoader, 
                loader_kwargs={"encoding": "utf-8"})
            docs = loader.load()
            documents.extend(docs)
            logger.info(f"   -> 找到 {len(docs)} 个 TXT 文档")

        # 2. 加载 Markdown (通常用于技术文档)
        if list(self.data_path.glob("**/*.md")):
            logger.info("正在加载 Markdown 文件...")
            md_loader = DirectoryLoader(
                self.data_path, 
                glob="**/*.md", 
                loader_cls=TextLoader,
                loader_kwargs={"encoding": "utf-8"})
            docs = md_loader.load()
            documents.extend(md_loader.load())
            logger.info(f"   -> 找到 {len(docs)} 个 Markdown 文档")
            
        # 3. (预留) PDF 加载逻辑可在此添加
        if not documents:
            logger.warning("⚠️ 未在数据目录找到任何支持的文档 (txt/md)")
        logger.info(f"总计加载原始文档: {len(documents)} 份")
        return documents

    def split_documents(self, documents):
        """
        阶段二：切分 (Transform)
        将长文档切成小块 (Chunks)
        """
        if not documents:
            return []

        logger.info(f"✂️ 开始切分文档 (Size={self.chunk_size}, Overlap={self.chunk_overlap})...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", "。", "！", "，", ""] # 针对中文优化分隔符
        )
        chunks = text_splitter.split_documents(documents)
        logger.info(f"文档切分完成，生成 Chunks: {len(chunks)} 个")
        return chunks