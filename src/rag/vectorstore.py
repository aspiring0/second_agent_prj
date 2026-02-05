# src/rag/vectorstore.py
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from config.settings import settings
from src.utils.logger import setup_logger

logger = setup_logger("RAG_Database")

DEFAULT_PROJECT_ID = "default"

class VectorDBManager:
    def __init__(self):
        self.persist_dir = str(settings.DB_DIR)
        self.embedding_fn = OpenAIEmbeddings(
            model=settings.EMBEDDING_MODEL,
            openai_api_key=settings.OPENAI_API_KEY,
            openai_api_base=settings.OPENAI_BASE_URL
        )

    def create_vector_db(self, chunks, project_id: str = DEFAULT_PROJECT_ID):
        if not chunks:
            logger.warning("没有需要入库的文档块")
            return None

        logger.info(f"🏷️ 为 {len(chunks)} 个文档块打上项目标签 project_id={project_id}")
        for chunk in chunks:
            chunk.metadata["project_id"] = project_id

        logger.info("💾 正在写入向量数据库 (Mode: Append)...")

        vectordb = Chroma(
            persist_directory=self.persist_dir,
            embedding_function=self.embedding_fn
        )
        vectordb.add_documents(documents=chunks)
        logger.info("🎉 入库成功！")
        return vectordb
