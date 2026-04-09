# src/rag/vectorstore.py
from langchain_chroma import Chroma
from config.settings import settings
from src.utils.logger import setup_logger
from src.utils.model_manager import model_manager

logger = setup_logger("RAG_Database")

DEFAULT_PROJECT_ID = "default"

class VectorDBManager:
    def __init__(self):
        self.persist_dir = str(settings.DB_DIR)
        # 使用模型管理器获取Embedding模型
        self.embedding_fn = model_manager.get_embedding_model()

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
