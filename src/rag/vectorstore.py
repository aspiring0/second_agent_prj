# src/rag/vectorstore.py
from config.settings import settings
from src.utils.logger import setup_logger

logger = setup_logger("RAG_Database")

DEFAULT_PROJECT_ID = "default"

class VectorDBManager:
    def __init__(self):
        # 使用抽象层获取向量存储
        from src.rag.stores import get_vector_store
        self.store = get_vector_store()

    def create_vector_db(self, chunks, project_id: str = DEFAULT_PROJECT_ID):
        if not chunks:
            logger.warning("没有需要入库的文档块")
            return None

        logger.info(f"为 {len(chunks)} 个文档块打上项目标签 project_id={project_id}")

        count = self.store.add_documents(chunks, project_id)
        logger.info(f"入库成功！添加 {count} 个文档块")

        # 返回底层存储实例（兼容旧代码）
        if hasattr(self.store, 'raw_client'):
            return self.store.raw_client
        return self.store
