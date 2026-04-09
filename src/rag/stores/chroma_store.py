# src/rag/stores/chroma_store.py
"""ChromaDB 向量存储实现"""

from typing import List, Tuple, Optional, Dict
from langchain_core.documents import Document
from langchain_chroma import Chroma

from config.settings import settings
from src.utils.logger import setup_logger
from src.utils.model_manager import model_manager
from . import VectorStoreBase

logger = setup_logger("ChromaStore")


class ChromaStore(VectorStoreBase):
    """ChromaDB 向量存储实现"""

    def __init__(self):
        self.persist_dir = str(settings.DB_DIR)
        self.embedding_fn = model_manager.get_embedding_model()
        self._db = Chroma(
            persist_directory=self.persist_dir,
            embedding_function=self.embedding_fn
        )
        logger.info(f"ChromaStore 初始化完成 (目录: {self.persist_dir})")

    @property
    def raw_client(self):
        """获取底层 ChromaDB 客户端（用于 get 等原始操作）"""
        return self._db

    def add_documents(self, documents: List[Document], project_id: str) -> int:
        for chunk in documents:
            chunk.metadata["project_id"] = project_id
        self._db.add_documents(documents=documents)
        logger.info(f"ChromaDB 添加 {len(documents)} 个文档 (project_id={project_id})")
        return len(documents)

    def similarity_search(self, query: str, top_k: int = 3,
                          filter: Optional[Dict] = None) -> List[Document]:
        return self._db.similarity_search(query, k=top_k, filter=filter)

    def similarity_search_with_score(self, query: str, top_k: int = 3,
                                     filter: Optional[Dict] = None) -> List[Tuple[Document, float]]:
        return self._db.similarity_search_with_score(query, k=top_k, filter=filter)

    def delete_by_filter(self, filter: Dict) -> bool:
        try:
            self._db._collection.delete(where=filter)
            return True
        except Exception as e:
            logger.error(f"ChromaDB 删除失败: {e}")
            return False

    def get_all_documents(self, filter: Optional[Dict] = None) -> List[Document]:
        kwargs = {"include": ["documents", "metadatas"]}
        if filter:
            kwargs["where"] = filter
        results = self._db.get(**kwargs)
        if not results or not results.get("documents"):
            return []
        documents = []
        for i, content in enumerate(results["documents"]):
            metadata = results["metadatas"][i] if results["metadatas"] else {}
            documents.append(Document(page_content=content, metadata=metadata))
        return documents

    def count(self, filter: Optional[Dict] = None) -> int:
        if filter:
            results = self._db.get(where=filter)
            return len(results.get("ids", []))
        return self._db._collection.count()

    def get(self, where: Optional[Dict] = None, include: Optional[List[str]] = None) -> Dict:
        """兼容 ChromaDB 原始 get 接口"""
        kwargs = {}
        if where:
            kwargs["where"] = where
        if include:
            kwargs["include"] = include
        return self._db.get(**kwargs)
