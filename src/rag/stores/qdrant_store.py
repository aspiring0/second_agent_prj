# src/rag/stores/qdrant_store.py
"""Qdrant 向量存储实现"""

from typing import List, Tuple, Optional, Dict
from langchain_core.documents import Document

from config.settings import settings
from src.utils.logger import setup_logger
from src.utils.model_manager import model_manager
from . import VectorStoreBase

logger = setup_logger("QdrantStore")


class QdrantStore(VectorStoreBase):
    """Qdrant 向量存储实现"""

    def __init__(self):
        try:
            from qdrant_client import QdrantClient
        except ImportError:
            raise ImportError("请安装 qdrant-client: pip install qdrant-client")

        host = getattr(settings, 'QDRANT_HOST', 'localhost')
        port = int(getattr(settings, 'QDRANT_PORT', 6333))

        self.client = QdrantClient(host=host, port=port)
        self.collection_name = getattr(settings, 'QDRANT_COLLECTION', 'rag_documents')
        self.embedding_fn = model_manager.get_embedding_model()

        self._ensure_collection()
        logger.info(f"QdrantStore 初始化完成 (host={host}:{port})")

    def _ensure_collection(self):
        """确保集合存在"""
        from qdrant_client.models import Distance, VectorParams

        collections = self.client.get_collections().collections
        names = [c.name for c in collections]

        if self.collection_name not in names:
            # 获取向量维度（通过嵌入一个测试字符串）
            test_embedding = self.embedding_fn.embed_query("test")
            vector_size = len(test_embedding)

            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
            )
            logger.info(f"创建 Qdrant 集合: {self.collection_name} (维度: {vector_size})")

    def add_documents(self, documents: List[Document], project_id: str) -> int:
        from langchain_qdrant import QdrantVectorStore

        for chunk in documents:
            chunk.metadata["project_id"] = project_id

        vector_store = QdrantVectorStore(
            client=self.client,
            collection_name=self.collection_name,
            embedding=self.embedding_fn,
        )
        vector_store.add_documents(documents)
        logger.info(f"Qdrant 添加 {len(documents)} 个文档 (project_id={project_id})")
        return len(documents)

    def similarity_search(self, query: str, top_k: int = 3,
                          filter: Optional[Dict] = None) -> List[Document]:
        from langchain_qdrant import QdrantVectorStore

        vector_store = QdrantVectorStore(
            client=self.client,
            collection_name=self.collection_name,
            embedding=self.embedding_fn,
        )
        qdrant_filter = self._build_filter(filter)
        return vector_store.similarity_search(query, k=top_k, filter=qdrant_filter)

    def similarity_search_with_score(self, query: str, top_k: int = 3,
                                     filter: Optional[Dict] = None) -> List[Tuple[Document, float]]:
        from langchain_qdrant import QdrantVectorStore

        vector_store = QdrantVectorStore(
            client=self.client,
            collection_name=self.collection_name,
            embedding=self.embedding_fn,
        )
        qdrant_filter = self._build_filter(filter)
        return vector_store.similarity_search_with_score(query, k=top_k, filter=qdrant_filter)

    def delete_by_filter(self, filter: Dict) -> bool:
        try:
            from qdrant_client.models import Filter, FieldCondition, MatchValue

            conditions = []
            for key, value in filter.items():
                conditions.append(FieldCondition(key=f"metadata.{key}", match=MatchValue(value=value)))

            self.client.delete(
                collection_name=self.collection_name,
                points_selector=Filter(must=conditions)
            )
            return True
        except Exception as e:
            logger.error(f"Qdrant 删除失败: {e}")
            return False

    def get_all_documents(self, filter: Optional[Dict] = None) -> List[Document]:
        from qdrant_client.models import Filter, FieldCondition, MatchValue

        scroll_filter = None
        if filter:
            conditions = [
                FieldCondition(key=f"metadata.{key}", match=MatchValue(value=value))
                for key, value in filter.items()
            ]
            scroll_filter = Filter(must=conditions)

        points, _ = self.client.scroll(
            collection_name=self.collection_name,
            limit=10000,
            with_payload=True,
            scroll_filter=scroll_filter
        )

        documents = []
        for point in points:
            payload = point.payload or {}
            content = payload.get("page_content", "")
            metadata = payload.get("metadata", {})
            documents.append(Document(page_content=content, metadata=metadata))
        return documents

    def count(self, filter: Optional[Dict] = None) -> int:
        if filter:
            docs = self.get_all_documents(filter=filter)
            return len(docs)
        info = self.client.get_collection(self.collection_name)
        return info.points_count

    def get(self, where: Optional[Dict] = None, include: Optional[List[str]] = None) -> Dict:
        """兼容 ChromaDB 的 get 接口"""
        docs = self.get_all_documents(filter=where)
        result = {"ids": [], "documents": [], "metadatas": []}
        for i, doc in enumerate(docs):
            result["ids"].append(str(i))
            result["documents"].append(doc.page_content)
            result["metadatas"].append(doc.metadata)
        return result

    @staticmethod
    def _build_filter(filter_dict: Optional[Dict]):
        """将简单 filter 字典转换为 Qdrant Filter"""
        if not filter_dict:
            return None
        from qdrant_client.models import Filter, FieldCondition, MatchValue

        conditions = [
            FieldCondition(key=f"metadata.{k}", match=MatchValue(value=v))
            for k, v in filter_dict.items()
        ]
        return Filter(must=conditions)
