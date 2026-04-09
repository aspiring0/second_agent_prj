# src/rag/stores/__init__.py
"""向量存储抽象层 - 支持多种后端（ChromaDB / Qdrant）"""

from abc import ABC, abstractmethod
from typing import List, Tuple, Optional, Dict, Any
from langchain_core.documents import Document


class VectorStoreBase(ABC):
    """向量存储抽象基类"""

    @abstractmethod
    def add_documents(self, documents: List[Document], project_id: str) -> int:
        """添加文档到向量库，返回添加的文档数量"""

    @abstractmethod
    def similarity_search(self, query: str, top_k: int = 3,
                          filter: Optional[Dict] = None) -> List[Document]:
        """语义检索"""

    @abstractmethod
    def similarity_search_with_score(self, query: str, top_k: int = 3,
                                     filter: Optional[Dict] = None) -> List[Tuple[Document, float]]:
        """带分数检索"""

    @abstractmethod
    def delete_by_filter(self, filter: Dict) -> bool:
        """按条件删除"""

    @abstractmethod
    def get_all_documents(self, filter: Optional[Dict] = None) -> List[Document]:
        """获取所有文档（用于BM25索引构建）"""

    @abstractmethod
    def count(self, filter: Optional[Dict] = None) -> int:
        """统计文档数"""

    @abstractmethod
    def get(self, where: Optional[Dict] = None, include: Optional[List[str]] = None) -> Dict:
        """获取原始数据（兼容 ChromaDB get 接口）"""


def get_vector_store() -> VectorStoreBase:
    """
    工厂函数：根据配置返回对应的向量存储实现

    配置 VECTOR_STORE_BACKEND 环境变量：
    - "chroma"（默认）: 使用 ChromaDB 本地存储
    - "qdrant": 使用 Qdrant 向量数据库
    """
    from config.settings import settings
    backend = getattr(settings, 'VECTOR_STORE_BACKEND', 'chroma')

    if backend == "chroma":
        from .chroma_store import ChromaStore
        return ChromaStore()
    elif backend == "qdrant":
        from .qdrant_store import QdrantStore
        return QdrantStore()
    else:
        raise ValueError(f"未知的向量存储后端: {backend}")
