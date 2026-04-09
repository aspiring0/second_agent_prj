# src/rag/reranker.py
"""
重排序模块 - 对检索结果进行二次排序
支持 Cohere Rerank API 和本地 BGE Cross-Encoder 两种后端
"""

from typing import List, Tuple
from langchain_core.documents import Document

from config.settings import settings
from src.utils.logger import setup_logger

logger = setup_logger("Reranker")


class Reranker:
    """
    重排序器：对检索结果二次排序，提升 Top-K 精度

    支持两种后端：
    - cohere: 使用 Cohere Rerank API（零部署，需 API Key）
    - bge: 使用本地 BGE Cross-Encoder（需 GPU，无 API 费用）
    """

    def __init__(self, backend: str = "cohere"):
        """
        初始化重排序器

        Args:
            backend: 重排序后端，"cohere" 或 "bge"
        """
        self.backend = backend
        logger.info(f"重排序器初始化完成，后端: {backend}")

    def rerank(
        self,
        query: str,
        documents: List[Tuple[Document, float]],
        top_k: int = 3,
    ) -> List[Tuple[Document, float]]:
        """
        对检索结果重排序

        Args:
            query: 用户查询
            documents: 检索结果列表 [(Document, score), ...]
            top_k: 返回的结果数量

        Returns:
            重排序后的 [(Document, rerank_score), ...] 列表
        """
        if not documents:
            return []

        if self.backend == "cohere":
            return self._cohere_rerank(query, documents, top_k)
        elif self.backend == "bge":
            return self._bge_rerank(query, documents, top_k)
        else:
            logger.warning(f"未知后端: {self.backend}，返回原始结果")
            return documents[:top_k]

    def _cohere_rerank(
        self,
        query: str,
        documents: List[Tuple[Document, float]],
        top_k: int,
    ) -> List[Tuple[Document, float]]:
        """
        使用 Cohere Rerank API 重排序

        Args:
            query: 查询文本
            documents: 检索结果
            top_k: 返回数量

        Returns:
            重排序后的结果列表
        """
        try:
            import cohere

            api_key = settings.COHERE_API_KEY
            if not api_key:
                logger.warning("COHERE_API_KEY 未设置，跳过重排序")
                return documents[:top_k]

            client = cohere.ClientV2(api_key=api_key)

            # 提取文档文本
            doc_texts = [doc.page_content for doc, _ in documents]

            # 调用 Cohere Rerank API
            response = client.rerank(
                model="rerank-v3.5",
                query=query,
                documents=doc_texts,
                top_n=min(top_k, len(doc_texts)),
            )

            # 构建重排序结果
            results = []
            for item in response.results:
                doc = documents[item.index][0]
                results.append((doc, item.relevance_score))

            logger.info(f"Cohere 重排序完成，返回 {len(results)} 条结果")
            return results

        except ImportError:
            logger.warning("cohere 库未安装，跳过重排序。安装: pip install cohere")
            return documents[:top_k]
        except Exception as e:
            logger.warning(f"Cohere 重排序失败: {e}，返回原始结果")
            return documents[:top_k]

    def _bge_rerank(
        self,
        query: str,
        documents: List[Tuple[Document, float]],
        top_k: int,
    ) -> List[Tuple[Document, float]]:
        """
        使用本地 BGE Cross-Encoder 重排序

        Args:
            query: 查询文本
            documents: 检索结果
            top_k: 返回数量

        Returns:
            重排序后的结果列表
        """
        try:
            from sentence_transformers import CrossEncoder

            # 加载 BGE reranker 模型
            model = CrossEncoder("BAAI/bge-reranker-base")

            # 构建查询-文档对
            pairs = [(query, doc.page_content) for doc, _ in documents]

            # 计算相关性分数
            scores = model.predict(pairs)

            # 按分数降序排列
            scored_docs = list(zip(documents, scores))
            scored_docs.sort(key=lambda x: x[1], reverse=True)

            results = []
            for (doc, _original_score), rerank_score in scored_docs[:top_k]:
                results.append((doc, float(rerank_score)))

            logger.info(f"BGE 重排序完成，返回 {len(results)} 条结果")
            return results

        except ImportError:
            logger.warning(
                "sentence-transformers 库未安装，跳过重排序。安装: pip install sentence-transformers"
            )
            return documents[:top_k]
        except Exception as e:
            logger.warning(f"BGE 重排序失败: {e}，返回原始结果")
            return documents[:top_k]
