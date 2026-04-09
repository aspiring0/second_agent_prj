# src/rag/hybrid_retriever.py
"""
混合检索模块 - 向量检索 + BM25 关键词检索 + RRF 融合
通过结合语义搜索和关键词精确匹配，提升检索质量
"""

from typing import List, Tuple, Optional
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from src.utils.logger import setup_logger

logger = setup_logger("Hybrid_Retriever")


class HybridRetriever:
    """
    混合检索器：向量 + BM25 + RRF 融合

    结合语义向量检索（捕捉语义相似性）和 BM25 关键词检索（捕捉精确匹配），
    使用 Reciprocal Rank Fusion (RRF) 算法融合两路结果。
    """

    def __init__(self, vector_store, documents: List[Document], project_id: str):
        """
        初始化混合检索器

        Args:
            vector_store: Chroma 向量数据库实例
            documents: 用于构建 BM25 索引的文档列表
            project_id: 项目/知识库 ID
        """
        self.vector_store = vector_store
        self.project_id = project_id

        # 构建 BM25 索引
        self.documents = documents
        self.tokenized_corpus = [self._tokenize(doc.page_content) for doc in documents]
        self.bm25 = BM25Okapi(self.tokenized_corpus)

        logger.info(f"混合检索器初始化完成，BM25 索引文档数: {len(documents)}")

    def _tokenize(self, text: str) -> List[str]:
        """
        简单分词：按字符级拆分中文 + 按空格拆分英文

        Args:
            text: 待分词文本

        Returns:
            分词结果列表
        """
        tokens = []
        # 按空格拆分英文单词
        for word in text.split():
            word = word.strip()
            if not word:
                continue
            # 对英文单词直接加入
            if word.isascii():
                tokens.append(word.lower())
            else:
                # 对中文字符，按字符拆分（简单策略，生产环境可用 jieba）
                for char in word:
                    if '\u4e00' <= char <= '\u9fff':
                        tokens.append(char)
                    elif char.isalnum():
                        tokens.append(char.lower())
        return tokens

    def retrieve(self, query: str, top_k: int = 5) -> List[Tuple[Document, float]]:
        """
        执行混合检索：向量检索 + BM25 检索 + RRF 融合

        Args:
            query: 用户查询
            top_k: 返回结果数量

        Returns:
            [(Document, score), ...] 融合后的结果列表
        """
        # 1. 向量检索
        vector_results = self._vector_search(query, top_k=top_k * 2)
        logger.info(f"向量检索返回 {len(vector_results)} 条结果")

        # 2. BM25 检索
        bm25_results = self._bm25_search(query, top_k=top_k * 2)
        logger.info(f"BM25 检索返回 {len(bm25_results)} 条结果")

        # 3. RRF 融合
        fused_results = self._rrf_fuse(vector_results, bm25_results, top_k=top_k)
        logger.info(f"RRF 融合后返回 {len(fused_results)} 条结果")

        return fused_results

    def _vector_search(self, query: str, top_k: int = 10) -> List[Tuple[Document, float]]:
        """
        向量语义检索

        Args:
            query: 查询文本
            top_k: 返回数量

        Returns:
            [(Document, score), ...] 列表
        """
        try:
            filter_rule = {"project_id": self.project_id}
            results = self.vector_store.similarity_search_with_score(
                query, k=top_k, filter=filter_rule
            )
            return results
        except Exception as e:
            logger.warning(f"向量检索失败: {e}")
            return []

    def _bm25_search(self, query: str, top_k: int = 10) -> List[Tuple[Document, float]]:
        """
        BM25 关键词检索

        Args:
            query: 查询文本
            top_k: 返回数量

        Returns:
            [(Document, score), ...] 列表
        """
        try:
            tokenized_query = self._tokenize(query)
            if not tokenized_query:
                return []

            scores = self.bm25.get_scores(tokenized_query)

            # 按分数降序排列，取 top_k
            scored_docs = list(zip(range(len(scores)), scores))
            scored_docs.sort(key=lambda x: x[1], reverse=True)
            top_indices = scored_docs[:top_k]

            results = []
            for idx, score in top_indices:
                if score > 0 and idx < len(self.documents):
                    results.append((self.documents[idx], float(score)))

            return results
        except Exception as e:
            logger.warning(f"BM25 检索失败: {e}")
            return []

    def _rrf_fuse(
        self,
        vector_results: List[Tuple[Document, float]],
        bm25_results: List[Tuple[Document, float]],
        top_k: int = 5,
        k: int = 60,
    ) -> List[Tuple[Document, float]]:
        """
        Reciprocal Rank Fusion (RRF) 融合算法

        将向量检索和 BM25 检索的结果通过 RRF 公式融合排名：
        RRF_score = sum(1 / (k + rank + 1))

        Args:
            vector_results: 向量检索结果
            bm25_results: BM25 检索结果
            top_k: 返回的结果数量
            k: RRF 平滑参数，默认 60

        Returns:
            融合排序后的 [(Document, rrf_score), ...] 列表
        """
        scores = {}
        doc_map = {}

        # 处理向量检索结果
        for rank, (doc, _score) in enumerate(vector_results):
            key = doc.page_content[:200]
            scores[key] = scores.get(key, 0) + 1.0 / (k + rank + 1)
            if key not in doc_map:
                doc_map[key] = doc

        # 处理 BM25 检索结果
        for rank, (doc, _score) in enumerate(bm25_results):
            key = doc.page_content[:200]
            scores[key] = scores.get(key, 0) + 1.0 / (k + rank + 1)
            if key not in doc_map:
                doc_map[key] = doc

        # 按 RRF 分数降序排列
        sorted_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        # 返回 top_k 结果
        results = []
        for key, rrf_score in sorted_results[:top_k]:
            if key in doc_map:
                results.append((doc_map[key], rrf_score))

        return results
