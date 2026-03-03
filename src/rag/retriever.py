# src/rag/retriever.py
"""
RAG检索模块 - 负责从向量数据库检索相关文档
包含：
1. Embedding缓存机制
2. 多知识库隔离
"""
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings
from config.settings import settings
from src.utils.logger import setup_logger
from src.utils.model_manager import model_manager
from typing import Dict, List, Tuple, Optional
import hashlib
import time
import json

logger = setup_logger("RAG_Retriever")

DEFAULT_PROJECT_ID = "default"


class CachedEmbeddings:
    """
    带缓存的Embedding包装器
    缓存查询向量，避免重复API调用
    """
    def __init__(self, embeddings: Embeddings, cache_size: int = 1000):
        self.embeddings = embeddings
        self.cache: Dict[str, List[float]] = {}
        self.cache_size = cache_size
        self.cache_hits = 0
        self.cache_misses = 0
    
    def _get_cache_key(self, text: str) -> str:
        """生成缓存键"""
        return hashlib.md5(text.encode('utf-8')).hexdigest()
    
    def embed_query(self, text: str) -> List[float]:
        """嵌入查询（带缓存）"""
        cache_key = self._get_cache_key(text)
        
        # 检查缓存
        if cache_key in self.cache:
            self.cache_hits += 1
            logger.debug(f"🎯 Embedding缓存命中 (命中率: {self.get_hit_rate():.1%})")
            return self.cache[cache_key]
        
        # 缓存未命中，调用API
        self.cache_misses += 1
        start_time = time.time()
        result = self.embeddings.embed_query(text)
        latency = (time.time() - start_time) * 1000
        logger.debug(f"📡 Embedding API调用 ({latency:.0f}ms)")
        
        # 存入缓存（LRU策略）
        if len(self.cache) >= self.cache_size:
            # 删除最早的缓存项
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
        
        self.cache[cache_key] = result
        return result
    
    def get_hit_rate(self) -> float:
        """获取缓存命中率"""
        total = self.cache_hits + self.cache_misses
        return self.cache_hits / total if total > 0 else 0
    
    def get_stats(self) -> Dict:
        """获取缓存统计"""
        return {
            "cache_size": len(self.cache),
            "max_size": self.cache_size,
            "hits": self.cache_hits,
            "misses": self.cache_misses,
            "hit_rate": self.get_hit_rate()
        }
    
    def clear_cache(self):
        """清空缓存"""
        self.cache.clear()
        self.cache_hits = 0
        self.cache_misses = 0
        logger.info("🧹 Embedding缓存已清空")


class VectorRetriever:
    def __init__(self, enable_cache: bool = True):
        self.persist_dir = str(settings.DB_DIR)
        self.enable_cache = enable_cache
        
        # 使用模型管理器获取Embedding模型
        base_embeddings = model_manager.get_embedding_model()
        
        # 包装为带缓存的Embedding
        if enable_cache:
            self.embeddings = CachedEmbeddings(base_embeddings)
            logger.info("✅ Embedding缓存已启用")
        else:
            self.embeddings = base_embeddings

        if not settings.DB_DIR.exists():
            logger.error(f"数据库目录不存在: {self.persist_dir}，请先运行入库脚本。")
            raise FileNotFoundError(f"数据库目录不存在: {self.persist_dir}")

        self.vector_db = Chroma(
            persist_directory=self.persist_dir,
            embedding_function=self.embeddings
        )

    def query(self, question: str, project_id: str = DEFAULT_PROJECT_ID, top_k=3) -> List[Tuple]:
        """
        检索相关文档
        
        Args:
            question: 查询问题
            project_id: 知识库ID
            top_k: 返回结果数量
            
        Returns:
            [(Document, score), ...] 列表
        """
        start_time = time.time()
        logger.info(f"🔍 检索: {question} [Project: {project_id}]")

        filter_rule = {"project_id": project_id}

        try:
            results = self.vector_db.similarity_search_with_score(
                question,
                k=top_k,
                filter=filter_rule
            )
            latency = (time.time() - start_time) * 1000
            
            # 记录缓存统计
            cache_info = ""
            if self.enable_cache and hasattr(self.embeddings, 'get_stats'):
                stats = self.embeddings.get_stats()
                cache_info = f" (缓存命中率: {stats['hit_rate']:.1%})"
            
            logger.info(f"✅ 检索到 {len(results)} 条记录 ({latency:.0f}ms){cache_info}")
            return results
        except Exception as e:
            logger.warning(f"检索为空或出错: {e}")
            return []
    
    def get_cache_stats(self) -> Optional[Dict]:
        """获取缓存统计信息"""
        if self.enable_cache and hasattr(self.embeddings, 'get_stats'):
            return self.embeddings.get_stats()
        return None
    
    def clear_cache(self):
        """清空Embedding缓存"""
        if self.enable_cache and hasattr(self.embeddings, 'clear_cache'):
            self.embeddings.clear_cache()
