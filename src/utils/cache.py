# src/utils/cache.py
"""
缓存模块 - 支持 Redis（优先）和内存（fallback）
通过 REDIS_HOST 环境变量控制：
- Redis 可用 → 持久化缓存（跨重启保留）
- Redis 不可用 → 自动 fallback 到内存缓存
"""
import json
from typing import Optional, Any

from config.settings import settings
from src.utils.logger import setup_logger

logger = setup_logger("Cache")


class RedisCache:
    """Redis 缓存层"""

    def __init__(self, host: str = "localhost", port: int = 6379, ttl: int = 86400):
        import redis
        self.client = redis.Redis(host=host, port=port, decode_responses=True)
        self.ttl = ttl

    def get(self, key: str) -> Optional[str]:
        return self.client.get(f"rag:{key}")

    def set(self, key: str, value: str):
        self.client.setex(f"rag:{key}", self.ttl, value)

    def delete(self, key: str):
        self.client.delete(f"rag:{key}")

    def exists(self, key: str) -> bool:
        return self.client.exists(f"rag:{key}") > 0

    def clear(self):
        """清空所有 rag: 前缀的缓存"""
        for key in self.client.scan_iter("rag:*"):
            self.client.delete(key)


class MemoryCache:
    """内存缓存（fallback）"""

    def __init__(self, max_size: int = 1000):
        self.cache: dict = {}
        self.max_size = max_size

    def get(self, key: str) -> Optional[str]:
        return self.cache.get(key)

    def set(self, key: str, value: str):
        if len(self.cache) >= self.max_size:
            # LRU: 删除最早的项
            oldest = next(iter(self.cache))
            del self.cache[oldest]
        self.cache[key] = value

    def delete(self, key: str):
        self.cache.pop(key, None)

    def exists(self, key: str) -> bool:
        return key in self.cache

    def clear(self):
        self.cache.clear()


class EmbeddingCache:
    """
    Embedding 缓存：优先 Redis，fallback 到内存
    """

    def __init__(self):
        self.backend = "memory"
        try:
            self.redis = RedisCache(
                host=settings.REDIS_HOST,
                port=settings.REDIS_PORT
            )
            self.redis.client.ping()
            self.backend = "redis"
            logger.info(f"Embedding缓存使用 Redis ({settings.REDIS_HOST}:{settings.REDIS_PORT})")
        except Exception:
            self.redis = None
            self.memory = MemoryCache()
            logger.info("Embedding缓存使用内存（Redis 不可用）")

    def get(self, key: str) -> Optional[str]:
        if self.backend == "redis":
            return self.redis.get(key)
        return self.memory.get(key)

    def set(self, key: str, value: str):
        if self.backend == "redis":
            self.redis.set(key, value)
        else:
            self.memory.set(key, value)

    def exists(self, key: str) -> bool:
        if self.backend == "redis":
            return self.redis.exists(key)
        return self.memory.exists(key)

    def clear(self):
        if self.backend == "redis":
            self.redis.clear()
        else:
            self.memory.clear()

    @property
    def backend_name(self) -> str:
        return self.backend
