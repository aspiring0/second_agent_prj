# config/settings.py
import os
from pathlib import Path
from dotenv import load_dotenv

# 加载 .env
load_dotenv()

class Settings:
    # 基础路径
    BASE_DIR = Path(__file__).resolve().parent.parent
    DATA_DIR = BASE_DIR / "data" / "raw"
    DB_DIR = BASE_DIR / "data" / "vector_db"
    LOG_DIR = BASE_DIR / "logs"

    # LLM & Embedding 配置
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    OPENAI_BASE_URL = os.getenv("OPENAI_API_BASE")
    
    # 模型选择 (方便统一修改)
    EMBEDDING_MODEL = "text-embedding-3-small" # 或 text-embedding-ada-002
    CHAT_MODEL = "gpt-3.5-turbo"
    
    # --- RAG 业务参数 ---
    # 切片大小：决定了喂给大模型的知识片段有多长
    CHUNK_SIZE = 500
    # 重叠大小：防止切片时把一句话切断了，保留一点上下文
    CHUNK_OVERLAP = 50
    CHUNK_SIZE = 500
    CHUNK_OVERLAP = 50
# 实例化对象，方便其他模块直接 import settings
settings = Settings()