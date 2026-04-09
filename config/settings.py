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

    # ==================== API Keys 配置 ====================
    # OpenAI
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    OPENAI_BASE_URL = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")
    
    # DeepSeek (国产替代，性价比高)
    DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
    
    # 智谱AI (GLM系列)
    ZHIPU_API_KEY = os.getenv("ZHIPU_API_KEY")
    
    # Moonshot (月之暗面)
    MOONSHOT_API_KEY = os.getenv("MOONSHOT_API_KEY")
    
    # Azure OpenAI (可选)
    AZURE_API_KEY = os.getenv("AZURE_API_KEY")
    AZURE_API_BASE = os.getenv("AZURE_API_BASE")
    
    # ==================== 默认模型配置 ====================
    # 可在运行时通过 ModelManager 动态切换
    # 这里设置的是默认值
    
    # Embedding模型：用于文档向量化
    # 推荐：text-embedding-3-small (性价比高)
    # 高质量：text-embedding-3-large
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    
    # 对话模型：用于RAG问答
    # 推荐：gpt-4o-mini (性价比高)
    # 高质量：gpt-4o
    # 国产替代：deepseek-chat, glm-4-flash
    CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")
    
    # Agent模型：用于多智能体系统（研究员/作家）
    # 建议使用能力更强的模型
    AGENT_MODEL = os.getenv("AGENT_MODEL", "gpt-4o-mini")
    
    # ==================== RAG 业务参数 ====================
    # 切片大小：决定了喂给大模型的知识片段有多长
    # 对于中文，建议 800-1000 字符（约 400-600 汉字）
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
    
    # 重叠大小：防止切片时把一句话切断了，保留一点上下文
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))
    
    # 检索参数
    RETRIEVAL_TOP_K = int(os.getenv("RETRIEVAL_TOP_K", "3"))
    
    # ==================== 模型切换功能开关 ====================
    # 是否启用动态模型切换（Web界面）
    ENABLE_MODEL_SWITCHING = os.getenv("ENABLE_MODEL_SWITCHING", "true").lower() == "true"

# 实例化对象，方便其他模块直接 import settings
settings = Settings()
