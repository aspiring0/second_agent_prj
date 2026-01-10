# src/rag/vectorstore.py
# 相当于在资源管理器里右键点击文件夹 -> 删除。
import shutil

# Chroma 是数据库本体，但 LangChain 为了能统一操作不同的数据库（比如换成 FAISS 或 Milvus），
# 封装了这个类。它负责把 LangChain 的 Document 对象转换成 Chroma 能存的格式。
from langchain_community.vectorstores import Chroma

# 它的作用是把一段文字（比如 "你好"）发给 API，API 返回一串数字向量。
# 即使你用第三方接口，只要它是兼容 OpenAI 格式的，都用这个包。
from langchain_openai import OpenAIEmbeddings
from config.settings import settings
from src.utils.logger import setup_logger

logger = setup_logger("RAG_Database")

class VectorDBManager:
    def __init__(self):
        self.persist_dir = str(settings.DB_DIR) # 数据库存哪儿
        
        # 初始化 Embedding 模型 (根据 settings 配置)
        # 这就是把文字变数字的核心引擎
        self.embedding_fn = OpenAIEmbeddings(
            model=settings.EMBEDDING_MODEL,
            openai_api_key=settings.OPENAI_API_KEY,
            openai_api_base=settings.OPENAI_BASE_URL
        )

    def create_vector_db(self, chunks, session_id=None): # <--- 改动1: 增加 session_id 参数
        """
        chunks: 切分好的文档块
        session_id: 如果提供了 session_id，这些文档将只属于该会话
        """
        # 如果没有chunks，直接返回
        if not chunks:
            logger.warning("没有需要入库的文档块")
            return None

        # --- 改动2: 给每个 chunk 强制增加 metadata ---
        if session_id:
            logger.info(f"🏷️ 正在为 {len(chunks)} 个文档块打上会话标签: {session_id}")
            for chunk in chunks:
                # 在原有的 metadata (如 source) 基础上，追加 session_id
                chunk.metadata["session_id"] = session_id
        else:
            # 如果没传 session_id，标记为 "global" (公共知识库)
            for chunk in chunks:
                chunk.metadata["session_id"] = "global"

        logger.info(f"💾 正在写入向量数据库 (Mode: Append)...")
        
        try:
            vectordb = Chroma(
                persist_directory=self.persist_dir,
                embedding_function=self.embedding_fn
            )
            # 增量添加
            vectordb.add_documents(documents=chunks)
            logger.info(f"🎉 入库成功！")
            return vectordb
        except Exception as e:
            logger.error(f"❌ 入库失败: {e}")
            raise e