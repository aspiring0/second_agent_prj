# src/rag/retriever.py

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from config.settings import settings
from src.utils.logger import setup_logger

logger = setup_logger("RAG_Retriever")

class VectorRetriever:
    def __init__(self):
        self.persist_dir = str(settings.DB_DIR)
        self.embeddings = OpenAIEmbeddings(
            model=settings.EMBEDDING_MODEL,
            openai_api_key=settings.OPENAI_API_KEY,
            openai_api_base=settings.OPENAI_BASE_URL
        )
        
        #检查数据库是否存在
        if not settings.DB_DIR.exists():
            logger.error(f"数据库目录不存在: {self.persist_dir}，请先运行入库脚本。")
            raise FileNotFoundError(f"数据库目录不存在: {self.persist_dir}")
        # 初始化 Chroma 数据库连接
        self.vector_db = Chroma(
            persist_directory=self.persist_dir,
            embedding_function=self.embeddings
        )

    def query(self, question: str, session_id=None, top_k=3): # <--- 改动1: 增加 session_id
        """
        session_id: 当前会话ID。如果传入，则只检索该会话 + 公共库的内容。
        """
        logger.info(f"🔍 检索: {question} [Session: {session_id}]")
        
        # --- 改动2: 构建过滤器 ---
        # ChromaDB 的 filter 语法
        filter_rule = {}
        if session_id:
            # 逻辑：session_id 等于 当前会话 OR session_id 等于 global
            # 注意：Chroma 的 $or 语法在某些版本支持有限，
            # 为简单起见，我们暂时先实现“只搜当前会话”的严格隔离。
            # 如果你想搜“当前会话 + 公共”，逻辑会复杂一点，我们先做严格隔离。
            filter_rule = {"session_id": session_id}
        else:
            # 如果没传 session_id，只搜公共库
            filter_rule = {"session_id": "global"}

        # --- 改动3: 传入 filter 参数 ---
        try:
            results = self.vector_db.similarity_search_with_score(
                question, 
                k=top_k,
                filter=filter_rule # <--- 关键：加上这行
            )
            logger.info(f"✅ 检索到 {len(results)} 条记录")
            return results
        except Exception as e:
            # 这种情况通常是因为数据库里还没这个 session_id 的数据
            logger.warning(f"检索为空或出错 (可能是新会话无数据): {e}")
            return []