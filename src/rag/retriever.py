# src/rag/retriever.py
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from config.settings import settings
from src.utils.logger import setup_logger

logger = setup_logger("RAG_Retriever")

DEFAULT_PROJECT_ID = "default"

class VectorRetriever:
    def __init__(self):
        self.persist_dir = str(settings.DB_DIR)
        self.embeddings = OpenAIEmbeddings(
            model=settings.EMBEDDING_MODEL,
            openai_api_key=settings.OPENAI_API_KEY,
            openai_api_base=settings.OPENAI_BASE_URL
        )

        if not settings.DB_DIR.exists():
            logger.error(f"数据库目录不存在: {self.persist_dir}，请先运行入库脚本。")
            raise FileNotFoundError(f"数据库目录不存在: {self.persist_dir}")

        self.vector_db = Chroma(
            persist_directory=self.persist_dir,
            embedding_function=self.embeddings
        )

    def query(self, question: str, project_id: str = DEFAULT_PROJECT_ID, top_k=3):
        logger.info(f"🔍 检索: {question} [Project: {project_id}]")

        filter_rule = {"project_id": project_id}

        try:
            results = self.vector_db.similarity_search_with_score(
                question,
                k=top_k,
                filter=filter_rule
            )
            logger.info(f"✅ 检索到 {len(results)} 条记录")
            return results
        except Exception as e:
            logger.warning(f"检索为空或出错: {e}")
            return []
