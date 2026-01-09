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
        self.vectordb = Chroma(
            persist_directory=self.persist_dir,
            embedding_function=self.embeddings
        )

    def query(self, question:str, top_k:int=3):
        """
        使用向量数据库进行相似度检索
        :param question: 用户查询的字符串
        :param k: 返回的相似文档数量
        :return: 最相似的 k 个文档
        """

        logger.info(f"正在检索与'{question}'相关的前 {top_k} 条文档...")
        try:
            docs = self.vectordb.similarity_search_with_score(
                query=question,
                k=top_k
            )
            logger.info(f"检索到 {len(docs)} 条相关文档。")
            return docs
        except Exception as e:
            logger.error(f"检索失败: {e}")
            raise e