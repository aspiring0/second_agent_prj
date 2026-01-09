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

    def create_vector_db(self, chunks):
        """
        阶段三：入库 (Load)
        将文本块向量化并存入 ChromaDB
        """
        # 1. 检查是否需要清理旧数据
        # 在开发阶段，为了保证数据纯净，每次入库我们通常选择“重建”
        if settings.DB_DIR.exists():
            logger.warning(f"🧹 检测到旧数据库，正在清理: {self.persist_dir}")
            shutil.rmtree(self.persist_dir)

        logger.info("💾 正在调用 Embedding 接口进行向量化 (这可能需要一点时间)...")
        
        try:
            # 2. 创建并持久化
            # from_documents 会自动做两件事：
            #   a. 调用 OpenAI 接口把 chunks 变成向量
            #   b. 把向量和原文存入本地文件夹
            vectordb = Chroma.from_documents(
                documents=chunks,
                embedding=self.embedding_fn,
                persist_directory=self.persist_dir
            )
            logger.info(f"🎉 知识库构建成功！数据已保存至: {self.persist_dir}")
            return vectordb
        except Exception as e:
            logger.error(f"❌ 向量化过程失败: {e}")
            raise e