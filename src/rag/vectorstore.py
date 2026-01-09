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

    def create_vector_db(self, chunks,mode="append"):
        """
        阶段三：入库 (Load)
        将文本块向量化并存入 ChromaDB
        """
        # 1. 检查是否需要清理旧数据
        # 在开发阶段，为了保证数据纯净，每次入库我们通常选择“重建”
        if mode=="overwrite" and settings.DB_DIR.exists():
            logger.warning(f"🧹 检测到旧数据库，正在清理: {self.persist_dir}")
            shutil.rmtree(self.persist_dir)

        # 2. 初始化数据库连接
        # 如果目录存在且有数据，Chroma 会自动加载旧数据
        vectordb = Chroma(
            persist_directory=self.persist_dir,
            embedding_function=self.embedding_fn
        )

        logger.info(f"💾 正在以 [{mode}] 模式写入数据...")
        
        try:
            # 3. 添加新文档 (add_documents 而不是 from_documents)
            # 注意：Chroma 会自动分配 ID，如果要防止同一份文件被重复添加，
            # 需要更复杂的逻辑（比如计算文件 Hash），我们暂时先做基础的追加。
            vectordb.add_documents(documents=chunks)
            
            logger.info(f"🎉 入库成功！新增块数: {len(chunks)}")
            return vectordb
        except Exception as e:
            logger.error(f"❌ 入库失败: {e}")
            raise e