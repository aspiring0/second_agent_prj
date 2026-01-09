# scripts/ingest_knowledge.py
import sys
import os



# 将项目根目录加入 python path，防止找不到模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.rag.etl import ContentProcessor
from src.rag.vectorstore import VectorDBManager
from src.utils.logger import setup_logger

logger = setup_logger("RAG_Ingestion")
def main():
    print("🚀 启动企业级知识库构建流程...")
    
    # 1. 提取与转换 (ETL)
    processor = ContentProcessor()

    # 2. 加载文档
    docs = processor.load_documents()
    if not docs:
        logger.error("❌ 流程终止：没有找到可处理的文档。请检查 data/raw 文件夹。")
        return
    # 3. 切分文档    
    chunks = processor.split_documents(docs)

    # 4. 存入向量库 (Load)
    # force_rebuild=True 表示每次运行此脚本都重建库，保证数据最新
    vector_manager = VectorDBManager()
    vector_manager.create_vector_db(chunks)

if __name__ == "__main__":
    main()