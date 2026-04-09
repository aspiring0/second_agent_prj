# test_isolation.py
import sys
import os
import shutil
from langchain_core.documents import Document

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.rag.vectorstore import VectorDBManager
from src.rag.retriever import VectorRetriever
from config.settings import settings

def main():
    print("🧪 开始会话隔离测试...")
    
    # 1. 准备假数据
    doc_a = [Document(page_content="这里的秘密密码是：Apple123", metadata={"source": "a.txt"})]
    doc_b = [Document(page_content="这里的秘密密码是：Banana456", metadata={"source": "b.txt"})]
    
    # 2. 存入数据库 (分别标记 session_id)
    vm = VectorDBManager()
    
    # 清空旧库方便测试
    if settings.DB_DIR.exists():
        shutil.rmtree(settings.DB_DIR)
        
    print("💾 正在存入数据...")
    vm.create_vector_db(doc_a, session_id="session_A")
    vm.create_vector_db(doc_b, session_id="session_B")
    
    # 3. 检索测试
    retriever = VectorRetriever()
    
    print("\n🧐 测试 1: 在 Session_A 中搜索 '密码'")
    results_a = retriever.query("密码", session_id="session_A")
    for doc, score in results_a:
        print(f"   -> 找到: {doc.page_content}")
        
    print("\n🧐 测试 2: 在 Session_B 中搜索 '密码'")
    results_b = retriever.query("密码", session_id="session_B")
    for doc, score in results_b:
        print(f"   -> 找到: {doc.page_content}")

if __name__ == "__main__":
    main()