# test_retrieval.py
import sys
import os

# 路径补丁
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.rag.retriever import VectorRetriever

def main():
    print("🚀 启动检索测试...")
    
    try:
        retriever = VectorRetriever()
        
        # 修改这里的问题，变成你刚才txt/pdf里有的内容
        # 比如你之前的 txt 里有 "公司休假制度"
        test_question = "Python 是什么？"  # 或者 "公司年假怎么算？"
        
        results = retriever.query(test_question, top_k=2)
        
        print(f"\n❓ 问题: {test_question}")
        print("-" * 50)
        
        for i, (doc, score) in enumerate(results):
            print(f"📄 [结果 {i+1}] (距离分: {score:.4f})")
            print(f"内容: {doc.page_content[:100]}...") # 只打印前100字
            print(f"来源: {doc.metadata.get('source', '未知')}")
            print("-" * 30)
            
    except Exception as e:
        print(f"❌ 发生错误: {e}")

if __name__ == "__main__":
    main()