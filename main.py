# main.py (或 test_rag_complete.py)
import sys
import os

# 路径补丁
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.rag.generator import RAGGenerator

def main():
    print("🚀 启动企业级 RAG 问答系统...")
    
    # 初始化生成器
    rag = RAGGenerator()
    
    while True:
        # 让用户在终端输入问题
        question = input("\n🙋 请输入问题 (输入 'exit' 退出): ")
        if question.lower() in ["exit", "quit"]:
            break
            
        if not question.strip():
            continue
            
        print("-" * 50)
        # 获取答案
        answer = rag.get_answer(question)
        
        print(f"🤖 AI 回答:\n{answer}")
        print("-" * 50)

if __name__ == "__main__":
    main()