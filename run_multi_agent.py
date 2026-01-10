# run_multi_agent.py
import sys
import os

# 把当前目录加入 Python 路径，防止找不到 src 包
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from langchain_core.messages import HumanMessage
from src.agent.graph import app
from src.utils.logger import setup_logger

logger = setup_logger("MultiAgent_Runner")

def main():
    print("🚀 启动双智能体协作系统 (Researcher + Writer)...")
    
    while True:
        question = input("\n🙋 请输入问题 (输入 'exit' 退出): ")
        if question.lower() in ["exit", "quit"]:
            break
        if not question.strip():
            continue

        print("-" * 50)
        logger.info(f"任务开始: {question}")
        
        # 构造初始状态：只包含用户的一句话
        initial_state = {"messages": [HumanMessage(content=question)]}
        
        # app.stream 会像流一样逐步输出每一个节点的执行结果
        # 这样我们就能看到：researcher 思考 -> tools 执行 -> writer 写作 的全过程
        for event in app.stream(initial_state):
            for node_name, node_output in event.items():
                logger.info(f"🔄 节点完成: {node_name}")
                
                # 如果是 writer 完成了，说明最后的结果出来了，打印出来
                if node_name == "writer":
                    last_msg = node_output["messages"][-1]
                    print(f"\n🤖 [最终回答]:\n{last_msg.content}")

        print("-" * 50)

if __name__ == "__main__":
    main()