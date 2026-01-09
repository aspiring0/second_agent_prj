# main.py
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# 1. 加载环境变量
load_dotenv()

# 2. 初始化模型
llm = ChatOpenAI(
    model="gpt-3.5-turbo", # 或者 deepseek-chat 等
    temperature=0
)

# 3. 测试调用
try:
    response = llm.invoke("你好，请只回复三个字：环境通畅")
    print(f"测试结果: {response.content}")
except Exception as e:
    print(f"连接失败: {e}")