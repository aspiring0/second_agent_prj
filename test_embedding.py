# test_embedding.py
import os
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

# 1. 加载配置
load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
base_url = os.getenv("OPENAI_API_BASE")

print(f"🔌 正在连接: {base_url}")
print(f"🔑 使用 Key: {api_key[:6]}******")

def test_embedding():
    try:
        # 尝试初始化，使用通用模型名 "text-embedding-3-small"
        # 如果你的服务商比较旧，可能需要改为 "text-embedding-ada-002"
        embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small", 
            openai_api_key=api_key,
            openai_api_base=base_url
        )
        
        print("📡 正在发送测试请求...")
        # 测试将“你好”两个字变成向量
        vector = embeddings.embed_query("你好")
        
        print("✅ 测试成功！")
        print(f"🔢 向量维度: {len(vector)}")
        print(f"👀 前10位数据: {vector[:10]}")
        return True

    except Exception as e:
        print("\n❌ 测试失败！")
        print(f"原因: {e}")
        print("-" * 30)
        print("💡 建议方案：")
        print("1. 你的服务商可能不支持 'text-embedding-3-small'，请修改代码尝试 'text-embedding-ada-002'。")
        print("2. 如果还不行，说明该服务商完全不支持 Embedding。我们需要切换到【本地 HuggingFace 模型】（免费且不用联网）。")
        return False

if __name__ == "__main__":
    test_embedding()