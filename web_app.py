# web_app.py
import streamlit as st
import os
import shutil
from langchain_core.messages import HumanMessage, AIMessage

# 引入我们的核心逻辑
from src.agent.graph import app as agent_app
from src.rag.etl import ContentProcessor
from src.rag.vectorstore import VectorDBManager
from config.settings import settings

# --- 页面配置 ---
st.set_page_config(page_title="企业级 RAG 智能助手", page_icon="🤖", layout="wide")

st.title("🤖 企业级 RAG + Multi-Agent 协作系统")

# --- 侧边栏：知识库管理 ---
with st.sidebar:
    st.header("📚 知识库管理")
    
    # 1. 文件上传组件
    uploaded_files = st.file_uploader(
        "上传文档 (TXT, MD)", 
        type=["txt", "md"], 
        accept_multiple_files=True
    )
    
    # 2. 上传与重建按钮
    if st.button("🚀 更新知识库"):
        if not uploaded_files:
            st.warning("请先选择文件！")
        else:
            status_text = st.empty()
            status_text.info("正在处理文件...")
            
            # 确保目录存在
            if not settings.DATA_DIR.exists():
                settings.DATA_DIR.mkdir(parents=True)
            
            # 保存上传的文件到 data/raw
            saved_count = 0
            for uploaded_file in uploaded_files:
                file_path = settings.DATA_DIR / uploaded_file.name
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                saved_count += 1
            
            status_text.info(f"文件保存成功 ({saved_count} 个)，开始构建索引...")
            
            # --- 调用后端逻辑 (ETL + 入库) ---
            try:
                # 1. 加载与切分
                processor = ContentProcessor()
                docs = processor.load_documents()
                chunks = processor.split_documents(docs)
                
                # 2. 向量化入库 (使用 append 模式)
                vector_manager = VectorDBManager()
                vector_manager.create_vector_db(chunks, mode="append")
                
                status_text.success("✅ 知识库更新完成！Agent 已读取最新文档。")
            except Exception as e:
                status_text.error(f"❌ 更新失败: {e}")

    st.divider()
    st.markdown("### 调试信息")
    st.info(f"当前模型: {settings.CHAT_MODEL}")

# --- 主界面：聊天窗口 ---

# 1. 初始化聊天历史 (Session State)
# Streamlit 每次刷新都会重置变量，所以需要用 session_state 记住聊天记录
if "messages" not in st.session_state:
    st.session_state.messages = []

# 2. 显示历史消息
for message in st.session_state.messages:
    # message 是 (role, content) 的字典
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 3. 处理用户输入
if prompt := st.chat_input("请输入你的问题..."):
    # A. 显示用户消息
    with st.chat_message("user"):
        st.markdown(prompt)
    # 记录到历史
    st.session_state.messages.append({"role": "user", "content": prompt})

    # B. 调用 Agent (显示思考过程)
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        # 构造 LangGraph 输入
        inputs = {"messages": [HumanMessage(content=prompt)]}
        
        # 实时流式输出 (Stream)
        # 这里的 stream 稍微复杂点，因为我们要过滤出“最终回答”
        try:
            status_container = st.status("🤖 Agent 正在思考...", expanded=True)
            
            for event in agent_app.stream(inputs):
                for node_name, node_output in event.items():
                    # 在折叠面板里显示思考过程
                    if node_name == "researcher":
                        status_container.write("🔍 研究员: 正在分析需求...")
                    elif node_name == "tools":
                        status_container.write("📚 工具: 正在检索知识库...")
                    elif node_name == "writer":
                        status_container.write("✍️ 作家: 正在撰写回复...")
                        # 拿到最终结果
                        final_msg = node_output["messages"][-1]
                        full_response = final_msg.content
            
            status_container.update(label="✅ 回答完成", state="complete", expanded=False)
            
            # 显示最终回答
            message_placeholder.markdown(full_response)
            
            # 记录 AI 回答到历史
            st.session_state.messages.append({"role": "assistant", "content": full_response})
            
        except Exception as e:
            st.error(f"发生错误: {e}")