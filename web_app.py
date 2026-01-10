# web_app.py
import streamlit as st
import uuid
import os
from langchain_core.messages import HumanMessage

from src.agent.graph import app as agent_app
from src.rag.etl import ContentProcessor
from src.rag.vectorstore import VectorDBManager
from config.settings import settings
from src.utils.db import init_db, create_session, get_all_sessions, save_message, get_messages

# 初始化数据库
init_db()

st.set_page_config(page_title="Agent 企业版 (隔离支持)", layout="wide")

# --- 侧边栏：会话管理 ---
with st.sidebar:
    st.title("🗂️ 会话管理")
    
    # 1. 新建会话
    if st.button("➕ 新建聊天"):
        new_id = str(uuid.uuid4())
        create_session(new_id, f"对话 {new_id[:4]}")
        st.session_state["current_session_id"] = new_id
        st.rerun()

    # 2. 获取会话列表
    sessions = get_all_sessions()
    if not sessions:
        first_id = str(uuid.uuid4())
        create_session(first_id, "默认对话")
        st.session_state["current_session_id"] = first_id
        st.rerun()

    # 3. 切换会话逻辑
    # 构造选项字典 {id: name}
    session_map = {s[0]: s[1] for s in sessions}
    session_ids = [s[0] for s in sessions]
    
    # 确保 session_state 里有值
    if "current_session_id" not in st.session_state:
        st.session_state["current_session_id"] = session_ids[0]
    
    # 保持选中状态
    current_idx = 0
    if st.session_state["current_session_id"] in session_ids:
        current_idx = session_ids.index(st.session_state["current_session_id"])
        
    selected_id = st.selectbox(
        "选择历史对话:", 
        options=session_ids,
        format_func=lambda x: session_map[x],
        index=current_idx
    )
    
    # 检测切换
    if selected_id != st.session_state["current_session_id"]:
        st.session_state["current_session_id"] = selected_id
        st.rerun()

    current_session_id = st.session_state["current_session_id"]

    st.divider()
    
    # --- 关键修改：带隔离的上传 ---
    st.markdown("### 📚 当前会话知识库")
    st.info(f"上传的文件将仅对【{session_map[current_session_id]}】可见")
    
    uploaded_files = st.file_uploader("上传文档", accept_multiple_files=True)
    
    if st.button("🚀 更新当前会话知识库"):
        if uploaded_files:
            # 注意：这里不需要 check data 目录是否存在了，因为我们根本不存

            status = st.empty()
            status.info(f"正在内存处理 {len(uploaded_files)} 个文件...")
            
            try:
                processor = ContentProcessor()
                
                # 🟢 核心修改：直接传 uploaded_files 对象列表
                # 不再需要传路径列表了
                docs = processor.load_uploaded_files(uploaded_files)
                
                if not docs:
                    status.warning("⚠️ 未能解析出有效内容")
                else:
                    chunks = processor.split_documents(docs)
                    
                    # 入库 (带 session_id)
                    VectorDBManager().create_vector_db(chunks, session_id=current_session_id)
                    
                    status.success(f"✅ 入库成功！新增 {len(chunks)} 个片段。")
                    
            except Exception as e:
                status.error(f"❌ 失败: {e}")

# --- 主界面 ---
st.header(f"💬 {session_map[current_session_id]}")

# 加载历史
db_messages = get_messages(current_session_id)
for msg in db_messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("输入问题..."):
    with st.chat_message("user"):
        st.markdown(prompt)
    save_message(current_session_id, "user", prompt)

    with st.chat_message("assistant"):
        status_box = st.status("Agent 思考中...", expanded=True)
        full_response = ""
        inputs = {"messages": [HumanMessage(content=prompt)]}
        
        # 🟢 关键点：把 session_id 打包进 config 传给 Agent
        # 这样 Agent 跑到 tools.py 时，就能拿出这个 id
        run_config = {"configurable": {"session_id": current_session_id}}
        
        try:
            # 传入 config
            for event in agent_app.stream(inputs, config=run_config):
                for node_name, node_output in event.items():
                    if node_name == "researcher":
                        status_box.write("🔍 研究员: 分析需求...")
                    elif node_name == "tools":
                        status_box.write("📚 工具: 检索【当前会话】资料...")
                    elif node_name == "writer":
                        status_box.write("✍️ 作家: 整理回答...")
                        full_response = node_output["messages"][-1].content
            
            status_box.update(label="✅ 完成", state="complete", expanded=False)
            st.markdown(full_response)
            save_message(current_session_id, "assistant", full_response)
            
        except Exception as e:
            st.error(f"Error: {e}")