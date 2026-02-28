"""
Web Application - 前端展示层 (Frontend)
只负责 UI 渲染和用户交互，所有业务逻辑由 Service 层处理
"""
import streamlit as st

# Service Layer
from src.service import KnowledgeBaseService, ChatService, DocumentService

# ==================== App State ====================
def init_app_state():
    """初始化应用状态"""
    if "view" not in st.session_state:
        st.session_state["view"] = "kb"
    if "current_project_id" not in st.session_state:
        st.session_state["current_project_id"] = None
    if "current_session_id" not in st.session_state:
        st.session_state["current_session_id"] = None


# ==================== Service Instances ====================
# 使用单例模式，避免重复初始化
@st.cache_resource
def get_kb_service() -> KnowledgeBaseService:
    return KnowledgeBaseService()

@st.cache_resource
def get_chat_service() -> ChatService:
    return ChatService()

@st.cache_resource
def get_doc_service() -> DocumentService:
    return DocumentService()


# ==================== Page: Knowledge Base ====================
def render_kb_page():
    """渲染知识库管理页面"""
    kb_service = get_kb_service()
    
    st.title("📚 知识库管理")
    
    # 确保有知识库
    kbs = kb_service.ensure_default_kb()
    kb_map = {kb.id: kb.name for kb in kbs}
    kb_ids = [kb.id for kb in kbs]
    
    # 选择知识库
    current_pid = st.session_state["current_project_id"]
    if current_pid is None or current_pid not in kb_ids:
        current_pid = "default" if "default" in kb_ids else kb_ids[0]
        st.session_state["current_project_id"] = current_pid
    
    selected = st.selectbox(
        "选择知识库",
        options=kb_ids,
        format_func=lambda x: f"{kb_map.get(x, x)} ({x})",
        index=kb_ids.index(current_pid)
    )
    st.session_state["current_project_id"] = selected
    
    # 知识库概览表格
    st.subheader("知识库概览")
    stats_table = kb_service.get_all_stats_table()
    st.dataframe(stats_table, use_container_width=True)
    
    col1, col2, col3 = st.columns(3)
    
    # 新建知识库
    with col1:
        st.subheader("新建")
        new_name = st.text_input("名称", value="", key="new_kb_name")
        new_id = st.text_input("ID（建议英文，如 kb_code）", value="", key="new_kb_id")
        if st.button("创建知识库"):
            if not new_id.strip() or not new_name.strip():
                st.warning("请填写名称与ID")
            elif kb_service.create_kb(new_id.strip(), new_name.strip()):
                st.session_state["current_project_id"] = new_id.strip()
                st.success("创建成功")
                st.rerun()
    
    # 浏览文件
    with col2:
        st.subheader("浏览")
        pid = st.session_state["current_project_id"]
        st.write(f"当前选择：{kb_map.get(pid, pid)} ({pid})")
        files = kb_service.get_kb_files(pid)
        if not files:
            st.caption("该知识库暂无文件")
        else:
            preview = [{
                "文件": f.source,
                "类型": f.file_type,
                "chunks": f.chunks_count,
                "入库时间": f.created_at,
            } for f in files[:10]]
            st.dataframe(preview, use_container_width=True, height=260)
    
    # 删除知识库
    with col3:
        st.subheader("删除")
        deletable = [kb.id for kb in kbs if kb.id != "default"]
        if deletable:
            del_id = st.selectbox(
                "选择要删除的知识库",
                options=deletable,
                format_func=lambda x: f"{kb_map.get(x, x)} ({x})",
            )
            if st.button("确认删除"):
                success, msg = kb_service.delete_kb(del_id)
                if success:
                    if st.session_state["current_project_id"] == del_id:
                        st.session_state["current_project_id"] = "default"
                    st.success(msg)
                    st.rerun()
                else:
                    st.error(msg)
        else:
            st.caption("无可删除的知识库")
    
    st.divider()
    
    if st.button("进入聊天"):
        _goto_chat(st.session_state["current_project_id"])


# ==================== Page: Chat ====================
def render_chat_page():
    """渲染聊天页面"""
    kb_service = get_kb_service()
    chat_service = get_chat_service()
    doc_service = get_doc_service()
    
    pid = st.session_state["current_project_id"]
    if not pid:
        st.session_state["view"] = "kb"
        st.rerun()
    
    # 确保会话存在
    sid = kb_service.get_or_create_session(pid, st.session_state.get("current_session_id"))
    st.session_state["current_session_id"] = sid
    
    st.title("💬 聊天")
    st.caption(f"当前知识库：{pid} ｜ 当前会话：{sid[:8]}")
    
    # 侧边栏
    with st.sidebar:
        render_chat_sidebar(kb_service, doc_service, pid, sid)
    
    # 主聊天区域
    render_chat_area(chat_service, pid, sid)


def render_chat_sidebar(kb_service, doc_service, pid, sid):
    """渲染聊天侧边栏"""
    st.header("🧭 控制台")
    
    if st.button("⬅️ 切换知识库"):
        st.session_state["view"] = "kb"
        st.rerun()
    
    st.divider()
    
    # 会话管理
    st.subheader("会话")
    sessions = kb_service.get_sessions(pid)
    session_map = {s[0]: s[1] for s in sessions}
    session_ids = [s[0] for s in sessions]
    
    # 处理当前会话不在列表中的情况
    if sid not in session_ids:
        sid = kb_service.get_or_create_session(pid)
        st.session_state["current_session_id"] = sid
        st.rerun()
    
    current_idx = session_ids.index(st.session_state["current_session_id"])
    selected_sid = st.selectbox(
        "选择会话",
        options=session_ids,
        format_func=lambda x: session_map.get(x, x),
        index=current_idx
    )
    if selected_sid != st.session_state["current_session_id"]:
        st.session_state["current_session_id"] = selected_sid
        st.rerun()
    
    # 新建会话
    new_title = st.text_input("新会话标题", value="", key="new_session_title")
    if st.button("➕ 新建会话"):
        if not new_title.strip():
            st.warning("请输入标题")
        else:
            new_id = kb_service.create_new_session(pid, new_title.strip())
            st.session_state["current_session_id"] = new_id
            st.rerun()
    
    st.divider()
    
    # 文件上传
    render_file_uploader(doc_service, pid)
    
    st.divider()
    
    # 文件目录
    with st.expander("📄 当前知识库文件目录", expanded=False):
        files = kb_service.get_kb_files(pid)
        if not files:
            st.caption("暂无文件")
        else:
            table = [{
                "文件": f.source,
                "类型": f.file_type,
                "chunks": f.chunks_count,
                "入库时间": f.created_at,
            } for f in files[:50]]
            st.dataframe(table, use_container_width=True, height=260)


def render_file_uploader(doc_service, pid):
    """渲染文件上传组件"""
    st.subheader("上传文件（写入当前知识库）")
    uploaded_files = st.file_uploader("上传文档", accept_multiple_files=True)
    
    if st.button("🚀 更新知识库"):
        if not uploaded_files:
            st.warning("请先上传文件")
        else:
            status = st.empty()
            status.info(f"正在处理 {len(uploaded_files)} 个文件...")
            
            result = doc_service.process_and_ingest(uploaded_files, pid)
            
            if result.success:
                status.success(f"✅ {result.message}")
            else:
                status.error(f"❌ {result.message}")


def render_chat_area(chat_service, pid, sid):
    """渲染聊天区域"""
    # 历史消息
    messages = chat_service.get_history(sid)
    for msg in messages:
        with st.chat_message(msg.role):
            st.markdown(msg.content)
    
    # 输入框
    if prompt := st.chat_input("输入问题..."):
        # 显示用户消息
        with st.chat_message("user"):
            st.markdown(prompt)
        chat_service.save_user_message(sid, prompt)
        
        # Agent 响应
        with st.chat_message("assistant"):
            status_box = st.status("Agent 思考中...", expanded=True)
            full_response = ""
            
            for event_type, data in chat_service.stream_agent_response(prompt, sid, pid):
                if event_type == "event":
                    status_box.write(data.description)
                elif event_type == "response":
                    full_response = data
                elif event_type == "error":
                    status_box.update(label="❌ 出错", state="error", expanded=False)
                    st.error(f"Error: {data}")
                    return
            
            status_box.update(label="✅ 完成", state="complete", expanded=False)
            st.markdown(full_response)
            chat_service.save_assistant_message(sid, full_response)


# ==================== Navigation ====================
def _goto_chat(project_id: str):
    """跳转到聊天页面"""
    kb_service = get_kb_service()
    sid = kb_service.get_or_create_session(project_id)
    st.session_state["current_project_id"] = project_id
    st.session_state["current_session_id"] = sid
    st.session_state["view"] = "chat"
    st.rerun()


# ==================== Main ====================
def main():
    st.set_page_config(page_title="RAG Kernel (Test UI)", layout="wide")
    init_app_state()
    
    if st.session_state["view"] == "kb":
        render_kb_page()
    else:
        render_chat_page()


if __name__ == "__main__":
    main()