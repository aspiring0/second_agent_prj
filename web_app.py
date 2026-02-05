import streamlit as st
import uuid
from pathlib import Path
from collections import Counter

from langchain_core.messages import HumanMessage

from src.agent.graph import app as agent_app
from src.rag.etl import ContentProcessor
from src.rag.vectorstore import VectorDBManager

from src.utils.db import (
    init_db,
    create_project, get_all_projects, delete_project,
    create_session,
    get_messages, save_message,
    get_sessions_by_project, get_latest_session_by_project,
    # Stage 3 additions:
    get_project_stats,
    add_project_file_record, list_project_files
)

# -------------------------
# Init
# -------------------------
init_db()
st.set_page_config(page_title="RAG Kernel (Test UI)", layout="wide")

# App state (two views)
if "view" not in st.session_state:
    st.session_state["view"] = "kb"  # kb / chat
if "current_project_id" not in st.session_state:
    st.session_state["current_project_id"] = None
if "current_session_id" not in st.session_state:
    st.session_state["current_session_id"] = None


# -------------------------
# Helpers
# -------------------------
def _ensure_default_project():
    projects = get_all_projects()
    if not projects:
        create_project("default", "默认知识库")
        projects = get_all_projects()
    return projects

def _project_maps(projects):
    project_map = {pid: name for pid, name in projects}
    project_ids = [pid for pid, _ in projects]
    return project_map, project_ids

def _fmt_project(pid: str, project_map: dict) -> str:
    name = project_map.get(pid, pid)
    return f"{name} ({pid})"

def _goto_chat(project_id: str):
    latest = get_latest_session_by_project(project_id)
    if latest:
        st.session_state["current_session_id"] = latest[0]
    else:
        sid = str(uuid.uuid4())
        create_session(sid, "默认会话", project_id=project_id)
        st.session_state["current_session_id"] = sid

    st.session_state["current_project_id"] = project_id
    st.session_state["view"] = "chat"
    st.rerun()


# -------------------------
# Page: Knowledge Base
# -------------------------
def kb_page():
    st.title("📚 知识库管理")

    projects = _ensure_default_project()
    project_map, project_ids = _project_maps(projects)

    # Choose KB
    if st.session_state["current_project_id"] is None:
        st.session_state["current_project_id"] = "default" if "default" in project_ids else project_ids[0]

    selected = st.selectbox(
        "选择知识库",
        options=project_ids,
        format_func=lambda x: _fmt_project(x, project_map),
        index=project_ids.index(st.session_state["current_project_id"])
        if st.session_state["current_project_id"] in project_ids else 0
    )
    st.session_state["current_project_id"] = selected

    # --- Stats table (Stage 3)
    st.subheader("知识库概览")
    rows = []
    for pid in project_ids:
        stats = get_project_stats(pid)
        rows.append({
            "知识库": project_map.get(pid, pid),
            "ID": pid,
            "文件数": stats.get("file_count", 0),
            "会话数": stats.get("session_count", 0),
            "消息数": stats.get("message_count", 0),
            "最近入库": stats.get("latest_file_time"),
            "最近会话": stats.get("latest_session_time"),
        })
    st.dataframe(rows, use_container_width=True)

    col1, col2, col3 = st.columns(3)

    # Create KB
    with col1:
        st.subheader("新建")
        new_name = st.text_input("名称", value="")
        new_id = st.text_input("ID（建议英文，如 kb_code）", value="")
        if st.button("创建知识库"):
            if not new_id.strip() or not new_name.strip():
                st.warning("请填写名称与ID")
            else:
                create_project(new_id.strip(), new_name.strip())
                st.session_state["current_project_id"] = new_id.strip()
                st.success("创建成功")
                st.rerun()

    # Browse files in KB (Stage 3)
    with col2:
        st.subheader("浏览")
        pid = st.session_state["current_project_id"]
        st.write(f"当前选择：{_fmt_project(pid, project_map)}")
        files = list_project_files(pid)
        if not files:
            st.caption("该知识库暂无文件目录记录（上传入库后会出现）")
        else:
            # files: (id, source, file_type, chunks_count, created_at)
            preview = [{
                "文件": f[1],
                "类型": f[2],
                "chunks": f[3],
                "入库时间": f[4],
            } for f in files[:10]]
            st.dataframe(preview, use_container_width=True, height=260)

    # Delete KB
    with col3:
        st.subheader("删除")
        deletable = [pid for pid in project_ids if pid != "default"]
        del_id = st.selectbox(
            "选择要删除的知识库",
            options=deletable,
            format_func=lambda x: _fmt_project(x, project_map),
        ) if deletable else None

        if del_id and st.button("确认删除"):
            try:
                delete_project(del_id)
                if st.session_state["current_project_id"] == del_id:
                    st.session_state["current_project_id"] = "default"
                st.success("删除成功（会话与目录记录已删；向量清理将在后续阶段补齐）")
                st.rerun()
            except Exception as e:
                st.error(f"删除失败: {e}")

    st.divider()

    if st.button("进入聊天"):
        _goto_chat(st.session_state["current_project_id"])


# -------------------------
# Page: Chat
# -------------------------
def chat_page():
    pid = st.session_state["current_project_id"]
    sid = st.session_state["current_session_id"]

    if not pid:
        st.session_state["view"] = "kb"
        st.rerun()

    # Ensure a session exists
    if not sid:
        latest = get_latest_session_by_project(pid)
        if latest:
            sid = latest[0]
            st.session_state["current_session_id"] = sid
        else:
            sid = str(uuid.uuid4())
            create_session(sid, "默认会话", project_id=pid)
            st.session_state["current_session_id"] = sid

    st.title("💬 聊天")
    st.caption(f"当前知识库：{pid} ｜ 当前会话：{sid[:8]}")

    # Sidebar controls
    with st.sidebar:
        st.header("🧭 控制台")

        if st.button("⬅️ 切换知识库"):
            st.session_state["view"] = "kb"
            st.rerun()

        st.divider()

        # Session list for current project
        st.subheader("会话")
        sessions = get_sessions_by_project(pid)
        session_map = {s[0]: s[1] for s in sessions}  # (id, name)
        session_ids = [s[0] for s in sessions]

        # Defensive: if current sid not in current project, open latest/create
        if sid not in session_ids:
            latest = get_latest_session_by_project(pid)
            if latest:
                sid = latest[0]
                st.session_state["current_session_id"] = sid
            else:
                sid = str(uuid.uuid4())
                create_session(sid, "默认会话", project_id=pid)
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

        # New session
        new_title = st.text_input("新会话标题", value="")
        if st.button("➕ 新建会话"):
            if not new_title.strip():
                st.warning("请输入标题")
            else:
                new_id = str(uuid.uuid4())
                create_session(new_id, new_title.strip(), project_id=pid)
                st.session_state["current_session_id"] = new_id
                st.rerun()

        st.divider()

        # Upload & ingest
        st.subheader("上传文件（写入当前知识库）")
        uploaded_files = st.file_uploader("上传文档", accept_multiple_files=True)

        if st.button("🚀 更新知识库"):
            if not uploaded_files:
                st.warning("请先上传文件")
            else:
                status = st.empty()
                status.info(f"正在处理 {len(uploaded_files)} 个文件...")

                try:
                    processor = ContentProcessor()
                    docs = processor.load_uploaded_files(uploaded_files)
                    if not docs:
                        status.warning("未解析出有效内容")
                    else:
                        chunks = processor.split_documents(docs)

                        # 1) write vectors (append)
                        VectorDBManager().create_vector_db(chunks, project_id=pid)

                        # 2) write catalog (Stage 3)
                        # precise count by doc.metadata["source"]
                        src_counter = Counter()
                        for c in chunks:
                            src = (c.metadata or {}).get("source", "unknown")
                            src_counter[src] += 1

                        for f in uploaded_files:
                            suffix = Path(f.name).suffix.lower().lstrip(".")
                            add_project_file_record(
                                project_id=pid,
                                source=f.name,
                                file_type=suffix,
                                chunks_count=src_counter.get(f.name, 0)
                            )

                        status.success(f"✅ 入库成功：新增 {len(chunks)} chunks；目录已更新。")

                except Exception as e:
                    status.error(f"❌ 入库失败: {e}")

        st.divider()

        # Catalog preview in chat (Stage 3)
        with st.expander("📄 当前知识库文件目录", expanded=False):
            files = list_project_files(pid)
            if not files:
                st.caption("暂无目录记录（上传后会出现）")
            else:
                table = [{
                    "文件": f[1],
                    "类型": f[2],
                    "chunks": f[3],
                    "入库时间": f[4],
                } for f in files[:50]]
                st.dataframe(table, use_container_width=True, height=260)

    # Main chat history
    sid = st.session_state["current_session_id"]
    db_messages = get_messages(sid)
    for msg in db_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Input + agent
    if prompt := st.chat_input("输入问题..."):
        with st.chat_message("user"):
            st.markdown(prompt)
        save_message(sid, "user", prompt)

        with st.chat_message("assistant"):
            status_box = st.status("Agent 思考中...", expanded=True)
            full_response = ""

            inputs = {"messages": [HumanMessage(content=prompt)]}
            run_config = {"configurable": {"session_id": sid, "project_id": pid}}

            try:
                for event in agent_app.stream(inputs, config=run_config):
                    for node_name, node_output in event.items():
                        if node_name == "researcher":
                            status_box.write("🔍 研究员: 分析需求...")
                        elif node_name == "tools":
                            status_box.write("📚 工具: 检索知识库资料...")
                        elif node_name == "writer":
                            status_box.write("✍️ 作家: 整理回答...")
                            full_response = node_output["messages"][-1].content

                status_box.update(label="✅ 完成", state="complete", expanded=False)
                st.markdown(full_response)
                save_message(sid, "assistant", full_response)

            except Exception as e:
                st.error(f"Error: {e}")


# -------------------------
# Router
# -------------------------
if st.session_state["view"] == "kb":
    kb_page()
else:
    chat_page()
