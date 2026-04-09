"""
Web Application - 前端展示层 (Frontend)
只负责 UI 渲染和用户交互，所有业务逻辑由 Service 层处理
"""
import os
from pathlib import Path

import streamlit as st

# Service Layer
from src.service import KnowledgeBaseService, ChatService, DocumentService
from src.utils.model_manager import model_manager
from config.settings import settings

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


def render_model_selector():
    """渲染模型选择器"""
    st.subheader("🤖 模型配置")
    
    # 获取当前模型状态
    status = model_manager.get_model_status()
    
    # Chat模型选择
    chat_models = model_manager.list_chat_models()
    current_chat = model_manager.get_current_chat_model_id()
    
    chat_options = {m.id: f"{m.name} ({m.provider.value})" for m in chat_models}
    chat_ids = list(chat_options.keys())
    
    # 检查当前模型是否在列表中
    if current_chat not in chat_ids:
        current_chat = chat_ids[0] if chat_ids else None
    
    selected_chat = st.selectbox(
        "对话模型",
        options=chat_ids,
        format_func=lambda x: chat_options.get(x, x),
        index=chat_ids.index(current_chat) if current_chat in chat_ids else 0,
        key="chat_model_selector"
    )
    
    if selected_chat and selected_chat != current_chat:
        success = model_manager.set_current_chat_model(selected_chat)
        if success:
            st.success(f"已切换到 {chat_options.get(selected_chat, selected_chat)}")
            st.rerun()
        else:
            st.error("切换模型失败，请检查API Key配置")
    
    # Embedding模型选择
    embedding_models = model_manager.list_embedding_models()
    current_embedding = model_manager.get_current_embedding_model_id()
    
    embedding_options = {m.id: f"{m.name} ({m.dimension}D)" for m in embedding_models}
    embedding_ids = list(embedding_options.keys())
    
    if current_embedding not in embedding_ids:
        current_embedding = embedding_ids[0] if embedding_ids else None
    
    selected_embedding = st.selectbox(
        "Embedding模型",
        options=embedding_ids,
        format_func=lambda x: embedding_options.get(x, x),
        index=embedding_ids.index(current_embedding) if current_embedding in embedding_ids else 0,
        key="embedding_model_selector"
    )
    
    if selected_embedding and selected_embedding != current_embedding:
        success = model_manager.set_current_embedding_model(selected_embedding)
        if success:
            st.success(f"已切换到 {embedding_options.get(selected_embedding, selected_embedding)}")
            st.rerun()
        else:
            st.error("切换模型失败，请检查API Key配置")
    
    # 显示当前模型信息
    with st.expander("模型状态", expanded=False):
        st.json(status)


# ==================== Settings Panel ====================
def _save_to_env_file(api_key: str, base_url: str, model_id: str) -> bool:
    """
    将配置写入 .env 文件（追加或更新对应行）

    Args:
        api_key: API Key
        base_url: API Base URL
        model_id: 模型 ID

    Returns:
        是否保存成功
    """
    env_path = Path(__file__).resolve().parent / ".env"
    lines: list[str] = []

    # 读取现有内容
    if env_path.exists():
        with open(env_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

    # 需要更新的键值对
    updates = {
        "OPENAI_API_KEY": api_key,
        "OPENAI_API_BASE": base_url,
        "CHAT_MODEL": model_id,
    }

    # 逐行更新或标记
    updated_keys: set[str] = set()
    for i, line in enumerate(lines):
        stripped = line.strip()
        for key, value in updates.items():
            if stripped.startswith(f"{key}="):
                lines[i] = f"{key}={value}\n"
                updated_keys.add(key)
                break

    # 追加未找到的键
    for key, value in updates.items():
        if key not in updated_keys:
            lines.append(f"{key}={value}\n")

    try:
        with open(env_path, "w", encoding="utf-8") as f:
            f.writelines(lines)
        return True
    except Exception as e:
        st.error(f"写入 .env 文件失败: {e}")
        return False


def render_settings_panel():
    """
    渲染侧边栏"设置"折叠区
    包含 API Key 输入、Base URL 配置、模型选择、测试连接、保存配置
    """
    # 从 session_state 读取或初始化默认值
    current_api_key = os.getenv("OPENAI_API_KEY", "")
    current_base_url = os.getenv("OPENAI_API_BASE", settings.OPENAI_BASE_URL or "")

    if "settings_api_key" not in st.session_state:
        st.session_state["settings_api_key"] = current_api_key
    if "settings_base_url" not in st.session_state:
        st.session_state["settings_base_url"] = current_base_url
    if "settings_model_id" not in st.session_state:
        st.session_state["settings_model_id"] = model_manager.get_current_chat_model_id()

    with st.expander("设置", expanded=False):
        # 1. API Key 输入框（密码模式）
        api_key_hint = ""
        if current_api_key and len(current_api_key) >= 8:
            api_key_hint = f"当前: ...{current_api_key[-8:]}"
        elif current_api_key:
            api_key_hint = f"当前: {current_api_key[:3]}***"

        input_api_key = st.text_input(
            "API Key",
            value=st.session_state["settings_api_key"],
            type="password",
            help=f"输入 OpenAI API Key。{api_key_hint}",
            key="settings_api_key_input",
        )

        # 2. API Base URL 输入框
        input_base_url = st.text_input(
            "API Base URL",
            value=st.session_state["settings_base_url"],
            help="留空则使用 OpenAI 官方地址",
            key="settings_base_url_input",
        )

        # 3. 模型下拉选择
        available_models = model_manager.get_available_models()
        model_options = {m["id"]: f"{m['name']} ({m['provider']})" for m in available_models}
        model_ids = list(model_options.keys())

        current_model = st.session_state["settings_model_id"]
        if current_model not in model_ids:
            current_model = model_ids[0] if model_ids else ""

        selected_model = st.selectbox(
            "默认对话模型",
            options=model_ids,
            format_func=lambda x: model_options.get(x, x),
            index=model_ids.index(current_model) if current_model in model_ids else 0,
            key="settings_model_selector",
        )
        st.session_state["settings_model_id"] = selected_model

        # 4. 测试连接按钮
        if st.button("测试连接", key="settings_test_connection"):
            test_key = input_api_key or current_api_key
            test_url = input_base_url or current_base_url
            if not test_key:
                st.warning("请先输入 API Key")
            else:
                with st.spinner("正在测试连接..."):
                    success, message = model_manager.test_connection(
                        api_key=test_key,
                        base_url=test_url if test_url else None,
                        model_name=selected_model,
                    )
                if success:
                    st.success(message)
                else:
                    st.error(message)

        # 5. 保存配置按钮
        if st.button("保存配置", key="settings_save_config"):
            save_key = input_api_key or current_api_key
            save_url = input_base_url or current_base_url
            save_model = selected_model

            if not save_key:
                st.warning("API Key 不能为空")
                return

            # 更新内存中的 API Key 和 Base URL
            ok = model_manager.update_api_key(save_key, save_url)
            if not ok:
                st.error("更新 API Key 失败")
                return

            # 切换模型
            model_manager.set_current_chat_model(save_model)

            # 写入 .env 文件
            env_ok = _save_to_env_file(save_key, save_url, save_model)
            if env_ok:
                # 同步 session_state
                st.session_state["settings_api_key"] = save_key
                st.session_state["settings_base_url"] = save_url
                st.session_state["settings_model_id"] = save_model
                st.success("配置已保存并生效")
            else:
                st.warning("内存中已更新，但 .env 文件写入失败")


def render_chat_sidebar(kb_service, doc_service, pid, sid):
    """渲染聊天侧边栏"""
    st.header("🧭 控制台")
    
    if st.button("⬅️ 切换知识库"):
        st.session_state["view"] = "kb"
        st.rerun()
    
    st.divider()
    
    # 模型选择器（如果启用）
    if settings.ENABLE_MODEL_SWITCHING:
        render_model_selector()
        st.divider()

    # 设置面板
    render_settings_panel()
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