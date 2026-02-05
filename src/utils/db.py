# src/utils/db.py
import os
import sys
import sqlite3
from datetime import datetime
from typing import List, Optional, Tuple, Dict

from config.settings import settings

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

DB_PATH = settings.BASE_DIR / "chat_history.db"

DEFAULT_PROJECT_ID = "default"
DEFAULT_PROJECT_NAME = "默认知识库"

def _now() -> str:
    return datetime.now().isoformat()

def _connect():
    # check_same_thread=False 方便后续可能的多线程/streamlit使用（保守设置）
    return sqlite3.connect(DB_PATH, check_same_thread=False)

def _ensure_column(cursor: sqlite3.Cursor, table: str, column: str, ddl: str):
    """
    如果 table 不存在列 column，则执行 ddl（通常是 ALTER TABLE ADD COLUMN ...）
    """
    cursor.execute(f"PRAGMA table_info({table})")
    cols = [row[1] for row in cursor.fetchall()]
    if column not in cols:
        cursor.execute(ddl)

def init_db():
    conn = _connect()
    cursor = conn.cursor()

    # 1) projects：知识库/数据集
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS projects (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            created_at TEXT
        )
    """)

    # 2) sessions：会话（绑定到 project）
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            id TEXT PRIMARY KEY,
            name TEXT,
            project_id TEXT,
            created_at TEXT
        )
    """)

    # 兼容旧库：补 sessions.project_id
    _ensure_column(
        cursor,
        table="sessions",
        column="project_id",
        ddl="ALTER TABLE sessions ADD COLUMN project_id TEXT"
    )

    # 3) messages：消息（属于 session）
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            role TEXT,
            content TEXT,
            created_at TEXT,
            FOREIGN KEY(session_id) REFERENCES sessions(id)
        )
    """)

    # 4) project_files：知识库文件目录（为“管理页显示更多信息 + 第四步可视化”准备）
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS project_files (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            project_id TEXT NOT NULL,
            source TEXT NOT NULL,              -- 原始文件名/来源（如 report.pdf / repo_url）
            file_type TEXT,                    -- pdf/md/txt/code/url...
            chunks_count INTEGER DEFAULT 0,
            created_at TEXT
        )
    """)

    # 兜底：保证默认知识库存在
    cursor.execute(
        "INSERT OR IGNORE INTO projects (id, name, created_at) VALUES (?, ?, ?)",
        (DEFAULT_PROJECT_ID, DEFAULT_PROJECT_NAME, _now())
    )

    # 给历史会话补默认 project_id
    cursor.execute(
        "UPDATE sessions SET project_id = ? WHERE project_id IS NULL OR project_id = ''",
        (DEFAULT_PROJECT_ID,)
    )

    conn.commit()
    conn.close()

# -------------------------
# Project CRUD
# -------------------------
def create_project(project_id: str, name: str):
    project_id = project_id.strip()
    name = name.strip()
    if not project_id:
        raise ValueError("project_id 不能为空")
    if not name:
        raise ValueError("project name 不能为空")

    conn = _connect()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT OR REPLACE INTO projects (id, name, created_at) VALUES (?, ?, ?)",
        (project_id, name, _now())
    )
    conn.commit()
    conn.close()

def get_all_projects() -> List[Tuple[str, str]]:
    conn = _connect()
    cursor = conn.cursor()
    cursor.execute("SELECT id, name FROM projects ORDER BY created_at DESC")
    rows = cursor.fetchall()
    conn.close()
    return rows

def get_project(project_id: str) -> Optional[Tuple[str, str]]:
    conn = _connect()
    cursor = conn.cursor()
    cursor.execute("SELECT id, name FROM projects WHERE id = ?", (project_id,))
    row = cursor.fetchone()
    conn.close()
    return row

def delete_project(project_id: str):
    if project_id == DEFAULT_PROJECT_ID:
        raise ValueError("默认知识库不允许删除")

    conn = _connect()
    cursor = conn.cursor()

    # 删掉该 project 下的 sessions/messages
    cursor.execute("SELECT id FROM sessions WHERE project_id = ?", (project_id,))
    session_ids = [r[0] for r in cursor.fetchall()]

    for sid in session_ids:
        cursor.execute("DELETE FROM messages WHERE session_id = ?", (sid,))
        cursor.execute("DELETE FROM sessions WHERE id = ?", (sid,))

    # 删掉该 project 的文件目录记录
    cursor.execute("DELETE FROM project_files WHERE project_id = ?", (project_id,))

    # 删 project
    cursor.execute("DELETE FROM projects WHERE id = ?", (project_id,))

    conn.commit()
    conn.close()

# -------------------------
# Session CRUD
# -------------------------
def create_session(session_id: str, name: str, project_id: str = DEFAULT_PROJECT_ID):
    session_id = session_id.strip()
    name = (name or "").strip()
    project_id = (project_id or DEFAULT_PROJECT_ID).strip()

    if not session_id:
        raise ValueError("session_id 不能为空")
    if not name:
        name = "未命名会话"

    conn = _connect()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT OR REPLACE INTO sessions (id, name, project_id, created_at) VALUES (?, ?, ?, ?)",
        (session_id, name, project_id, _now())
    )
    conn.commit()
    conn.close()

def get_all_sessions() -> List[Tuple[str, str]]:
    conn = _connect()
    cursor = conn.cursor()
    cursor.execute("SELECT id, name FROM sessions ORDER BY created_at DESC")
    rows = cursor.fetchall()
    conn.close()
    return rows

def get_sessions_by_project(project_id: str) -> List[Tuple[str, str]]:
    conn = _connect()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT id, name FROM sessions WHERE project_id = ? ORDER BY created_at DESC",
        (project_id,)
    )
    rows = cursor.fetchall()
    conn.close()
    return rows

def get_latest_session_by_project(project_id: str) -> Optional[Tuple[str, str]]:
    conn = _connect()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT id, name FROM sessions WHERE project_id = ? ORDER BY created_at DESC LIMIT 1",
        (project_id,)
    )
    row = cursor.fetchone()
    conn.close()
    return row

def get_session_project_id(session_id: str) -> str:
    conn = _connect()
    cursor = conn.cursor()
    cursor.execute("SELECT project_id FROM sessions WHERE id = ?", (session_id,))
    row = cursor.fetchone()
    conn.close()
    if not row or not row[0]:
        return DEFAULT_PROJECT_ID
    return row[0]

def set_session_project_id(session_id: str, project_id: str):
    conn = _connect()
    cursor = conn.cursor()
    cursor.execute(
        "UPDATE sessions SET project_id = ? WHERE id = ?",
        (project_id, session_id)
    )
    conn.commit()
    conn.close()

def delete_session(session_id: str):
    conn = _connect()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
    cursor.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
    conn.commit()
    conn.close()

# -------------------------
# Messages
# -------------------------
def save_message(session_id: str, role: str, content: str):
    conn = _connect()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO messages (session_id, role, content, created_at) VALUES (?, ?, ?, ?)",
        (session_id, role, content, _now())
    )
    conn.commit()
    conn.close()

def get_messages(session_id: str) -> List[Dict[str, str]]:
    conn = _connect()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT role, content FROM messages WHERE session_id = ? ORDER BY created_at",
        (session_id,)
    )
    rows = cursor.fetchall()
    conn.close()
    return [{"role": role, "content": content} for role, content in rows]

# -------------------------
# Project Files Catalog (for UI)
# -------------------------
def add_project_file_record(
    project_id: str,
    source: str,
    file_type: str = "",
    chunks_count: int = 0
):
    """
    每次入库后可以写入一条目录记录。
    source: 原始文件名/URL/仓库名
    """
    conn = _connect()
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO project_files (project_id, source, file_type, chunks_count, created_at)
        VALUES (?, ?, ?, ?, ?)
        """,
        (project_id, source, file_type, int(chunks_count), _now())
    )
    conn.commit()
    conn.close()

def list_project_files(project_id: str) -> List[Tuple[int, str, str, int, str]]:
    """
    返回: (id, source, file_type, chunks_count, created_at)
    """
    conn = _connect()
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT id, source, file_type, chunks_count, created_at
        FROM project_files
        WHERE project_id = ?
        ORDER BY created_at DESC
        """,
        (project_id,)
    )
    rows = cursor.fetchall()
    conn.close()
    return rows

def delete_project_file_record(file_id: int):
    conn = _connect()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM project_files WHERE id = ?", (file_id,))
    conn.commit()
    conn.close()

# -------------------------
# Project Stats (for KB page)
# -------------------------
def get_project_stats(project_id: str) -> Dict[str, object]:
    """
    给知识库管理页用：会话数、消息数、文件数、最近会话时间
    """
    conn = _connect()
    cursor = conn.cursor()

    # sessions count + latest session time
    cursor.execute(
        "SELECT COUNT(*), MAX(created_at) FROM sessions WHERE project_id = ?",
        (project_id,)
    )
    session_count, latest_session_time = cursor.fetchone()

    # messages count (join)
    cursor.execute(
        """
        SELECT COUNT(*)
        FROM messages m
        JOIN sessions s ON m.session_id = s.id
        WHERE s.project_id = ?
        """,
        (project_id,)
    )
    (message_count,) = cursor.fetchone()

    # files count
    cursor.execute(
        "SELECT COUNT(*), MAX(created_at) FROM project_files WHERE project_id = ?",
        (project_id,)
    )
    file_count, latest_file_time = cursor.fetchone()

    conn.close()

    return {
        "session_count": session_count or 0,
        "message_count": message_count or 0,
        "file_count": file_count or 0,
        "latest_session_time": latest_session_time,
        "latest_file_time": latest_file_time,
    }
