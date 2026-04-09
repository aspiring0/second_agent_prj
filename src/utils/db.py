# src/utils/db.py
"""
数据库模块 - 支持 SQLite（默认）和 PostgreSQL（Docker 环境）
通过 POSTGRES_HOST 环境变量控制：
- 未配置或为空 → 使用 SQLite（向后兼容）
- 已配置 → 使用 PostgreSQL（连接池）
"""
import os
import sys
from datetime import datetime
from typing import List, Optional, Tuple, Dict

from config.settings import settings

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

DEFAULT_PROJECT_ID = "default"
DEFAULT_PROJECT_NAME = "默认知识库"

# ==================== 后端选择 ====================

_USE_POSTGRES = bool(getattr(settings, 'POSTGRES_HOST', '')
                     and getattr(settings, 'POSTGRES_PASSWORD', ''))

# PostgreSQL 连接池
_pg_pool = None


def _now() -> str:
    return datetime.now().isoformat()


# ==================== SQLite 后端 ====================

DB_PATH = settings.BASE_DIR / "chat_history.db"


def _sqlite_connect():
    return __import__('sqlite3').connect(DB_PATH, check_same_thread=False)


def _ensure_column_sqlite(cursor, table: str, column: str, ddl: str):
    """如果表不存在某列，则添加"""
    cursor.execute(f"PRAGMA table_info({table})")
    cols = [row[1] for row in cursor.fetchall()]
    if column not in cols:
        cursor.execute(ddl)


# ==================== PostgreSQL 后端 ====================

def _pg_get_conn():
    """从连接池获取连接"""
    global _pg_pool
    if _pg_pool is None:
        try:
            import psycopg2
            from psycopg2 import pool
            _pg_pool = pool.ThreadedConnectionPool(
                2, 10,
                host=settings.POSTGRES_HOST,
                port=settings.POSTGRES_PORT,
                dbname=settings.POSTGRES_DB,
                user=settings.POSTGRES_USER,
                password=settings.POSTGRES_PASSWORD
            )
        except ImportError:
            raise ImportError("请安装 psycopg2-binary: pip install psycopg2-binary")
    return _pg_pool.getconn()


def _pg_return_conn(conn):
    """归还连接到连接池"""
    global _pg_pool
    if _pg_pool and conn:
        _pg_pool.putconn(conn)


# ==================== 统一接口 ====================

def _connect():
    """获取数据库连接（自动选择后端）"""
    if _USE_POSTGRES:
        return _pg_get_conn()
    return _sqlite_connect()


def _close(conn):
    """关闭/归还连接"""
    if _USE_POSTGRES:
        _pg_return_conn(conn)
    else:
        conn.close()


def _placeholder(n: int) -> str:
    """返回占位符：SQLite 用 ?，PostgreSQL 用 %s"""
    if _USE_POSTGRES:
        return ", ".join(["%s"] * n)
    return ", ".join(["?"] * n)


def _ph() -> str:
    """单个占位符"""
    return "%s" if _USE_POSTGRES else "?"


# ==================== 初始化 ====================

def init_db():
    conn = _connect()
    cursor = conn.cursor()

    if _USE_POSTGRES:
        # PostgreSQL: 使用 SERIAL 和 %s 占位符
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS projects (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                created_at TEXT
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                name TEXT,
                project_id TEXT,
                created_at TEXT
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id SERIAL PRIMARY KEY,
                session_id TEXT,
                role TEXT,
                content TEXT,
                created_at TEXT,
                FOREIGN KEY (session_id) REFERENCES sessions(id)
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS project_files (
                id SERIAL PRIMARY KEY,
                project_id TEXT NOT NULL,
                source TEXT NOT NULL,
                file_type TEXT,
                chunks_count INTEGER DEFAULT 0,
                created_at TEXT
            )
        """)

        cursor.execute(
            f"INSERT INTO projects (id, name, created_at) VALUES ({_ph()}, {_ph()}, {_ph()}) "
            f"ON CONFLICT (id) DO NOTHING",
            (DEFAULT_PROJECT_ID, DEFAULT_PROJECT_NAME, _now())
        )

        cursor.execute(
            f"UPDATE sessions SET project_id = {_ph()} WHERE project_id IS NULL OR project_id = ''",
            (DEFAULT_PROJECT_ID,)
        )
    else:
        # SQLite: 原有逻辑
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS projects (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                created_at TEXT
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                name TEXT,
                project_id TEXT,
                created_at TEXT
            )
        """)

        _ensure_column_sqlite(
            cursor,
            table="sessions",
            column="project_id",
            ddl="ALTER TABLE sessions ADD COLUMN project_id TEXT"
        )

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

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS project_files (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                project_id TEXT NOT NULL,
                source TEXT NOT NULL,
                file_type TEXT,
                chunks_count INTEGER DEFAULT 0,
                created_at TEXT
            )
        """)

        cursor.execute(
            "INSERT OR IGNORE INTO projects (id, name, created_at) VALUES (?, ?, ?)",
            (DEFAULT_PROJECT_ID, DEFAULT_PROJECT_NAME, _now())
        )

        cursor.execute(
            "UPDATE sessions SET project_id = ? WHERE project_id IS NULL OR project_id = ''",
            (DEFAULT_PROJECT_ID,)
        )

    conn.commit()
    _close(conn)


# ==================== Project CRUD ====================

def create_project(project_id: str, name: str):
    project_id = project_id.strip()
    name = name.strip()
    if not project_id:
        raise ValueError("project_id 不能为空")
    if not name:
        raise ValueError("project name 不能为空")

    conn = _connect()
    cursor = conn.cursor()

    if _USE_POSTGRES:
        cursor.execute(
            f"INSERT INTO projects (id, name, created_at) VALUES ({_ph()}, {_ph()}, {_ph()}) "
            f"ON CONFLICT (id) DO UPDATE SET name = EXCLUDED.name, created_at = EXCLUDED.created_at",
            (project_id, name, _now())
        )
    else:
        cursor.execute(
            "INSERT OR REPLACE INTO projects (id, name, created_at) VALUES (?, ?, ?)",
            (project_id, name, _now())
        )

    conn.commit()
    _close(conn)


def get_all_projects() -> List[Tuple[str, str]]:
    conn = _connect()
    cursor = conn.cursor()
    cursor.execute("SELECT id, name FROM projects ORDER BY created_at DESC")
    rows = cursor.fetchall()
    _close(conn)
    return rows


def get_project(project_id: str) -> Optional[Tuple[str, str]]:
    conn = _connect()
    cursor = conn.cursor()
    cursor.execute(f"SELECT id, name FROM projects WHERE id = {_ph()}", (project_id,))
    row = cursor.fetchone()
    _close(conn)
    return row


def delete_project(project_id: str):
    if project_id == DEFAULT_PROJECT_ID:
        raise ValueError("默认知识库不允许删除")

    conn = _connect()
    cursor = conn.cursor()

    cursor.execute(f"SELECT id FROM sessions WHERE project_id = {_ph()}", (project_id,))
    session_ids = [r[0] for r in cursor.fetchall()]

    for sid in session_ids:
        cursor.execute(f"DELETE FROM messages WHERE session_id = {_ph()}", (sid,))
        cursor.execute(f"DELETE FROM sessions WHERE id = {_ph()}", (sid,))

    cursor.execute(f"DELETE FROM project_files WHERE project_id = {_ph()}", (project_id,))
    cursor.execute(f"DELETE FROM projects WHERE id = {_ph()}", (project_id,))

    conn.commit()
    _close(conn)


# ==================== Session CRUD ====================

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

    if _USE_POSTGRES:
        cursor.execute(
            f"INSERT INTO sessions (id, name, project_id, created_at) VALUES ({_ph()}, {_ph()}, {_ph()}, {_ph()}) "
            f"ON CONFLICT (id) DO UPDATE SET name = EXCLUDED.name, project_id = EXCLUDED.project_id",
            (session_id, name, project_id, _now())
        )
    else:
        cursor.execute(
            "INSERT OR REPLACE INTO sessions (id, name, project_id, created_at) VALUES (?, ?, ?, ?)",
            (session_id, name, project_id, _now())
        )

    conn.commit()
    _close(conn)


def get_all_sessions() -> List[Tuple[str, str]]:
    conn = _connect()
    cursor = conn.cursor()
    cursor.execute("SELECT id, name FROM sessions ORDER BY created_at DESC")
    rows = cursor.fetchall()
    _close(conn)
    return rows


def get_sessions_by_project(project_id: str) -> List[Tuple[str, str]]:
    conn = _connect()
    cursor = conn.cursor()
    cursor.execute(
        f"SELECT id, name FROM sessions WHERE project_id = {_ph()} ORDER BY created_at DESC",
        (project_id,)
    )
    rows = cursor.fetchall()
    _close(conn)
    return rows


def get_latest_session_by_project(project_id: str) -> Optional[Tuple[str, str]]:
    conn = _connect()
    cursor = conn.cursor()
    cursor.execute(
        f"SELECT id, name FROM sessions WHERE project_id = {_ph()} ORDER BY created_at DESC LIMIT 1",
        (project_id,)
    )
    row = cursor.fetchone()
    _close(conn)
    return row


def get_session_project_id(session_id: str) -> str:
    conn = _connect()
    cursor = conn.cursor()
    cursor.execute(f"SELECT project_id FROM sessions WHERE id = {_ph()}", (session_id,))
    row = cursor.fetchone()
    _close(conn)
    if not row or not row[0]:
        return DEFAULT_PROJECT_ID
    return row[0]


def set_session_project_id(session_id: str, project_id: str):
    conn = _connect()
    cursor = conn.cursor()
    cursor.execute(
        f"UPDATE sessions SET project_id = {_ph()} WHERE id = {_ph()}",
        (project_id, session_id)
    )
    conn.commit()
    _close(conn)


def delete_session(session_id: str):
    conn = _connect()
    cursor = conn.cursor()
    cursor.execute(f"DELETE FROM messages WHERE session_id = {_ph()}", (session_id,))
    cursor.execute(f"DELETE FROM sessions WHERE id = {_ph()}", (session_id,))
    conn.commit()
    _close(conn)


# ==================== Messages ====================

def save_message(session_id: str, role: str, content: str):
    conn = _connect()
    cursor = conn.cursor()
    cursor.execute(
        f"INSERT INTO messages (session_id, role, content, created_at) VALUES ({_ph()}, {_ph()}, {_ph()}, {_ph()})",
        (session_id, role, content, _now())
    )
    conn.commit()
    _close(conn)


def get_messages(session_id: str) -> List[Dict[str, str]]:
    conn = _connect()
    cursor = conn.cursor()
    cursor.execute(
        f"SELECT role, content FROM messages WHERE session_id = {_ph()} ORDER BY created_at",
        (session_id,)
    )
    rows = cursor.fetchall()
    _close(conn)
    return [{"role": role, "content": content} for role, content in rows]


# ==================== Project Files ====================

def add_project_file_record(
    project_id: str,
    source: str,
    file_type: str = "",
    chunks_count: int = 0
):
    conn = _connect()
    cursor = conn.cursor()
    cursor.execute(
        f"INSERT INTO project_files (project_id, source, file_type, chunks_count, created_at) "
        f"VALUES ({_ph()}, {_ph()}, {_ph()}, {_ph()}, {_ph()})",
        (project_id, source, file_type, int(chunks_count), _now())
    )
    conn.commit()
    _close(conn)


def list_project_files(project_id: str) -> List[Tuple[int, str, str, int, str]]:
    conn = _connect()
    cursor = conn.cursor()
    cursor.execute(
        f"SELECT id, source, file_type, chunks_count, created_at "
        f"FROM project_files WHERE project_id = {_ph()} ORDER BY created_at DESC",
        (project_id,)
    )
    rows = cursor.fetchall()
    _close(conn)
    return rows


def delete_project_file_record(file_id: int):
    conn = _connect()
    cursor = conn.cursor()
    cursor.execute(f"DELETE FROM project_files WHERE id = {_ph()}", (file_id,))
    conn.commit()
    _close(conn)


# ==================== Project Stats ====================

def get_project_stats(project_id: str) -> Dict[str, object]:
    conn = _connect()
    cursor = conn.cursor()

    cursor.execute(
        f"SELECT COUNT(*), MAX(created_at) FROM sessions WHERE project_id = {_ph()}",
        (project_id,)
    )
    session_count, latest_session_time = cursor.fetchone()

    cursor.execute(
        f"SELECT COUNT(*) FROM messages m "
        f"JOIN sessions s ON m.session_id = s.id WHERE s.project_id = {_ph()}",
        (project_id,)
    )
    (message_count,) = cursor.fetchone()

    cursor.execute(
        f"SELECT COUNT(*), MAX(created_at) FROM project_files WHERE project_id = {_ph()}",
        (project_id,)
    )
    file_count, latest_file_time = cursor.fetchone()

    _close(conn)

    return {
        "session_count": session_count or 0,
        "message_count": message_count or 0,
        "file_count": file_count or 0,
        "latest_session_time": latest_session_time,
        "latest_file_time": latest_file_time,
    }
