-- scripts/init_db.sql
-- PostgreSQL 初始化脚本（从 SQLite schema 迁移）

-- 知识库/项目表
CREATE TABLE IF NOT EXISTS projects (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    created_at TEXT
);

-- 会话表（绑定到项目）
CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    name TEXT,
    project_id TEXT,
    created_at TEXT
);

-- 消息表（属于会话）
CREATE TABLE IF NOT EXISTS messages (
    id SERIAL PRIMARY KEY,
    session_id TEXT,
    role TEXT,
    content TEXT,
    created_at TEXT,
    FOREIGN KEY (session_id) REFERENCES sessions(id)
);

-- 知识库文件目录
CREATE TABLE IF NOT EXISTS project_files (
    id SERIAL PRIMARY KEY,
    project_id TEXT NOT NULL,
    source TEXT NOT NULL,
    file_type TEXT,
    chunks_count INTEGER DEFAULT 0,
    created_at TEXT
);

-- 插入默认知识库
INSERT INTO projects (id, name, created_at)
VALUES ('default', '默认知识库', NOW()::text)
ON CONFLICT (id) DO NOTHING;

-- 索引
CREATE INDEX IF NOT EXISTS idx_sessions_project_id ON sessions(project_id);
CREATE INDEX IF NOT EXISTS idx_messages_session_id ON messages(session_id);
CREATE INDEX IF NOT EXISTS idx_project_files_project_id ON project_files(project_id);
