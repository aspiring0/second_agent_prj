# src/service/kb_service.py
"""
知识库服务 - Knowledge Base Service
负责知识库的 CRUD 操作和状态管理
"""
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

from src.utils.db import (
    init_db,
    create_project,
    get_all_projects,
    delete_project,
    get_project_stats,
    list_project_files,
    get_sessions_by_project,
    get_latest_session_by_project,
    create_session,
)
from src.utils.logger import setup_logger

logger = setup_logger("KB_SERVICE")


@dataclass
class KnowledgeBase:
    """知识库数据模型"""
    id: str
    name: str
    
    @classmethod
    def from_tuple(cls, data: Tuple[str, str]) -> "KnowledgeBase":
        return cls(id=data[0], name=data[1])


@dataclass
class KnowledgeBaseStats:
    """知识库统计数据"""
    file_count: int
    session_count: int
    message_count: int
    latest_file_time: Optional[str]
    latest_session_time: Optional[str]


@dataclass
class FileRecord:
    """文件记录"""
    id: int
    source: str
    file_type: str
    chunks_count: int
    created_at: str


class KnowledgeBaseService:
    """知识库管理服务"""
    
    _initialized = False
    
    def __init__(self):
        # 确保数据库只初始化一次
        if not KnowledgeBaseService._initialized:
            init_db()
            KnowledgeBaseService._initialized = True
            logger.info("✅ 知识库服务初始化完成")
    
    def get_all_kbs(self) -> List[KnowledgeBase]:
        """获取所有知识库"""
        projects = get_all_projects()
        return [KnowledgeBase.from_tuple(p) for p in projects]
    
    def ensure_default_kb(self) -> List[KnowledgeBase]:
        """确保至少有一个默认知识库"""
        projects = self.get_all_kbs()
        if not projects:
            logger.info("📝 创建默认知识库")
            create_project("default", "默认知识库")
            projects = self.get_all_kbs()
        return projects
    
    def create_kb(self, kb_id: str, name: str) -> bool:
        """创建新知识库"""
        try:
            create_project(kb_id.strip(), name.strip())
            logger.info(f"✅ 创建知识库: {name} ({kb_id})")
            return True
        except Exception as e:
            logger.error(f"❌ 创建知识库失败: {e}")
            return False
    
    def delete_kb(self, kb_id: str) -> Tuple[bool, str]:
        """删除知识库（不允许删除 default）"""
        if kb_id == "default":
            return False, "不允许删除默认知识库"
        
        try:
            delete_project(kb_id)
            logger.info(f"✅ 删除知识库: {kb_id}")
            return True, "删除成功"
        except Exception as e:
            logger.error(f"❌ 删除知识库失败: {e}")
            return False, str(e)
    
    def get_kb_stats(self, kb_id: str) -> KnowledgeBaseStats:
        """获取知识库统计信息"""
        stats = get_project_stats(kb_id)
        return KnowledgeBaseStats(
            file_count=stats.get("file_count", 0),
            session_count=stats.get("session_count", 0),
            message_count=stats.get("message_count", 0),
            latest_file_time=stats.get("latest_file_time"),
            latest_session_time=stats.get("latest_session_time"),
        )
    
    def get_kb_files(self, kb_id: str, limit: int = 50) -> List[FileRecord]:
        """获取知识库文件列表"""
        files = list_project_files(kb_id)
        return [
            FileRecord(
                id=f[0],
                source=f[1],
                file_type=f[2],
                chunks_count=f[3],
                created_at=f[4]
            )
            for f in files[:limit]
        ]
    
    def get_all_stats_table(self) -> List[Dict]:
        """获取所有知识库的统计表格数据"""
        kbs = self.get_all_kbs()
        rows = []
        for kb in kbs:
            stats = self.get_kb_stats(kb.id)
            rows.append({
                "知识库": kb.name,
                "ID": kb.id,
                "文件数": stats.file_count,
                "会话数": stats.session_count,
                "消息数": stats.message_count,
                "最近入库": stats.latest_file_time or "-",
                "最近会话": stats.latest_session_time or "-",
            })
        return rows
    
    # ==================== 会话管理 ====================
    
    def get_sessions(self, kb_id: str) -> List[Tuple[str, str]]:
        """获取知识库的所有会话"""
        return get_sessions_by_project(kb_id)
    
    def get_or_create_session(self, kb_id: str, session_id: Optional[str] = None) -> str:
        """获取或创建会话"""
        if session_id:
            sessions = self.get_sessions(kb_id)
            session_ids = [s[0] for s in sessions]
            if session_id in session_ids:
                return session_id
        
        # 尝试获取最新会话
        latest = get_latest_session_by_project(kb_id)
        if latest:
            return latest[0]
        
        # 创建新会话
        import uuid
        new_id = str(uuid.uuid4())
        create_session(new_id, "默认会话", project_id=kb_id)
        logger.info(f"📝 创建新会话: {new_id[:8]}...")
        return new_id
    
    def create_new_session(self, kb_id: str, title: str) -> str:
        """创建新会话"""
        import uuid
        new_id = str(uuid.uuid4())
        create_session(new_id, title.strip(), project_id=kb_id)
        logger.info(f"📝 创建新会话: {title} ({new_id[:8]}...)")
        return new_id
    
    def get_session_map(self, kb_id: str) -> Dict[str, str]:
        """获取会话 ID -> 名称 的映射"""
        sessions = self.get_sessions(kb_id)
        return {s[0]: s[1] for s in sessions}