# src/service/__init__.py
"""
Service Layer - 业务逻辑层
负责所有业务逻辑处理，与 UI 层解耦
"""

from src.service.kb_service import KnowledgeBaseService
from src.service.chat_service import ChatService
from src.service.document_service import DocumentService

__all__ = [
    "KnowledgeBaseService",
    "ChatService", 
    "DocumentService",
]