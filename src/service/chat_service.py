# src/service/chat_service.py
"""
聊天服务 - Chat Service
负责聊天消息管理和 Agent 交互
"""
from typing import List, Dict, Iterator, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from langchain_core.messages import HumanMessage

from src.agent.graph import app as agent_app
from src.utils.db import get_messages, save_message
from src.utils.logger import setup_logger

logger = setup_logger("CHAT_SERVICE")


class AgentNodeType(Enum):
    """Agent 节点类型"""
    RESEARCHER = "researcher"
    TOOLS = "tools"
    WRITER = "writer"


@dataclass
class ChatMessage:
    """聊天消息"""
    role: str
    content: str
    
    @classmethod
    def from_dict(cls, data: Dict) -> "ChatMessage":
        return cls(role=data["role"], content=data["content"])


@dataclass
class AgentEvent:
    """Agent 事件"""
    node_type: AgentNodeType
    description: str
    
    @classmethod
    def from_stream_event(cls, node_name: str) -> "AgentEvent":
        """从流事件创建"""
        descriptions = {
            AgentNodeType.RESEARCHER: "🔍 研究员: 分析需求...",
            AgentNodeType.TOOLS: "📚 工具: 检索知识库资料...",
            AgentNodeType.WRITER: "✍️ 作家: 整理回答...",
        }
        
        try:
            node_type = AgentNodeType(node_name)
        except ValueError:
            node_type = AgentNodeType.RESEARCHER
            
        return AgentEvent(
            node_type=node_type,
            description=descriptions.get(node_type, f"⚙️ {node_name}: 处理中...")
        )


class ChatService:
    """聊天服务"""
    
    def __init__(self):
        self.agent_app = agent_app
        logger.info("✅ 聊天服务初始化完成")
    
    def get_history(self, session_id: str) -> List[ChatMessage]:
        """获取会话历史消息"""
        messages = get_messages(session_id)
        return [ChatMessage.from_dict(m) for m in messages]
    
    def save_user_message(self, session_id: str, content: str) -> None:
        """保存用户消息"""
        save_message(session_id, "user", content)
        logger.debug(f"💾 保存用户消息: {content[:50]}...")
    
    def save_assistant_message(self, session_id: str, content: str) -> None:
        """保存助手消息"""
        save_message(session_id, "assistant", content)
        logger.debug(f"💾 保存助手消息: {content[:50]}...")
    
    def stream_agent_response(
        self, 
        prompt: str, 
        session_id: str, 
        project_id: str
    ) -> Iterator[Tuple[str, AgentEvent | str]]:
        """
        流式获取 Agent 响应
        
        Yields:
            Tuple[str, AgentEvent | str]: 
                - "event": AgentEvent 事件
                - "response": 最终响应文本
                - "error": 错误信息
        """
        inputs = {"messages": [HumanMessage(content=prompt)]}
        run_config = {"configurable": {"session_id": session_id, "project_id": project_id}}
        
        full_response = ""
        
        try:
            for event in self.agent_app.stream(inputs, config=run_config):
                for node_name, node_output in event.items():
                    agent_event = AgentEvent.from_stream_event(node_name)
                    yield "event", agent_event
                    
                    # 捕获最终响应
                    if node_name == "writer":
                        full_response = node_output["messages"][-1].content
            
            yield "response", full_response
            
        except Exception as e:
            logger.error(f"❌ Agent 执行错误: {e}")
            yield "error", str(e)
    
    def chat(
        self, 
        prompt: str, 
        session_id: str, 
        project_id: str
    ) -> Tuple[bool, str, List[AgentEvent]]:
        """
        执行聊天（非流式）
        
        Returns:
            Tuple[bool, str, List[AgentEvent]]: 
                - 是否成功
                - 响应内容或错误信息
                - 事件列表
        """
        events = []
        full_response = ""
        
        for event_type, data in self.stream_agent_response(prompt, session_id, project_id):
            if event_type == "event":
                events.append(data)
            elif event_type == "response":
                full_response = data
            elif event_type == "error":
                return False, data, events
        
        return True, full_response, events


class ChatUIHelper:
    """聊天 UI 辅助类（用于 Streamlit 的状态显示）"""
    
    @staticmethod
    def get_status_label(event: AgentEvent) -> str:
        """获取状态标签"""
        return event.description
    
    @staticmethod
    def get_final_status(success: bool) -> Tuple[str, str]:
        """获取最终状态"""
        if success:
            return "✅ 完成", "complete"
        else:
            return "❌ 出错", "error"