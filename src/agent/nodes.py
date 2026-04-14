# src/agent/nodes.py
"""
多智能体节点实现
包含 Researcher（研究员）和 Writer（作家）两个核心节点
"""
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate

from config.settings import settings
from src.agent.state import AgentState
from src.agent.tools_dir import get_all_tools

# 获取所有工具
all_tools = get_all_tools()
from src.agent.prompts import (
    get_researcher_system_message, 
    get_writer_prompt,
    PromptManager
)
from src.utils.logger import setup_logger
from src.utils.model_manager import model_manager

logger = setup_logger("MultiAgent_Nodes")


def get_llm_with_tools():
    """
    获取绑定工具的LLM实例
    支持动态模型切换
    """
    llm = model_manager.get_chat_model(temperature=0.1)
    return llm.bind_tools(all_tools)


# --- 角色 1: 研究员 (Researcher) ---
def researcher_node(state: AgentState) -> AgentState:
    """
    研究员节点：负责理解用户意图，调用工具获取信息

    职责：
    1. 分析用户问题类型
    2. 选择合适的工具获取信息
    3. 将获取的信息传递给 Writer

    使用统一的提示词管理模块获取系统提示
    """
    logger.info("🔬 [研究员] 正在分析用户问题...")

    # 1. 获取当前所有的聊天记录
    messages = state["messages"]

    # 2. 使用统一的提示词管理模块获取系统提示
    system_prompt = get_researcher_system_message()

    # 3. 在函数体内调用 get_llm_with_tools()，确保每次使用最新模型
    _llm_with_tools = get_llm_with_tools()

    # 4. 调用模型（带工具绑定）
    # 我们把 [人设] + [历史记录] 一起发给模型
    response = _llm_with_tools.invoke([system_prompt] + messages)
    
    # 5. 记录调试信息
    if response.tool_calls:
        tool_names = [tc.get("name", "unknown") for tc in response.tool_calls]
        logger.info(f"🔧 [研究员] 决定调用工具: {tool_names}")
    else:
        logger.info("✅ [研究员] 无需调用工具，准备移交 Writer")

    # 6. 返回结果
    # LangGraph 会自动根据 state.py 里的定义，把这个 response 追加到 messages 列表里
    return {"messages": [response]}


# --- 角色 2: 作家 (Writer) ---
def writer_node(state: AgentState):
    """
    作家节点：基于研究员提供的信息，撰写最终回答
    
    职责：
    1. 阅读对话历史（包含工具返回的检索结果）
    2. 整理信息，撰写结构清晰的回答
    3. 标注信息来源（如适用）
    
    使用统一的提示词管理模块获取写作模板
    """
    logger.info("✍️ [作家] 正在撰写回答...")
    
    messages = state["messages"]
    
    # 将历史消息转换为字符串，让作家能够"看见"研究员查到的内容
    conversation_str = _format_conversation_history(messages)
    
    # 使用统一的提示词管理模块获取写作模板
    prompt = get_writer_prompt()
    
    # 获取 LLM 实例（作家使用稍高的温度以增加创造性）
    llm = model_manager.get_chat_model(temperature=0.3)
    chain = prompt | llm
    
    # 调用模型生成回答
    response = chain.invoke({"history": conversation_str})
    
    logger.info(f"✅ [作家] 回答完成，长度: {len(response.content)} 字符")
    
    return {"messages": [response]}


def _format_conversation_history(messages) -> str:
    """
    格式化对话历史，提取关键信息
    
    将消息列表转换为易读的字符串格式，
    特别标注工具调用和返回结果
    """
    formatted_lines = []
    
    for msg in messages:
        msg_type = getattr(msg, 'type', 'unknown')
        content = getattr(msg, 'content', '')
        
        # 处理不同类型的消息
        if msg_type == 'human':
            formatted_lines.append(f"👤 [用户]: {content}")
        elif msg_type == 'ai':
            # 检查是否有工具调用
            tool_calls = getattr(msg, 'tool_calls', None)
            if tool_calls:
                tool_info = []
                for tc in tool_calls:
                    tool_name = tc.get('name', 'unknown')
                    tool_args = tc.get('args', {})
                    tool_info.append(f"{tool_name}({tool_args})")
                formatted_lines.append(f"🤖 [研究员-调用工具]: {', '.join(tool_info)}")
            else:
                formatted_lines.append(f"🤖 [研究员]: {content}")
        elif msg_type == 'tool':
            # 工具返回结果，通常内容较长，截取关键部分
            tool_name = getattr(msg, 'name', 'unknown')
            content_preview = content[:2000] + "..." if len(content) > 2000 else content
            formatted_lines.append(f"🔧 [工具-{tool_name}]: {content_preview}")
        else:
            formatted_lines.append(f"【{msg_type}】: {content}")
    
    return "\n\n".join(formatted_lines)


# --- 辅助函数 ---

def get_conversation_summary(messages) -> str:
    """
    获取对话摘要（用于长对话场景）
    
    当对话历史过长时，可以调用此函数生成摘要，
    减少 token 消耗
    """
    conversation_str = _format_conversation_history(messages)
    
    if len(conversation_str) > 2000:
        # 如果对话太长，只保留最近的消息
        recent_messages = messages[-6:]  # 保留最近6条
        conversation_str = _format_conversation_history(recent_messages)
        logger.info("📝 对话历史较长，已截取最近部分")
    
    return conversation_str