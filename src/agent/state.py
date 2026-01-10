#src/agent/state.py
'''
src.agent.state 的 Docstring
共享笔记本
'''

import operator  # 标准库：提供函数式操作工具（如 itemgetter、attrgetter 以及算术/比较等操作符对应的函数）
from typing import Annotated, List, TypedDict  # 类型注解：Annotated（为类型附加元信息）、List（列表类型提示）、TypedDict（结构化字典类型提示）
from langchain_core.messages import BaseMessage  # LangChain Core：对话/消息对象的基类，表示单条消息的通用接口

#定义智能体之间传递的数据结构
class AgentState(TypedDict):
    """
    智能体之间传递的数据结构
    Annotated[List[BaseMessage], operator.add] 的意思是：
    当有新消息来的时候，不是覆盖旧消息，而是追加 (add) 到列表后面。
    """
    messages: Annotated[List[BaseMessage], operator.add]
    """
    消息列表，用于存储智能体之间传递的消息
    """
    next_step: str
