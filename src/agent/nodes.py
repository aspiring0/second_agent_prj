# src/agent/nodes.py
from langchain_openai import ChatOpenAI

# 导入消息类型和提示词模板
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate

from config.settings import settings
from src.agent.state import AgentState
from src.agent.tools import ask_knowledge_base
from src.utils.logger import setup_logger

logger = setup_logger("MultiAgent_Nodes")

llm = ChatOpenAI(
    model=settings.CHAT_MODEL,
    temperature=0.1,
    openai_api_key=settings.OPENAI_API_KEY,
    openai_api_base=settings.OPENAI_BASE_URL
)

# 关键点：bind_tools
# 它的作用是告诉大模型：“你可以使用这些工具”。
# 这样模型在输出时，就有可能输出 tool_calls (工具调用请求)，而不仅仅是文本。
llm_with_tools = llm.bind_tools([ask_knowledge_base])

#角色1：研究员
def researcher_node(state: AgentState) -> AgentState:
    """
    研究员节点：负责查找和提供相关信息
    """
    logger.info("研究员节点正在处理消息...")
    # 1. 获取当前所有的聊天记录
    messages = state["messages"]
    #人设
    system_prompt = SystemMessage(content="""
    你是公司的首席数据研究员。
    你的职责是根据用户的问题，利用【知识库查询工具】去检索精确的事实。
    不要自己编造数据，一切以工具返回的结果为准。
    如果你觉得已经查到了足够的信息，就停止调用工具。                          
    """)

    # 3. 调用模型
    # 我们把 [人设] + [历史记录] 一起发给模型
    response = llm_with_tools.invoke([system_prompt] + messages)
    
    # 4. 返回结果
    # LangGraph 会自动根据 state.py 里的定义，把这个 response 追加到 messages 列表里
    return {"messages": [response]}

# --- 角色 2: 作家 (Writer) ---
def writer_node(state: AgentState):
    logger.info("✍️ [作家] 拿到资料，正在撰写报告...")
    
    messages = state["messages"]
    # 【修复点】：将历史消息转换为字符串
    # 这样作家才能真正“看见”研究员刚才查到了什么
    # 我们遍历消息历史，把每一句是谁说的拼起来
    conversation_str = "\n".join([f"【{msg.type}】: {msg.content}" for msg in messages])
    
    # 调试打印，看看作家到底看见了什么 (可选)
    # print(f"DEBUG_CONTEXT: {conversation_str[:200]}...")
    # 作家的人设
    # 注意：作家不需要工具，它只需要基于上下文（研究员查到的结果）来写
    prompt = ChatPromptTemplate.from_template("""
    你是公司的资深技术作家。
    请阅读下面的【对话历史】，特别是【tool】或【ai】提供的检索事实。
    
    【对话历史】:
    {history}
    
    你的任务是：基于上面的事实，写一篇结构清晰、通俗易懂的回答。
    1. 不要暴露你是在读数据库，要像一个专家一样自然地表达。
    2. 严格基于历史信息中的事实，不要编造（比如不要扯到医疗AI，除非历史里提到了）。
    3. 重点回答用户的见解类或总结类需求。
    
    只输出最终回答即可。
    """)
    
    # 把拼好的字符串传给 {history}
    chain = prompt | llm
    response = chain.invoke({"history": conversation_str})
    
    return {"messages": [response]}
    
    