# src/agent/graph.py
'''
langgraph.graph:

StateGraph: 用于创建基于状态的图（State Machine）。

END: 一个特殊的节点，代表流程结束。
'''

from langgraph.graph import StateGraph, END
'''
langgraph.prebuilt:

ToolNode: 这是一个神器。
它是一个现成的节点，专门用来执行工具。
当大模型发出“我要调用工具”的指令时，流程流转到这里，
它自动运行 Python 函数，并将结果返回。
'''
from langgraph.prebuilt import ToolNode

from src.agent.state import AgentState
from src.agent.nodes import researcher_node, writer_node
from src.agent.tools import ask_knowledge_base

# 1. 创建图实例，传入我们定义的状态结构
workflow = StateGraph(AgentState)

# 2. 添加节点 (注册所有的工位)
# 给 researcher_node 起个名字叫 "researcher"
workflow.add_node("researcher", researcher_node)
# 给 writer_node 起个名字叫 "writer"
workflow.add_node("writer", writer_node)

# ToolNode 是 LangGraph 预置的，我们只需要把工具列表传给它
# 它会自动识别 messages 里的 tool_calls 并执行
workflow.add_node("tools", ToolNode([ask_knowledge_base]))

# 3. 定义入口 (Entry Point)
# 一旦任务开始，第一个干活的是 "researcher"
workflow.set_entry_point("researcher")

# 4. 定义条件分支 (Conditional Edges)
# 这是图最复杂的逻辑：研究员说完话后，下一步去哪？
def should_continue(state: AgentState):
    # 获取最后一条消息（也就是研究员刚说完的话）
    last_message = state["messages"][-1]
    
    # 检查消息里有没有 tool_calls (工具调用请求)
    if last_message.tool_calls:
        # 如果有，说明研究员想查资料 -> 转给 "tools" 节点
        return "tools"
    
    # 如果没有 tool_calls，说明研究员觉得不用查了，或者已经查完了
    # -> 转给 "writer" 节点去写总结
    return "writer"

# 添加条件边：从 researcher 出发，根据 should_continue 的返回值决定去哪
workflow.add_conditional_edges(
    "researcher",     # 起点
    should_continue,  # 决策函数
    {
        "tools": "tools",   # 如果函数返回 "tools"，就去 tools 节点
        "writer": "writer"  # 如果函数返回 "writer"，就去 writer 节点
    }
)

# 5. 定义普通边 (Edges)
# 只要 "tools" 节点执行完（查完资料），必须强制回到 "researcher"
# 让研究员看看查到的结果对不对，需不需要再查
workflow.add_edge("tools", "researcher")

# 只要 "writer" 写完，流程就结束
workflow.add_edge("writer", END)

# 6. 编译图 (Compile)
# 生成可执行的 Application 对象
app = workflow.compile()