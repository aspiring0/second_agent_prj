# src/agent/graph.py
"""
LangGraph workflow definition.
Sprint 1: query_rewriter -> researcher -> [tools] -> researcher -> writer -> END
"""

from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

from config.settings import settings
from src.agent.state import AgentState
from src.agent.nodes import researcher_node, writer_node, all_tools
from src.agent.nodes_query import query_rewriter_node

# 1. create workflow with state schema
workflow = StateGraph(AgentState)

# 2. add nodes
# Sprint 1: query rewriter node (before researcher)
if settings.ENABLE_QUERY_REWRITE:
    workflow.add_node("query_rewriter", query_rewriter_node)

workflow.add_node("researcher", researcher_node)
workflow.add_node("writer", writer_node)
workflow.add_node("tools", ToolNode(all_tools))

# 3. entry point
# Sprint 1: if query rewrite enabled, start from query_rewriter; otherwise researcher
if settings.ENABLE_QUERY_REWRITE:
    workflow.set_entry_point("query_rewriter")
    workflow.add_edge("query_rewriter", "researcher")
else:
    workflow.set_entry_point("researcher")

# 4. conditional edges from researcher
def should_continue(state: AgentState):
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tools"
    return "writer"

workflow.add_conditional_edges(
    "researcher",
    should_continue,
    {
        "tools": "tools",
        "writer": "writer"
    }
)

# 5. edges
workflow.add_edge("tools", "researcher")
workflow.add_edge("writer", END)

# 6. compile
app = workflow.compile()
