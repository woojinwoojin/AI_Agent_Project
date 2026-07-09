"""LangGraph 그래프 구성: START -> router -> (rag|tool|response) -> response -> END."""
from langgraph.graph import END, START, StateGraph

from app.graph.edges import route_by_intent
from app.graph.nodes import rag_node, response_node, router_node, tool_node
from app.graph.state import AgentState

_compiled = None


def create_graph():
    builder = StateGraph(AgentState)
    builder.add_node("router", router_node)
    builder.add_node("rag", rag_node)
    builder.add_node("tool", tool_node)
    builder.add_node("response", response_node)

    builder.add_edge(START, "router")
    builder.add_conditional_edges(
        source="router",
        path=route_by_intent,
        path_map={"rag": "rag", "tool": "tool", "response": "response"},
    )
    builder.add_edge("rag", "response")
    builder.add_edge("tool", "response")
    builder.add_edge("response", END)
    return builder.compile()


def get_graph():
    global _compiled
    if _compiled is None:
        _compiled = create_graph()
    return _compiled
