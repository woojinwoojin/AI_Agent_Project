"""LangGraph 그래프 구성: START -> router -> (rag|tool|response) -> response -> END."""

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph

from app.graph.edges import route_by_intent
from app.graph.nodes import rag_node, response_node, router_node, tool_node
from app.graph.state import AgentState

_compiled = None
_checkpointer: BaseCheckpointSaver | None = None


def set_checkpointer(checkpointer: BaseCheckpointSaver | None) -> None:
    """앱 시작 시(lifespan) 체크포인터를 주입한다. 이후 첫 get_graph() 호출에서 반영."""
    global _checkpointer, _compiled
    _checkpointer = checkpointer
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
    return builder.compile(checkpointer=_checkpointer)


def get_graph():
    global _compiled
    if _compiled is None:
        _compiled = create_graph()
    return _compiled
