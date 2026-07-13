"""조건부 라우팅: router 이후 intent에 따라 분기."""

from typing import Literal

from app.graph.state import AgentState


def route_by_intent(state: AgentState) -> Literal["rag", "tool", "response"]:
    intent = state.get("intent") or "chat"
    if intent == "rag":
        return "rag"
    if intent == "tool":
        return "tool"
    return "response"
