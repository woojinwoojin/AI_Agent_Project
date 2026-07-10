"""LangGraph 에이전트 상태 정의."""
from typing import Annotated, Literal, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

    # Router가 결정한 의도
    intent: Literal["chat", "rag", "tool"] | None

    # Router가 분류한 카테고리 (rag 검색 필터용). None이면 전체 검색.
    category_l1: str | None

    # RAG 검색 결과 (문서 내용 + 출처)
    retrieved_docs: list[dict]

    # Tool 정보
    tool_name: str | None
    tool_args: dict | None
    tool_result: dict | None

    # 가드레일: RAG로 답을 못 찾아 문의처로 안내해야 할 때
    guardrail: bool
    contact: dict | None

    session_id: str


def create_initial_state(session_id: str, messages: list[BaseMessage] | None = None) -> AgentState:
    return AgentState(
        messages=messages or [],
        intent=None,
        category_l1=None,
        retrieved_docs=[],
        tool_name=None,
        tool_args=None,
        tool_result=None,
        guardrail=False,
        contact=None,
        session_id=session_id,
    )
