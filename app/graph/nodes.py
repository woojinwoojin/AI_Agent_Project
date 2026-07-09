"""LangGraph 노드: router / rag / tool / response."""
import json
import re
from typing import Literal

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_upstage import ChatUpstage
from pydantic import BaseModel, Field

from app import config
from app.core.prompts import (
    GUARDRAIL_GROUNDING,
    RAG_GROUNDING,
    RESPONSE_PROMPT,
    ROUTER_PROMPT,
    TOOL_GROUNDING,
)
from app.graph.state import AgentState
from app.repositories.contacts import format_contact, match_contact
from app.repositories.rag import get_rag_repository
from app.tools.executor import ToolExecutor


class IntentRoute(BaseModel):
    """LLM은 의도만 분류. (숫자 인자 추출은 구조화 출력이 불안정하여 규칙으로 처리)"""

    intent: Literal["chat", "rag", "tool"] = Field(description="사용자 의도")


# chat으로 오분류돼도 '사실 정보'를 묻는 신호가 있으면 rag로 강제 (근거 없는 답변/환각 방지)
_INFO_SIGNALS = (
    "문의", "연락처", "연락", "전화", "번호", "규정", "일정", "신청", "방법",
    "장학", "기숙사", "생활관", "벌점", "졸업", "수강", "성적", "학점", "도서관",
    "포털", "휴학", "복학", "전과", "재수강", "교육과정", "등록금", "증명", "취업",
)


def _looks_informational(text: str) -> bool:
    return any(sig in text for sig in _INFO_SIGNALS)


def _find_int(pattern: str, text: str) -> int | None:
    m = re.search(pattern, text)
    return int(m.group(1)) if m else None


def _detect_track(text: str) -> str | None:
    t = text.lower()
    if "aiot" in t:
        return "AIoT"
    if "vision" in t or "language" in t or "비전" in text or "자연어" in text:
        return "Vision & Language"
    if "intelligent" in t or "인텔리전트" in text:
        return "Intelligent SW"
    if "부트캠프" in text or "bootcamp" in t:
        return "AI부트캠프"
    return None


def resolve_tool(text: str) -> tuple[str | None, dict | None]:
    """자연어에서 도구 이름과 인자를 규칙 기반으로 추출."""
    학년 = _find_int(r"([1-4])\s*학년", text)
    학기 = _find_int(r"([1-2])\s*학기", text)

    # 1) 졸업요건 계산: '학점' + ('졸업' 또는 '남')
    if "학점" in text and ("졸업" in text or "남" in text):
        args: dict = {}
        for key, pat in [
            ("전공필수", r"전공\s*필수\s*(\d+)"),
            ("전공선택", r"전공\s*선택\s*(\d+)"),
            ("공통필수", r"공통\s*필수\s*(\d+)"),
            ("공통선택", r"공통\s*선택\s*(\d+)"),
        ]:
            v = _find_int(pat, text)
            if v is not None:
                args[key] = v
        if "전공필수" not in args and "전공선택" not in args:
            v = _find_int(r"전공\D{0,3}(\d+)\s*학점", text) or _find_int(r"(\d+)\s*학점", text)
            if v is not None:
                args["전공"] = v
        if args:
            return "calc_graduation_progress", args

    # 2) 과목 추천: 학년+학기
    if 학년 and 학기:
        args = {"학년": 학년, "학기": 학기}
        trk = _detect_track(text)
        if trk:
            args["트랙"] = trk
        return "recommend_courses", args

    return None, None


def get_llm() -> ChatUpstage:
    return ChatUpstage(
        api_key=config.UPSTAGE_API_KEY,
        model=config.LLM_MODEL,
        timeout=30,
        max_retries=2,
    )


tool_executor = ToolExecutor()


async def router_node(state: AgentState) -> dict:
    """LLM으로 의도 분류 → tool이면 규칙 기반으로 도구/인자 결정."""
    user_input = state["messages"][-1].content
    structured_llm = get_llm().with_structured_output(IntentRoute)
    try:
        result = await structured_llm.ainvoke(
            [SystemMessage(content=ROUTER_PROMPT), HumanMessage(content=user_input)]
        )
        intent = result.intent
    except Exception:
        intent = "rag"

    # 안전망: chat으로 분류됐어도 '사실 정보'를 묻는 질문이면 rag로 (근거 없는 환각 방지)
    if intent == "chat" and _looks_informational(user_input):
        intent = "rag"

    tool_name, tool_args = None, None
    if intent == "tool":
        tool_name, tool_args = resolve_tool(user_input)
        if tool_name is None:
            intent = "rag"  # 도구 판별 실패 → RAG로 폴백

    return {"intent": intent, "tool_name": tool_name, "tool_args": tool_args}


async def rag_node(state: AgentState) -> dict:
    """질문 관련 문서 검색. 자료가 없거나 관련도가 낮으면 가드레일로 전환."""
    user_input = state["messages"][-1].content
    try:
        docs = await get_rag_repository().search_similar(user_input, k=5)
    except Exception:
        docs = []

    top_score = docs[0]["score"] if docs else 0.0
    if not docs or top_score < config.GUARDRAIL_MIN_SCORE:
        # 자료로 답할 수 없음 → 질문 주제에 맞는 문의처를 찾아 안내
        return {
            "retrieved_docs": docs,
            "guardrail": True,
            "contact": match_contact(user_input),
        }
    return {"retrieved_docs": docs, "guardrail": False, "contact": None}


async def tool_node(state: AgentState) -> dict:
    """Router가 고른 도구 실행."""
    result = await tool_executor.execute(
        tool_name=state["tool_name"],
        tool_args=state["tool_args"] or {},
        session_id=state["session_id"],
    )
    return {"tool_result": result}


async def response_node(state: AgentState) -> dict:
    """intent별 그라운딩을 붙여 최종 응답 생성."""
    llm = get_llm()
    user_input = state["messages"][-1].content
    intent = state["intent"]

    if intent == "rag" and state.get("guardrail"):
        contact_text = format_contact(state.get("contact"))
        system_prompt = f"{RESPONSE_PROMPT}\n\n{GUARDRAIL_GROUNDING.format(contact=contact_text)}"
    elif intent == "rag":
        context = "\n\n".join(
            f"[자료{i+1}] {d['content']}" for i, d in enumerate(state["retrieved_docs"])
        ) or "(관련 자료 없음)"
        system_prompt = f"{RESPONSE_PROMPT}\n\n{RAG_GROUNDING.format(context=context)}"
    elif intent == "tool":
        tool_result = json.dumps(state["tool_result"], ensure_ascii=False)
        system_prompt = f"{RESPONSE_PROMPT}\n\n{TOOL_GROUNDING.format(tool_result=tool_result)}"
    else:
        system_prompt = RESPONSE_PROMPT

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_input),
    ]
    answer = await llm.ainvoke(messages)
    return {"messages": [AIMessage(content=answer.content)]}
