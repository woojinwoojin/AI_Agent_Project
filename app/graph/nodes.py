"""LangGraph 노드: router / rag / tool / response."""

import json
import logging
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

logger = logging.getLogger("app.rag")


# "none" = 카테고리 미분류(전체 검색). Optional(null)보다 명시 값이 구조화 출력에서 안정적.
CATEGORY_L1 = Literal[
    "graduation",
    "course",
    "academic_calendar",
    "social_service",
    "leave_return",
    "contact",
    "none",
]


class IntentRoute(BaseModel):
    """LLM은 의도+카테고리만 분류. (숫자 인자 추출은 구조화 출력이 불안정하여 규칙으로 처리)"""

    intent: Literal["chat", "rag", "tool"] = Field(description="사용자 의도")
    category_l1: CATEGORY_L1 = Field(
        default="none",
        description="intent=rag 일 때 질문이 속한 카테고리. 판단 어려우면 'none'.",
    )


# chat으로 오분류돼도 '사실 정보'를 묻는 신호가 있으면 rag로 강제 (근거 없는 답변/환각 방지)
_INFO_SIGNALS = (
    "문의",
    "연락처",
    "연락",
    "전화",
    "번호",
    "규정",
    "일정",
    "신청",
    "방법",
    "장학",
    "기숙사",
    "생활관",
    "벌점",
    "졸업",
    "수강",
    "성적",
    "학점",
    "도서관",
    "포털",
    "휴학",
    "복학",
    "전과",
    "재수강",
    "교육과정",
    "등록금",
    "증명",
    "취업",
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


# 카테고리 키워드 분류(결정적). Solar 구조화출력이 category를 잘 안 채워 규칙을 주 경로로 쓴다.
# 순서 = 우선순위(위에서부터 먼저 매칭). 시간/일정 신호는 course 보다 먼저 둬 '언제' 질문을 일정으로.
_CATEGORY_KEYWORDS: list[tuple[str, tuple[str, ...]]] = [
    ("social_service", ("사회봉사", "봉사활동", "봉사시간", "자원봉사", "봉사")),
    ("leave_return", ("휴학", "복학", "휴학연기", "복적")),
    (
        "graduation",
        ("졸업", "학위", "졸업인증", "외국어인증", "외국어 졸업", "졸업요건", "졸업학점"),
    ),
    (
        "academic_calendar",
        (
            "일정",
            "날짜",
            "언제",
            "며칠",
            "기간",
            "개강",
            "종강",
            "방학",
            "시험",
            "중간고사",
            "기말고사",
            "성적",
            "등록금",
            "계절학기",
        ),
    ),
    (
        "course",
        (
            "수강신청",
            "수강 신청",
            "수강정정",
            "수강 정정",
            "수강포기",
            "수강 포기",
            "수강",
            "과목",
            "교육과정",
            "커리큘럼",
            "트랙",
            "시간표",
            "강의",
            "전공필수",
            "전공선택",
            "이수구분",
        ),
    ),
    ("contact", ("전화번호", "연락처", "문의", "사무실", "어디에 물어", "어디로 문의")),
]


def classify_categories(text: str) -> list[str]:
    """질문을 category_l1 후보로 분류(키워드 규칙).

    기존에는 첫 매칭에서 즉시 return 해 "복학 기간"처럼 leave_return(복학)과
    academic_calendar(기간)에 동시에 걸치는 질문이 leave_return 하나로만
    좁혀져, 실제 일정 데이터가 담긴 문서가 검색 범위에서 빠지는 문제가 있었다.
    매칭되는 카테고리를 전부 모아 반환한다(순서 = 우선순위, 첫 항목이 주 카테고리).
    """
    return [cat for cat, words in _CATEGORY_KEYWORDS if any(w in text for w in words)]


# 키워드 표에 없는 표현(예: "복학 몇 월부터 가능해?")도 놓치지 않도록,
# 시간 신호가 있으면 관련 category_l1을 추가로 후보에 넣는다(하드 필터가 아니라 확장).
_TIME_SIGNAL_WORDS = (
    "기간",
    "언제",
    "며칠",
    "날짜",
    "마감",
    "일정",
    "개강",
    "종강",
    "까지",
    "부터",
)

_RELATED_CATEGORIES: dict[str, tuple[str, ...]] = {
    "leave_return": ("academic_calendar",),
    "social_service": ("academic_calendar",),
    "graduation": ("academic_calendar",),
    "course": ("academic_calendar",),
}


def expand_categories(text: str, categories: list[str]) -> list[str]:
    """시간 신호가 있으면 제도 카테고리에 연관된 academic_calendar 등을 추가."""
    if not categories or not any(w in text for w in _TIME_SIGNAL_WORDS):
        return categories

    expanded = list(categories)
    for cat in categories:
        for related in _RELATED_CATEGORIES.get(cat, ()):
            if related not in expanded:
                expanded.append(related)
    return expanded


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
    llm_category = "none"
    try:
        result = await structured_llm.ainvoke(
            [SystemMessage(content=ROUTER_PROMPT), HumanMessage(content=user_input)]
        )
        intent = result.intent
        llm_category = result.category_l1
    except Exception:
        intent = "rag"

    # 안전망: chat으로 분류됐어도 '사실 정보'를 묻는 질문이면 rag로 (근거 없는 환각 방지)
    if intent == "chat" and _looks_informational(user_input):
        intent = "rag"

    # 카테고리 분류: 키워드 규칙(다중 매칭 + 시간 신호 확장) 주 경로 + LLM 보조(규칙 미매칭 시).
    # ("none"/contact 는 문서가 없어 필터 안 함 → 전체 검색 후 가드레일이 문의처 안내)
    rule_categories: list[str] = []
    expanded_categories: list[str] = []
    categories: list[str] | None = None
    if intent == "rag":
        rule_categories = classify_categories(user_input)
        expanded_categories = expand_categories(user_input, rule_categories)
        categories = [c for c in expanded_categories if c != "contact"] or None
        if not categories and llm_category not in ("none", "contact"):
            categories = [llm_category]

    tool_name, tool_args = None, None
    if intent == "tool":
        tool_name, tool_args = resolve_tool(user_input)
        if tool_name is None:
            intent = "rag"  # 도구 판별 실패 → RAG로 폴백

    logger.info(
        json.dumps(
            {
                "stage": "router",
                "session_id": state.get("session_id"),
                "question": user_input,
                "intent": intent,
                "rule_categories": rule_categories,
                "expanded_categories": expanded_categories,
                "llm_category": llm_category,
                "final_categories": categories,
            },
            ensure_ascii=False,
        )
    )

    return {
        "intent": intent,
        "category_l1": categories,
        "tool_name": tool_name,
        "tool_args": tool_args,
    }


async def rag_node(state: AgentState) -> dict:
    """질문 관련 문서 검색. 자료가 없거나 관련도가 낮으면 가드레일로 전환."""
    user_input = state["messages"][-1].content
    try:
        docs = await get_rag_repository().search_similar(
            user_input,
            k=5,
            category_l1=state.get("category_l1"),
            session_id=state.get("session_id"),
        )
    except Exception:
        docs = []

    top_score = docs[0]["score"] if docs else 0.0
    guardrail = not docs or top_score < config.GUARDRAIL_MIN_SCORE
    contact = match_contact(user_input) if guardrail else None

    logger.info(
        json.dumps(
            {
                "stage": "guardrail",
                "session_id": state.get("session_id"),
                "question": user_input,
                "top_score": top_score,
                "guardrail": guardrail,
                "contact_matched": contact is not None,
            },
            ensure_ascii=False,
        )
    )

    if guardrail:
        # 자료로 답할 수 없음 → 질문 주제에 맞는 문의처를 찾아 안내
        return {"retrieved_docs": docs, "guardrail": True, "contact": contact}
    return {"retrieved_docs": docs, "guardrail": False, "contact": None}


async def tool_node(state: AgentState) -> dict:
    """Router가 고른 도구 실행."""
    result = await tool_executor.execute(
        tool_name=state["tool_name"],
        tool_args=state["tool_args"] or {},
        session_id=state["session_id"],
    )
    return {"tool_result": result}


def build_response_inputs(state: AgentState) -> tuple[str, str]:
    """최종 응답 생성을 위한 system_prompt, user_input 생성."""
    user_input = state["messages"][-1].content
    intent = state["intent"]

    if intent == "rag" and state.get("guardrail"):
        contact_text = format_contact(state.get("contact"))
        system_prompt = f"{RESPONSE_PROMPT}\n\n{GUARDRAIL_GROUNDING.format(contact=contact_text)}"

    elif intent == "rag":
        context = (
            "\n\n".join(
                f"[자료{i + 1}] {d['content']}" for i, d in enumerate(state["retrieved_docs"])
            )
            or "(관련 자료 없음)"
        )
        system_prompt = f"{RESPONSE_PROMPT}\n\n{RAG_GROUNDING.format(context=context)}"

    elif intent == "tool":
        tool_result = json.dumps(state["tool_result"], ensure_ascii=False)
        system_prompt = f"{RESPONSE_PROMPT}\n\n{TOOL_GROUNDING.format(tool_result=tool_result)}"

    else:
        system_prompt = RESPONSE_PROMPT

    return system_prompt, user_input


async def response_node(state: AgentState) -> dict:
    """intent별 그라운딩을 붙여 최종 응답 생성."""
    llm = get_llm()

    system_prompt, user_input = build_response_inputs(state)

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_input),
    ]

    answer = await llm.ainvoke(messages)
    return {"messages": [AIMessage(content=answer.content)]}
