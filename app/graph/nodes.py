"""LangGraph 노드: router / rag / tool / response."""

import json
import logging
import re
from datetime import datetime
from typing import Literal

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_upstage import ChatUpstage
from pydantic import BaseModel, Field

from app import config
from app.core.prompts import (
    ACADEMIC_CALENDAR_HINT,
    GUARDRAIL_GROUNDING,
    RAG_GROUNDING,
    RESPONSE_PROMPT,
    ROUTER_PROMPT,
    TOOL_GROUNDING,
)
from app.graph.state import AgentState
from app.repositories.contacts import format_contact, match_contact
from app.repositories.rag import get_rag_repository
from app.services.reminder_time import now_kst, parse_remind_at
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


# chat 오분류 안전망: 예전엔 "사실 정보 신호 블랙리스트"에 있는 단어가 있어야만
# rag로 되돌렸는데("졸업"/"학점"/... 없으면 그냥 chat 통과), "학년별로 알려줘"처럼
# 목록에 없는 표현은 그대로 chat으로 빠져 근거 문서 하나 없이 LLM이 학과 정보를
# (가짜 과목명·전화번호·이메일까지) 지어내는 사고로 이어졌다.
# ROUTER_PROMPT 자체가 "chat은 인사/감사/잡담/사용법 정도로 매우 좁게"라고 명시하므로,
# 블랙리스트 대신 화이트리스트로 뒤집는다: 진짜 잡담으로 보이는 짧은 인사/감사/작별/
# 사용법 질문이 아니면 전부 rag로 보내 최소한 가드레일(문의처 안내)을 거치게 한다.
_SMALLTALK_PATTERNS = (
    "안녕",
    "hi",
    "hello",
    "헬로",
    "고마워",
    "고마웠",
    "감사",
    "고맙",
    "잘가",
    "바이",
    "bye",
    "수고",
    "뭘 도와",
    "뭐 도와",
    "무엇을 도와",
    "어떻게 써",
    "사용법",
    "어떻게 사용",
    "넌 누구",
    "너는 누구",
    "넌 뭐야",
    "너는 뭐야",
    "너 뭐하는",
)


def _looks_like_smalltalk(text: str) -> bool:
    stripped = text.strip()
    low = stripped.lower()
    return any(p in stripped or p in low for p in _SMALLTALK_PATTERNS)


def _looks_like_reminder(text: str) -> bool:
    """이메일 리마인드 요청 신호가 있는지(주소 유무와 무관). 주소가 없어도
    reminder 흐름으로 보내 되묻기(awaiting_email)부터 시작하게 한다."""
    return any(sig in text for sig in _REMINDER_SIGNALS)


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


# 이메일 리마인드(ADR-007): 이메일 주소는 개인정보이므로 LLM 구조화 출력으로
# 추출하지 않고, 이 정규식으로 규칙 기반 추출만 한다.
_EMAIL_PATTERN = r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"
_REMINDER_SIGNALS = (
    "리마인드",
    "메일로 알려줘",
    "메일로 보내",
    "이메일로 알려줘",
    "이메일로 보내",
    "이메일 보내",
    "메일 보내",
    "메일 발송",
    "이메일 발송",
)


def resolve_tool(text: str) -> tuple[str | None, dict | None]:
    """자연어에서 도구 이름과 인자를 규칙 기반으로 추출."""
    학년 = _find_int(r"([1-4])\s*학년", text)
    학기 = _find_int(r"([1-2])\s*학기", text)

    # 이메일 리마인드는 이제 별도의 멀티턴 흐름(reminder_node)이 처리한다.
    # router가 리마인드 신호를 보면 intent=reminder로 보내 "물어보고→확인받고→
    # 발송"을 거치므로, 여기(도구 규칙)서는 리마인드를 다루지 않는다.

    # 1) 졸업요건 계산: '학점' + (졸업/남음/이수완료를 암시하는 표현)
    # "전선 30학점 들었는데 얼마나 더 들어야돼?"처럼 '졸업'/'남'이 없이
    # 줄임말(전선·전필·교필·교선)만 쓰는 질문도 있어 트리거 표현을 넓혀둔다.
    # "4학년 1학기까지 120학점 채웠어"처럼 완료형 서술만 있고 질문형 신호가
    # 없는 경우("채워야"는 안 잡힘)도 있어 "채웠"/"이수했"도 추가한다 — 이게
    # 없으면 학년+학기 숫자 때문에 아래 2)번(과목 추천)으로 잘못 빠진다.
    _REMAINING_SIGNALS = ("졸업", "남", "더", "부족", "채워야", "얼마나", "채웠", "이수했")
    if "학점" in text and any(w in text for w in _REMAINING_SIGNALS):
        args: dict = {}
        for key, pat in [
            ("전공필수", r"(?:전공\s*필수|전필)\s*(\d+)"),
            ("전공선택", r"(?:전공\s*선택|전선)\s*(\d+)"),
            ("공통필수", r"(?:공통\s*필수|교필)\s*(\d+)"),
            ("공통선택", r"(?:공통\s*선택|교선)\s*(\d+)"),
        ]:
            v = _find_int(pat, text)
            if v is not None:
                args[key] = v
        if not args:
            v = _find_int(r"전공\D{0,3}(\d+)\s*학점", text) or _find_int(r"(\d+)\s*학점", text)
            if v is not None:
                args["전공"] = v
        if args:
            return "calc_graduation_progress", args

    # 2) 과목 추천: 학년+학기 숫자 + 추천을 원한다는 신호가 함께 있을 때만.
    # 숫자만 보고 무조건 반환하면 "4학년 1학기까지 120학점 채웠어"처럼 학년/
    # 학기가 다른 맥락(예: 이수학점 서술)으로 언급된 문장까지 과목 추천으로
    # 잘못 라우팅된다.
    _RECOMMEND_SIGNALS = (
        "추천",
        "들어야",
        "뭐 들어",
        "무슨 과목",
        "개설",
        "과목",
        "수강",
        "시간표",
        "커리큘럼",
    )
    if 학년 and 학기 and any(w in text for w in _RECOMMEND_SIGNALS):
        args = {"학년": 학년, "학기": 학기}
        trk = _detect_track(text)
        if trk:
            args["트랙"] = trk
        return "recommend_courses", args

    return None, None


# 리마인드 pending_action(awaiting_email/awaiting_confirm) 진행 중에도, 사용자가
# 완전히 다른 학사 질문을 하면 흐름을 끊어야 한다(그렇지 않으면 계속 이메일/확인을
# 되묻는 문제가 생김). classify_categories/resolve_tool로 이미 잡히는 학사 키워드는
# 여기서 중복 나열하지 않고, 그 규칙들에 안 걸리는 일반적인 질문 표현만 추가한다.
_GENERIC_QUESTION_SIGNALS = ("알려줘", "궁금해", "뭐야", "무엇", "설명해")


def _looks_like_new_question(text: str) -> bool:
    """pending_action 중이어도 새 학사 질문으로 볼지 판별."""
    if classify_categories(text):
        return True
    if resolve_tool(text)[0] is not None:
        return True
    return any(w in text for w in _GENERIC_QUESTION_SIGNALS)


def get_llm() -> ChatUpstage:
    return ChatUpstage(
        api_key=config.UPSTAGE_API_KEY,
        model=config.LLM_MODEL,
        temperature=0.0,
        timeout=30,
        max_retries=2,
    )


tool_executor = ToolExecutor()


async def router_node(state: AgentState) -> dict:
    """LLM으로 의도 분류 + 규칙 기반 도구 판별(resolve_tool)이 우선. 규칙이 도구
    패턴을 찾으면 LLM 판단과 무관하게 intent=tool로 승격한다."""
    user_input = state["messages"][-1].content

    # 진행 중인 리마인드 확인 대화가 있으면, 원칙적으로 이번 사용자 답을 그 대화의
    # 응답으로 이어받는다("응"·이메일 주소 같은 짧은 답이 rag/chat으로 새는 것을 막기
    # 위해 LLM 재분류를 건너뜀). 다만 사용자가 이메일/확인 대신 완전히 다른 학사
    # 질문을 하면 pending을 계속 붙잡고 있으면 안 되므로(계속 이메일을 되묻는 문제),
    # 그런 경우엔 pending을 버리고 아래 일반 라우팅으로 흘려보낸다.
    pending = state.get("pending_action")
    if pending and pending.get("type") == "reminder":
        stage = pending.get("stage")
        is_continuation = (
            (
                stage == "awaiting_email"
                and (_extract_email(user_input) or any(w in user_input for w in _CONFIRM_NO))
            )
            or (stage == "awaiting_confirm" and _classify_confirm(user_input) != "unclear")
            or not _looks_like_new_question(user_input)
        )

        logger.info(
            json.dumps(
                {
                    "stage": "router",
                    "session_id": state.get("session_id"),
                    "question": user_input,
                    "intent": "reminder" if is_continuation else "reroute_new_question",
                    "reminder_stage": stage,
                },
                ensure_ascii=False,
            )
        )

        if is_continuation:
            return {
                "intent": "reminder",
                "retrieved_docs": [],
                "guardrail": False,
                "contact": None,
                "tool_result": None,
            }

        # 새 학사 질문으로 판단 → 리마인드 대기 상태를 버리고 아래 일반 라우팅을 계속 진행
        pending = None

    structured_llm = get_llm().with_structured_output(IntentRoute)
    llm_category = "none"
    try:
        result = await structured_llm.ainvoke(
            [SystemMessage(content=ROUTER_PROMPT), HumanMessage(content=user_input)]
        )
        llm_intent = result.intent
        llm_category = result.category_l1
    except Exception:
        llm_intent = "rag"
    intent = llm_intent

    # 도구 판별: 카테고리 규칙과 동일하게 규칙(resolve_tool)을 주 경로로 쓴다.
    # LLM structured output이 "학점 30 남았는데 얼마나 더?" 같은 축약 표현(전선/전필 등)에서
    # intent를 tool 대신 rag로 잘못 분류하는 경우가 있어, LLM 판단과 무관하게 규칙이 도구
    # 패턴을 찾으면 강제로 tool로 승격한다. LLM이 tool이라 했는데 규칙이 못 찾으면(드문
    # 오탐) rag로 폴백한다.
    tool_name, tool_args = resolve_tool(user_input)
    tool_forced_by_rule = False

    if tool_name is not None:
        if intent != "tool":
            tool_forced_by_rule = True
        intent = "tool"
    elif intent == "tool":
        intent = "rag"  # LLM은 tool이라 했지만 규칙이 인자를 못 찾음 → RAG로 폴백

    # 안전망: chat으로 분류됐어도 진짜 잡담(인사/감사/사용법 등)이 아니면 rag로 강제.
    # (근거 문서 없이 LLM이 학과 정보를 자유생성하다 지어내는 환각 방지)
    if intent == "chat" and not _looks_like_smalltalk(user_input):
        intent = "rag"

    # 새 리마인드 요청: 리마인드 신호가 있으면 reminder 흐름으로 시작한다.
    # (졸업계산·과목추천처럼 규칙으로 확정된 tool 요청은 그대로 두고 그 외에서만 승격)
    if intent != "tool" and _looks_like_reminder(user_input):
        intent = "reminder"

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

    logger.info(
        json.dumps(
            {
                "stage": "router",
                "session_id": state.get("session_id"),
                "question": user_input,
                "llm_intent": llm_intent,
                "intent": intent,
                "tool_forced_by_rule": tool_forced_by_rule,
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
        # 체크포인터로 턴 간 상태가 영속되므로, 이번 턴에 rag/tool을 안 타면
        # 지난 턴의 검색 결과·가드레일·도구 결과가 그대로 남아있을 수 있다.
        # 매 턴 router에서 명시적으로 리셋해 이전 턴 상태가 새지 않게 한다.
        "retrieved_docs": [],
        "guardrail": False,
        "contact": None,
        "tool_result": None,
        # 이 경로(early return이 아닌 일반 라우팅)에 도달했다는 건 pending_action이
        # 애초에 없었거나, 위에서 새 질문으로 판단해 버렸다는 뜻 → 명시적으로 비운다.
        "pending_action": pending,
    }


async def rag_node(state: AgentState) -> dict:
    """질문 관련 문서 검색. 자료가 없거나 관련도가 낮으면 가드레일로 전환."""
    user_input = state["messages"][-1].content
    categories = state.get("category_l1")
    # 카테고리가 여러 개 걸린 복합 질문(예: "2학기 수강신청" -> academic_calendar
    # + course)은 k=5로 좁히면 한 카테고리가 상위를 독식해 다른 카테고리 문서가
    # 아예 잘려나갈 수 있다. 여유를 더 준다.
    k = 8 if categories and len(categories) > 1 else 5
    try:
        docs = await get_rag_repository().search_similar(
            user_input,
            k=k,
            category_l1=categories,
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


# ── 이메일 리마인드 멀티턴 확인 흐름 ─────────────────────────────────────
# 외부 상태를 바꾸는 이메일 발송은 "물어보고 → 확인받고 → 발송"으로만 실행한다
# (README §6.2). 진행 단계는 pending_action(체크포인터로 턴 간 영속)에 담고,
#   {"type":"reminder", "stage":"awaiting_email"|"awaiting_confirm",
#    "content":str, "remind_at":ISO, "remind_label":str, "email":str|None}
# 안내/확인 문구는 이 노드가 '결정적 템플릿'으로 직접 만든다: 이메일 주소를 응답
# LLM에 넘기지 않아 ADR-007을 지키고(주소는 여기서만 다룸), 주소를 정확히 되비추며,
# 턴마다 동일하게 동작한다. 그래서 response 노드를 타지 않고 바로 END로 간다.

# 확인 응답(예/아니오) 규칙 분류용. 애매하면 재확인하므로 오분류 위험은 낮다.
_CONFIRM_YES = (
    "응",
    "네",
    "넵",
    "예",
    "그래",
    "좋아",
    "좋습니다",
    "보내",
    "부탁",
    "ㅇㅇ",
    "오케",
    "ok",
    "okay",
    "yes",
    "맞아",
    "해줘",
    "해 줘",
    "진행",
    "등록",
)
_CONFIRM_NO = (
    "아니",
    "아뇨",
    "취소",
    "됐어",
    "됐습니다",
    "됐네",
    "싫",
    "no",
    "하지마",
    "하지 마",
    "지마",
    "지 마",
    "그만",
    "말아",
    "말래",
    "말자",
    "안 보",
)


def _extract_email(text: str) -> str | None:
    m = re.search(_EMAIL_PATTERN, text)
    return m.group(0) if m else None


def _classify_confirm(text: str) -> Literal["yes", "no", "unclear"]:
    low = text.strip().lower()
    # 부정을 먼저 본다("아니 보내지마"처럼 긍정어가 섞여도 취소로 처리)
    if any(p in low for p in _CONFIRM_NO):
        return "no"
    if any(p in low for p in _CONFIRM_YES):
        return "yes"
    return "unclear"


def _timing_phrase(pending: dict) -> str:
    label = pending.get("remind_label") or ""
    return "지금 바로" if label == "지금 바로" else f"{label}에"


def _ask_email_msg(pending: dict) -> str:
    return (
        f"네! 요청하신 내용을 {_timing_phrase(pending)} 리마인드 메일로 보내드릴 수 있어요. 📮\n"
        "어느 이메일 주소로 받으실지 알려주시겠어요? (예: hong@gachon.ac.kr)"
    )


def _ask_confirm_msg(pending: dict) -> str:
    return (
        "보내기 전에 확인해주세요! 아래 내용으로 리마인드 메일을 보낼까요?\n"
        f"- 받는 사람: {pending['email']}\n"
        f"- 발송 시점: {pending.get('remind_label')}\n"
        f"- 내용: {pending['content']}\n\n"
        "'네'라고 답하시면 예약할게요. 취소하려면 '아니오'라고 답해주세요."
    )


def _reask_confirm_msg(pending: dict) -> str:
    return (
        "'네'(보내기) 또는 '아니오'(취소)로 답해주세요. "
        f"{pending['email']}로 {_timing_phrase(pending)} 보낼 예정이에요."
    )


_REMINDER_REASK_EMAIL = (
    "앗, 이메일 주소를 못 찾았어요. 예: hong@gachon.ac.kr 처럼 받으실 주소를 알려주시겠어요?"
)
_REMINDER_CANCELED = "알겠어요, 리마인드는 취소했어요. 필요하면 언제든 다시 말씀해 주세요! 🙂"
_REMINDER_REGISTER_FAILED = "죄송해요, 예약 등록 중 문제가 생겼어요. 잠시 후 다시 시도해 주세요."


def _reminder_reply(text: str, pending: dict | None) -> dict:
    """리마인드 노드의 반환 형태: 사용자에게 보낼 메시지 + 진행 상태(pending) 갱신."""
    return {"messages": [AIMessage(content=text)], "pending_action": pending}


async def _register_reminder(pending: dict, session_id: str | None) -> dict:
    """확인 완료 → reminder_requests에 예약 등록(실제 발송은 스케줄러가 처리)."""
    remind_at = datetime.fromisoformat(pending["remind_at"])
    result = await tool_executor.execute(
        tool_name="send_reminder_email",
        tool_args={
            "이메일": pending["email"],
            "내용": pending["content"],
            "발송예정시각": remind_at,
        },
        session_id=session_id,
    )
    if not result.get("success"):
        return _reminder_reply(_REMINDER_REGISTER_FAILED, None)

    if pending.get("remind_label") == "지금 바로":
        msg = "✅ 리마인드 예약을 등록했어요! 곧 메일이 도착할 거예요."
    else:
        msg = f"✅ 리마인드 예약 완료! {pending.get('remind_label')}에 메일로 알려드릴게요."
    return _reminder_reply(msg, None)


async def reminder_node(state: AgentState) -> dict:
    """이메일 리마인드 멀티턴 확인 흐름(물어보고 → 확인받고 → 발송)."""
    user_input = state["messages"][-1].content
    pending = state.get("pending_action")
    session_id = state.get("session_id")

    # ── 진행 중인 대화 이어받기 ─────────────────────────────
    if pending and pending.get("type") == "reminder":
        stage = pending.get("stage")

        if stage == "awaiting_email":
            if any(w in user_input for w in _CONFIRM_NO):
                return _reminder_reply(_REMINDER_CANCELED, None)
            email = _extract_email(user_input)
            if not email:
                return _reminder_reply(_REMINDER_REASK_EMAIL, pending)
            pending = {**pending, "email": email, "stage": "awaiting_confirm"}
            return _reminder_reply(_ask_confirm_msg(pending), pending)

        if stage == "awaiting_confirm":
            decision = _classify_confirm(user_input)
            if decision == "no":
                return _reminder_reply(_REMINDER_CANCELED, None)
            if decision == "unclear":
                return _reminder_reply(_reask_confirm_msg(pending), pending)
            return await _register_reminder(pending, session_id)

    # ── 새 리마인드 요청 시작 ───────────────────────────────
    now = now_kst()
    remind_at = parse_remind_at(user_input, now=now)
    # parse_remind_at은 날짜 표현이 없으면 now를 그대로 반환 → '즉시 발송'으로 본다.
    immediate = abs((remind_at - now).total_seconds()) < 60
    label = "지금 바로" if immediate else remind_at.strftime("%Y-%m-%d %H:%M")

    base = {
        "type": "reminder",
        "content": user_input,
        "remind_at": remind_at.isoformat(),
        "remind_label": label,
        "email": _extract_email(user_input),
    }
    if base["email"]:
        pending = {**base, "stage": "awaiting_confirm"}
        return _reminder_reply(_ask_confirm_msg(pending), pending)
    pending = {**base, "stage": "awaiting_email"}
    return _reminder_reply(_ask_email_msg(pending), pending)


# 학사일정 안내 트리거: category_l1이 academic_calendar 이거나, 질문에 날짜/일정
# 성격의 단어가 있으면 학사일정 페이지 안내 힌트를 붙인다. "예비수강신청 일자"처럼
# 카테고리는 course로 잡혀도 실제로는 일정 질문인 경우를 놓치지 않으려 텍스트도 본다.
_SCHEDULE_HINT_WORDS = (
    "학사일정",
    "일정",
    "일자",
    "날짜",
    "언제",
    "며칠",
    "기간",
    "마감",
    "개강",
    "종강",
    "방학",
    "시험",
    "중간고사",
    "기말고사",
    "성적",
    "수강신청",
    "예비수강신청",
    "수강정정",
    "수강포기",
    "등록금",
    "계절학기",
)


def _is_schedule_related(state: AgentState) -> bool:
    """질문이 학사일정(날짜/기간) 성격인지 판단."""
    if "academic_calendar" in (state.get("category_l1") or []):
        return True
    text = state["messages"][-1].content
    return any(w in text for w in _SCHEDULE_HINT_WORDS)


def build_response_inputs(state: AgentState) -> tuple[str, str]:
    """최종 응답 생성을 위한 system_prompt, user_input 생성."""
    user_input = state["messages"][-1].content
    intent = state["intent"]
    schedule_related = _is_schedule_related(state)

    if intent == "rag" and state.get("guardrail"):
        contact_text = format_contact(state.get("contact"))
        grounding = GUARDRAIL_GROUNDING.format(contact=contact_text)
        if schedule_related:
            grounding = f"{ACADEMIC_CALENDAR_HINT}\n{grounding}"
        system_prompt = f"{RESPONSE_PROMPT}\n\n{grounding}"

    elif intent == "rag":
        context = (
            "\n\n".join(
                f"[자료{i + 1}] {d['content']}" for i, d in enumerate(state["retrieved_docs"])
            )
            or "(관련 자료 없음)"
        )
        grounding = RAG_GROUNDING.format(context=context)
        if schedule_related:
            grounding = f"{ACADEMIC_CALENDAR_HINT}\n{grounding}"
        system_prompt = f"{RESPONSE_PROMPT}\n\n{grounding}"

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
