"""평가 시나리오 세트 (멘토 사전질문1 대응).

각 시나리오는 페르소나 + turns. turn 라벨:
  q            : 사용자 질문
  intent       : 기대 intent (rag|tool|reminder|chat|ask_year)
  category     : 기대 category_l1 후보(list) — rag일 때만. classify 6종
                 {graduation, course, academic_calendar, social_service, leave_return, contact}
  answerable   : 우리 자료로 답 가능한가 (True=답변 기대 / False=가드레일 기대)
  tool_name    : intent=tool일 때 기대 도구
  expect_facts : 답변에 등장해야 하는 핵심 사실 키워드(부분일치, 최소만)
  expect_source: 답변에 출처/문의처가 붙어야 하는가
  note         : 설명

category는 '주 카테고리 포함'으로 채점(예측이 라벨을 포함하면 정답).
answerable=False 인데 guardrail 발동 = 정답. multi-turn은 turns 여러 개.
"""

SCENARIOS = [
    # ── 페르소나 A: 신입생(1학년) ─────────────────────────────
    {
        "id": "S01",
        "persona": "신입생(1학년)",
        "turns": [
            {
                "q": "인공지능학과 전공필수 과목 뭐가 있어?",
                "intent": "rag",
                "category": ["course"],
                "answerable": True,
                "expect_facts": ["프로그래밍기초"],
                "expect_source": True,
                "note": "전공필수 목록 RAG",
            },
        ],
    },
    {
        "id": "S02",
        "persona": "신입생(1학년)",
        "turns": [
            {
                "q": "1학년 1학기에는 무슨 과목을 들어?",
                "intent": "tool",
                "tool_name": "recommend_courses",
                "answerable": True,
                "expect_facts": ["프로그래밍기초"],
                "expect_source": True,
                "note": "학년/학기 과목추천 도구",
            },
        ],
    },
    {
        "id": "S03",
        "persona": "신입생(1학년)",
        "turns": [
            {
                "q": "수강신청은 언제 해?",
                "intent": "rag",
                "category": ["academic_calendar"],
                "answerable": True,
                "expect_source": True,
                "note": "일정 RAG",
            },
        ],
    },
    {
        "id": "S04",
        "persona": "신입생(1학년)",
        "turns": [
            {
                "q": "안녕! 너는 뭐하는 앱이야?",
                "intent": "chat",
                "answerable": True,
                "expect_source": False,
                "note": "잡담/사용법",
            },
        ],
    },
    # ── 페르소나 B: 2학년 재학생 ──────────────────────────────
    {
        "id": "S05",
        "persona": "2학년 재학생",
        "turns": [
            {
                "q": "2학년 2학기에 뭐 들으면 좋아?",
                "intent": "tool",
                "tool_name": "recommend_courses",
                "answerable": True,
                "expect_source": True,
                "note": "과목추천",
            },
        ],
    },
    {
        "id": "S06",
        "persona": "2학년 재학생",
        "turns": [
            {
                "q": "전공 30학점 들었는데 졸업까지 얼마나 남았어?",
                "intent": "tool",
                "tool_name": "calc_graduation_progress",
                "answerable": True,
                "expect_facts": ["42"],
                "expect_source": True,
                "note": "졸업계산: 72-30=42",
            },
        ],
    },
    {
        "id": "S07",
        "persona": "2학년 재학생",
        "turns": [
            {
                "q": "자료구조는 몇 학년 과목이야?",
                "intent": "rag",
                "category": ["course"],
                "answerable": True,
                "expect_facts": ["2학년"],
                "expect_source": True,
                "note": "과목 개설학년 RAG",
            },
        ],
    },
    # ── 페르소나 C: 3학년(트랙 고민) ──────────────────────────
    {
        "id": "S08",
        "persona": "3학년(트랙 고민)",
        "turns": [
            {
                "q": "AIoT 트랙은 무슨 과목이 있어?",
                "intent": "rag",
                "category": ["course"],
                "answerable": True,
                "expect_facts": ["AIoT"],
                "expect_source": True,
                "note": "트랙 과목 RAG",
            },
        ],
    },
    {
        "id": "S09",
        "persona": "3학년(트랙 고민)",
        "turns": [
            {
                "q": "Vision & Language 트랙 3학년 1학기 과목 추천해줘",
                "intent": "tool",
                "tool_name": "recommend_courses",
                "answerable": True,
                "expect_source": True,
                "note": "트랙+학년+학기 추천",
            },
        ],
    },
    {
        "id": "S10",
        "persona": "3학년(트랙 고민)",
        "turns": [
            {
                "q": "머신러닝 수업은 뭘 배워?",
                "intent": "rag",
                "category": None,
                "answerable": True,
                "note": "키워드 밖(LLM 라우팅) — 과목 설명",
            },
        ],
    },
    # ── 페르소나 D: 졸업 앞둔 4학년 ───────────────────────────
    {
        "id": "S11",
        "persona": "졸업예정 4학년",
        "turns": [
            {
                "q": "졸업하려면 총 몇 학점 필요해?",
                "intent": "ask_year",
                "answerable": True,
                "note": "졸업 학점=년도민감 -> 학번 되묻기(설계상 정상)",
            },
            {
                "q": "23학번이야",
                "intent": "rag",
                "category": ["graduation"],
                "answerable": True,
                "expect_facts": ["120"],
                "expect_source": True,
                "note": "학번 답 -> 총 졸업학점 120",
            },
        ],
    },
    {
        "id": "S12",
        "persona": "졸업예정 4학년",
        "turns": [
            {
                "q": "외국어 졸업인증 기준이 어떻게 돼?",
                "intent": "rag",
                "category": ["graduation"],
                "answerable": True,
                "expect_source": True,
                "note": "외국어 졸업인증",
            },
        ],
    },
    {
        "id": "S13",
        "persona": "졸업예정 4학년",
        "turns": [
            {
                "q": "졸업작품은 몇 학년 과목이야?",
                "intent": "rag",
                "category": ["course"],
                "answerable": True,
                "expect_facts": ["4학년"],
                "expect_source": True,
                "note": "졸업작품 II 개설학년(과목)",
            },
        ],
    },
    # ── 페르소나 E: 학번 되묻기(년도-민감) ────────────────────
    {
        "id": "S14",
        "persona": "학번 미상 학생",
        "turns": [
            {
                "q": "졸업요건 알려줘",
                "intent": "ask_year",
                "answerable": True,
                "note": "년도-민감 + 학번 모름 -> 되묻기",
            },
            {
                "q": "23학번이요",
                "intent": "rag",
                "category": ["graduation"],
                "answerable": True,
                "expect_facts": ["전공필수"],
                "expect_source": True,
                "note": "학번 답 -> 원질문 이어서 rag",
            },
        ],
    },
    {
        "id": "S15",
        "persona": "21학번 복학생",
        "turns": [
            {
                "q": "21학번 졸업 전공필수 몇 학점이야?",
                "intent": "rag",
                "category": ["graduation"],
                "answerable": True,
                "expect_facts": ["38"],
                "expect_source": True,
                "note": "학번 명시 -> 2021 기준(전필38)",
            },
        ],
    },
    # ── 페르소나 F: 휴학/복학/사회봉사 ────────────────────────
    {
        "id": "S16",
        "persona": "휴학 고민 학생",
        "turns": [
            {
                "q": "휴학은 어떻게 신청해?",
                "intent": "rag",
                "category": ["leave_return"],
                "answerable": True,
                "expect_source": True,
                "note": "휴학 절차",
            },
        ],
    },
    {
        "id": "S17",
        "persona": "복학 예정자",
        "turns": [
            {
                "q": "복학 기간 언제야?",
                "intent": "rag",
                "category": ["leave_return", "academic_calendar"],
                "answerable": True,
                "expect_source": True,
                "note": "복합카테고리(복학+일정)",
            },
        ],
    },
    {
        "id": "S18",
        "persona": "봉사시간 필요 학생",
        "turns": [
            {
                "q": "사회봉사 몇 시간 해야 졸업해?",
                "intent": "rag",
                "category": ["social_service"],
                "answerable": True,
                "expect_source": True,
                "note": "사회봉사 이수기준",
            },
        ],
    },
    # ── 페르소나 G: 문의처 ────────────────────────────────────
    {
        "id": "S19",
        "persona": "문의처 찾는 학생",
        "turns": [
            {
                "q": "학과 사무실 전화번호 알려줘",
                "intent": "rag",
                "category": ["contact"],
                "answerable": False,
                "expect_facts": ["031-750-8668"],
                "expect_source": True,
                "note": "연락처 -> 가드레일 경로로 contacts.json 번호",
            },
        ],
    },
    # ── 페르소나 H: 이메일 리마인드(멀티턴) ───────────────────
    {
        "id": "S20",
        "persona": "리마인드 요청 학생",
        "turns": [
            {
                "q": "내일 오후 3시에 수강신청 리마인드 메일 보내줘",
                "intent": "reminder",
                "answerable": True,
                "note": "리마인드 시작 -> 이메일 되물음",
            },
            {
                "q": "hong@gachon.ac.kr 로 보내줘",
                "intent": "reminder",
                "answerable": True,
                "note": "이메일 -> 확인 요청",
            },
            {
                "q": "응 보내줘",
                "intent": "reminder",
                "answerable": True,
                "expect_facts": ["예약"],
                "note": "승인 -> 등록완료",
            },
        ],
    },
    # ── 가드레일(자료 범위 밖) — 지어내면 안 됨 ───────────────
    {
        "id": "S21",
        "persona": "기숙사 문의 학생",
        "turns": [
            {
                "q": "기숙사 벌점 몇 점이면 퇴사야?",
                "intent": "rag",
                "category": None,
                "answerable": False,
                "expect_source": True,
                "note": "자료 없음 -> 가드레일+생활관 안내",
            },
        ],
    },
    {
        "id": "S22",
        "persona": "일상 질문 학생",
        "turns": [
            {
                "q": "오늘 학식 메뉴 뭐야?",
                "intent": "rag",
                "category": None,
                "answerable": False,
                "expect_source": False,
                "note": "범위 밖 -> 가드레일",
            },
        ],
    },
    {
        "id": "S23",
        "persona": "장학금 문의 학생",
        "turns": [
            {
                "q": "국가장학금 신청 조건 자세히 알려줘",
                "intent": "rag",
                "category": None,
                "answerable": False,
                "expect_source": True,
                "note": "장학 세부(3순위 범위밖) -> 가드레일+문의처/링크",
            },
        ],
    },
    {
        "id": "S24",
        "persona": "동아리 관심 학생",
        "turns": [
            {
                "q": "IT 동아리 뭐뭐 있어?",
                "intent": "rag",
                "category": None,
                "answerable": False,
                "expect_source": True,
                "note": "동아리(범위밖) -> 가드레일+학생지원팀",
            },
        ],
    },
    {
        "id": "S25",
        "persona": "황당 질문 학생",
        "turns": [
            {
                "q": "내일 날씨 어때?",
                "intent": "rag",
                "category": None,
                "answerable": False,
                "expect_source": False,
                "note": "완전 무관 -> 가드레일",
            },
        ],
    },
]


def flat_turns():
    """(scenario_id, persona, turn_index, turn) 평탄화."""
    for sc in SCENARIOS:
        for i, t in enumerate(sc["turns"]):
            yield sc["id"], sc["persona"], i, t


if __name__ == "__main__":
    n_sc = len(SCENARIOS)
    n_turn = sum(len(s["turns"]) for s in SCENARIOS)
    n_guard = sum(1 for _, _, _, t in flat_turns() if not t["answerable"])
    print(f"시나리오 {n_sc}개, 총 턴 {n_turn}개 (가드레일 기대 {n_guard}개)")
    from collections import Counter

    intents = Counter(t["intent"] for _, _, _, t in flat_turns())
    print("intent 분포:", dict(intents))
