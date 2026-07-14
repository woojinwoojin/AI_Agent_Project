"""공식 링크 안내(build_link_hint / _detect_link_topics) 규칙 테스트.

라우터 category_l1(coarse) + 질문 텍스트 키워드(fine) 하이브리드 매칭이
의도대로 동작하는지, 노출 개수 제한과 무관 질문 미노출을 고정한다.
"""

from app.core.prompts import OFFICIAL_LINKS, build_link_hint
from app.graph.nodes import _detect_link_topics


class _Msg:
    def __init__(self, content: str):
        self.content = content


def _state(text: str, categories=None):
    return {"messages": [_Msg(text)], "category_l1": categories}


def test_category_match():
    """라우터가 매긴 category_l1만으로도 매칭된다."""
    assert _detect_link_topics(_state("수강신청 언제야?", ["academic_calendar"])) == [
        "academic_calendar"
    ]
    assert _detect_link_topics(_state("졸업하려면 몇 학점?", ["graduation"])) == ["graduation"]
    assert _detect_link_topics(_state("휴학 어떻게 해?", ["leave_return"])) == ["leave_return"]


def test_keyword_match_beyond_categories():
    """6개 라우터 카테고리 밖 주제도 키워드로 잡힌다."""
    assert _detect_link_topics(_state("등록금 언제까지 내?")) == ["tuition"]
    assert _detect_link_topics(_state("국가장학금 어떻게 신청해?")) == ["scholarship"]
    assert _detect_link_topics(_state("재학증명서 어디서 떼?")) == ["certificate"]
    assert _detect_link_topics(_state("기숙사 벌점 기준 알려줘")) == ["dormitory"]
    assert _detect_link_topics(_state("교환학생 가고 싶어")) == ["intl"]


def test_calendar_keyword_wins_priority():
    """'예비수강신청 일자'는 category=course여도 학사일정이 먼저 노출된다."""
    topics = _detect_link_topics(_state("예비수강신청 일자 알려줘", ["course"]))
    assert topics[0] == "academic_calendar"
    assert "course" in topics


def test_smalltalk_no_link():
    """인사/무관 질문에는 링크를 붙이지 않는다."""
    assert _detect_link_topics(_state("안녕하세요")) == []
    assert build_link_hint(_detect_link_topics(_state("안녕하세요"))) == ""


def test_hint_shows_all_matched_with_desc():
    """매칭된 링크는 제한 없이 전부 노출되고, 각 링크에 설명(desc)과 URL이 붙는다."""
    keys = list(OFFICIAL_LINKS.keys())
    hint = build_link_hint(keys)
    # 항목 줄("  - ...") 수가 매칭 개수와 같아야 한다(캡 없음).
    assert hint.count("\n  - ") == len(keys)
    for k in keys:
        spec = OFFICIAL_LINKS[k]
        assert spec["desc"] in hint
        assert spec["url"] in hint


def test_all_links_have_desc_and_gachon_urls():
    """모든 링크가 설명을 갖고, 가천대 공식 도메인 절대경로인지(오타 방지)."""
    for spec in OFFICIAL_LINKS.values():
        assert spec["desc"]
        assert spec["url"].startswith("https://www.gachon.ac.kr/")
