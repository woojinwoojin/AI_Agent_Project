import json
import re
from pathlib import Path
from typing import Any
from app.services.llm_service import generate_answer_with_llm



DOCS_PATH = Path("data/docs.json")


def load_docs() -> list[dict[str, Any]]:
    if not DOCS_PATH.exists():
        raise FileNotFoundError(f"docs.json 파일을 찾을 수 없습니다: {DOCS_PATH}")

    with DOCS_PATH.open("r", encoding="utf-8") as file:
        return json.load(file)


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def tokenize(query: str) -> list[str]:
    query = normalize_text(query)

    stopwords = {
        "은", "는", "이", "가", "을", "를", "에", "에서", "으로", "로",
        "언제", "어떻게", "뭐", "무엇", "알려줘", "궁금해", "해요", "인가요",
        "좀", "관련", "대해"
    }

    tokens = re.findall(r"[가-힣a-zA-Z0-9]+", query)
    return [token for token in tokens if token not in stopwords and len(token) >= 2]


def score_doc(query: str, doc: dict[str, Any]) -> int:
    tokens = tokenize(query)

    title = normalize_text(doc.get("title", ""))
    category = normalize_text(doc.get("category", ""))
    area = normalize_text(doc.get("area", ""))
    content = normalize_text(doc.get("content", ""))
    keywords = " ".join(doc.get("keywords", []))
    keywords = normalize_text(keywords)

    score = 0

    for token in tokens:
        if token in title:
            score += 5
        if token in area:
            score += 4
        if token in category:
            score += 3
        if token in keywords:
            score += 4
        if token in content:
            score += 2

    # 자주 나오는 질문에 대한 보정
    if any(word in query for word in ["수강신청", "수강 신청", "신청"]):
        if doc.get("category") in ["course_registration", "academic_calendar"]:
            score += 5

    if any(word in query for word in ["졸업", "졸업요건", "졸업 학점", "졸업학점"]):
        if doc.get("category") == "graduation":
            score += 5

    if any(word in query for word in ["복학", "휴학"]):
        if doc.get("category") in ["leave_return", "academic_calendar"]:
            score += 5

    if any(word in query for word in ["사회봉사", "봉사"]):
        if doc.get("category") == "social_service":
            score += 5

    if any(word in query for word in ["전과", "전공 변경", "전공변경"]):
        if doc.get("category") == "major_change":
            score += 5

    if any(word in query for word in ["재입학"]):
        if doc.get("category") == "readmission":
            score += 5

    return score


def retrieve_docs(query: str, top_k: int = 3) -> list[dict[str, Any]]:
    docs = load_docs()

    scored_docs = []
    for doc in docs:
        score = score_doc(query, doc)
        if score > 0:
            scored_docs.append((score, doc))

    scored_docs.sort(key=lambda item: item[0], reverse=True)

    return [doc for _, doc in scored_docs[:top_k]]


def build_answer(query: str, docs: list[dict[str, Any]]) -> dict[str, Any]:
    if not docs:
        return {
            "answer": (
                "현재 등록된 자료에서는 해당 내용을 확인하기 어렵습니다.\n\n"
                "정확한 내용은 학과사무실, 교무처 또는 관련 담당 부서에 문의하는 것을 권장합니다."
            ),
            "sources": [],
            "requires_confirmation": False,
            "type": "guardrail",
        }

    answer_parts = []

    answer_parts.append("관련 자료를 기준으로 정리하면 다음과 같아요.\n")

    for index, doc in enumerate(docs, start=1):
        title = doc.get("title", "제목 없음")
        content = doc.get("content", "")

        answer_parts.append(f"{index}. {title}")
        answer_parts.append(content)

    answer_parts.append(
        "\n※ 위 내용은 업로드된 공식 자료 기반 요약이며, 실제 신청·졸업판정 등은 학교 공식 시스템 또는 담당 부서 확인이 필요합니다."
    )

    sources = [
        {
            "title": doc.get("title"),
            "source": doc.get("source"),
            "source_page": doc.get("source_page"),
            "category": doc.get("category"),
        }
        for doc in docs
    ]

    return {
        "answer": "\n\n".join(answer_parts),
        "sources": sources,
        "requires_confirmation": False,
        "type": "rag_answer",
    }


def rag_answer(query: str) -> dict[str, Any]:
    retrieved_docs = retrieve_docs(query, top_k=3)
    return build_answer(query, retrieved_docs)

def rag_answer(query: str) -> dict:
    retrieved_docs = retrieve_docs(query, top_k=3)
    return generate_answer_with_llm(query, retrieved_docs)