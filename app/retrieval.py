"""pgvector 기반 RAG 검색 + 간단 키워드 보정."""
import re

from app import embeddings


SYNONYM_MAP = {
    "수강": ["수강신청", "예비수강신청", "수강정정", "수강과목포기"],
    "수강신청": ["예비수강신청", "수강정정", "수강학점"],
    "졸업": ["졸업요건", "졸업학점", "졸업인증", "외국어능력 졸업인증"],
    "졸업학점": ["졸업요건", "전공필수", "전공선택", "교양필수"],
    "복학": ["복학기간", "휴학연기", "학적변동"],
    "휴학": ["미등록휴학", "등록휴학", "휴학연기", "학기중휴학"],
    "사회봉사": ["봉사활동", "30시간", "P/F", "졸업인증"],
    "봉사": ["사회봉사", "봉사활동", "30시간"],
    "전과": ["전공변경", "전과신청기간"],
    "재입학": ["재입학신청기간", "제적", "퇴학"],
    "외국어": ["외국어능력 졸업인증", "국제어학원"],
}


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def tokenize(text: str) -> list[str]:
    text = normalize(text)
    tokens = re.findall(r"[가-힣a-zA-Z0-9]+", text)

    stopwords = {
        "언제", "뭐야", "무엇", "어떻게", "알려줘", "궁금해",
        "관련", "대해", "좀", "나는", "제가", "하면", "되나요",
    }

    return [token for token in tokens if len(token) >= 2 and token not in stopwords]


def expand_query(query: str) -> str:
    expanded_terms = []

    for key, values in SYNONYM_MAP.items():
        if key in query:
            expanded_terms.extend(values)

    if not expanded_terms:
        return query

    return f"{query} {' '.join(expanded_terms)}"


def keyword_bonus(query: str, text: str) -> float:
    query_tokens = tokenize(query)
    text = normalize(text)

    bonus = 0.0

    for token in query_tokens:
        if token in text:
            bonus += 0.03

    return min(bonus, 0.15)


def deduplicate_hits(hits: list[dict]) -> list[dict]:
    seen = set()
    unique_hits = []

    for hit in hits:
        key = (
            hit.get("source"),
            hit.get("page"),
            normalize(hit.get("content", ""))[:120],
        )

        if key in seen:
            continue

        seen.add(key)
        unique_hits.append(hit)

    return unique_hits


def search(conn, query: str, k: int = 4, candidates: int = 12) -> list[dict]:
    """질의와 가장 유사한 문서 청크 반환.

    1. query를 확장한다.
    2. pgvector로 후보 문서를 넉넉히 가져온다.
    3. 원질문 키워드가 실제 문서에 포함되면 점수를 보정한다.
    4. 중복 문서를 제거한다.
    """

    expanded_query = expand_query(query)
    qvec = embeddings.embed_query(expanded_query)

    qlit = "[" + ",".join(map(str, qvec)) + "]"

    rows = conn.execute(
        """
        SELECT source, page, content,
               1 - (embedding <=> %s::vector) AS vector_score
        FROM documents
        ORDER BY embedding <=> %s::vector
        LIMIT %s
        """,
        (qlit, qlit, candidates),
    ).fetchall()

    hits = []

    for row in rows:
        source = row[0]
        page = row[1]
        content = row[2]
        vector_score = float(row[3])

        text_for_keyword = f"{source} {page} {content}"
        bonus = keyword_bonus(query, text_for_keyword)
        final_score = vector_score + bonus

        hits.append(
            {
                "source": source,
                "page": page,
                "content": content,
                "vector_score": vector_score,
                "keyword_bonus": bonus,
                "score": final_score,
            }
        )

    hits = deduplicate_hits(hits)
    hits.sort(key=lambda item: item["score"], reverse=True)

    return hits[:k]