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


_SELECT_COLS = (
    "source, page, content, category_l1, priority, academic_year, keywords, "
    "1 - (embedding <=> %s::vector) AS vector_score"
)


def _fetch_candidates(conn, qlit: str, candidates: int, category_l1: str | None):
    """pgvector 유사도 상위 후보 조회. category_l1 지정 시 해당 카테고리로 필터.
    is_active=TRUE 문서만 대상으로 한다(멘토링 결과 §9)."""
    if category_l1:
        return conn.execute(
            f"""
            SELECT {_SELECT_COLS}
            FROM documents
            WHERE is_active = TRUE AND category_l1 = %s
            ORDER BY embedding <=> %s::vector
            LIMIT %s
            """,
            (qlit, category_l1, qlit, candidates),
        ).fetchall()
    return conn.execute(
        f"""
        SELECT {_SELECT_COLS}
        FROM documents
        WHERE is_active = TRUE
        ORDER BY embedding <=> %s::vector
        LIMIT %s
        """,
        (qlit, qlit, candidates),
    ).fetchall()


def search(
    conn,
    query: str,
    k: int = 4,
    candidates: int = 12,
    category_l1: str | None = None,
) -> list[dict]:
    """질의와 가장 유사한 문서 청크 반환.

    1. query를 확장한다.
    2. pgvector로 후보 문서를 넉넉히 가져온다. (category_l1 지정 시 그 카테고리 안에서만)
       - 멘토 원칙: 카테고리로 검색 공간을 먼저 좁힌다.
       - 안전망: 카테고리 필터 결과가 비면 전체에서 다시 검색(오분류로 답을 놓치지 않도록).
    3. lightweight reranker(vector/keyword/category/priority/recency)로 재정렬한다.
    4. 중복 문서를 제거한다.
    """
    from app.services import reranker  # 지연 임포트(순환 방지)

    expanded_query = expand_query(query)
    qvec = embeddings.embed_query(expanded_query)

    qlit = "[" + ",".join(map(str, qvec)) + "]"

    rows = _fetch_candidates(conn, qlit, candidates, category_l1)
    used_category = category_l1
    if category_l1 and not rows:
        # 카테고리 필터로 아무것도 못 찾음 → 전체 검색으로 폴백
        rows = _fetch_candidates(conn, qlit, candidates, None)
        used_category = None

    hits = [
        {
            "source": row[0],
            "page": row[1],
            "content": row[2],
            "category_l1": row[3],
            "priority": row[4],
            "academic_year": row[5],
            "keywords": row[6],
            "vector_score": float(row[7]),
        }
        for row in rows
    ]

    hits = deduplicate_hits(hits)
    hits = reranker.rerank(query, used_category, hits)

    if hits:
        hits[0]["_filtered_by"] = used_category  # 디버깅: 어떤 카테고리로 필터했는지
    return hits[:k]