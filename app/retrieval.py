"""pgvector 유사도 검색 (RAG)."""
from app import embeddings


def search(conn, query: str, k: int = 4) -> list[dict]:
    """질의와 가장 유사한 문서 청크 top-k 반환."""
    qvec = embeddings.embed_query(query)
    # 파이썬 리스트를 pgvector 리터럴로 만들어 ::vector 캐스팅 (SELECT 표현식엔 타입 힌트가 없음)
    qlit = "[" + ",".join(map(str, qvec)) + "]"
    rows = conn.execute(
        """
        SELECT source, page, content,
               1 - (embedding <=> %s::vector) AS score
        FROM documents
        ORDER BY embedding <=> %s::vector
        LIMIT %s
        """,
        (qlit, qlit, k),
    ).fetchall()
    return [
        {"source": r[0], "page": r[1], "content": r[2], "score": float(r[3])}
        for r in rows
    ]
