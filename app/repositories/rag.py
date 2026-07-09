"""RAG 검색 리포지토리 (pgvector). 병합 시 이 구현만 Supabase로 교체하면 됨."""
import asyncio

from app import db, retrieval


class RagRepository:
    async def search_similar(self, query: str, k: int = 4) -> list[dict]:
        def _search():
            conn = db.connect()
            try:
                return retrieval.search(conn, query, k=k)
            finally:
                conn.close()

        return await asyncio.to_thread(_search)


_repo: RagRepository | None = None


def get_rag_repository() -> RagRepository:
    global _repo
    if _repo is None:
        _repo = RagRepository()
    return _repo
