"""Upstage 임베딩 클라이언트 (OpenAI 호환)."""

from openai import OpenAI

from app import config

_client = OpenAI(api_key=config.UPSTAGE_API_KEY, base_url=config.UPSTAGE_BASE_URL)


def embed_passages(texts: list[str]) -> list[list[float]]:
    """문서(passage) 임베딩. 적재용."""
    if not texts:
        return []
    resp = _client.embeddings.create(model=config.EMBED_MODEL_PASSAGE, input=texts)
    return [d.embedding for d in resp.data]


def embed_query(text: str) -> list[float]:
    """질의(query) 임베딩. 검색용."""
    resp = _client.embeddings.create(model=config.EMBED_MODEL_QUERY, input=[text])
    return resp.data[0].embedding
