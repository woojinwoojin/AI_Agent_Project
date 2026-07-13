"""Upstage Solar LLM 클라이언트."""

from collections.abc import Iterator

from openai import OpenAI

from app import config

_client = OpenAI(api_key=config.UPSTAGE_API_KEY, base_url=config.UPSTAGE_BASE_URL)


def chat(messages: list[dict], temperature: float = 0.0) -> str:
    """Solar 채팅 완성. messages=[{role, content}, ...]"""
    resp = _client.chat.completions.create(
        model=config.LLM_MODEL,
        messages=messages,
        temperature=temperature,
    )
    return resp.choices[0].message.content


def chat_stream(messages: list[dict], temperature: float = 0.0) -> Iterator[str]:
    """Solar 채팅 응답을 토큰 단위로 스트리밍."""
    stream = _client.chat.completions.create(
        model=config.LLM_MODEL,
        messages=messages,
        temperature=temperature,
        stream=True,
    )

    for chunk in stream:
        if not chunk.choices:
            continue

        delta = chunk.choices[0].delta
        content = getattr(delta, "content", None)

        if content:
            yield content
