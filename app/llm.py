"""Upstage Solar LLM 클라이언트."""
from openai import OpenAI

from app import config

_client = OpenAI(api_key=config.UPSTAGE_API_KEY, base_url=config.UPSTAGE_BASE_URL)


def chat(messages: list[dict], temperature: float = 0.2) -> str:
    """Solar 채팅 완성. messages=[{role, content}, ...]"""
    resp = _client.chat.completions.create(
        model=config.LLM_MODEL,
        messages=messages,
        temperature=temperature,
    )
    return resp.choices[0].message.content
