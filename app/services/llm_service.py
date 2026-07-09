import os
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    api_key=os.getenv("UPSTAGE_API_KEY"),
    base_url="https://api.upstage.ai/v1",
)


def format_context(docs: list[dict[str, Any]]) -> str:
    if not docs:
        return ""

    context_blocks = []

    for index, doc in enumerate(docs, start=1):
        context_blocks.append(
            f"""
[자료 {index}]
제목: {doc.get("title", "제목 없음")}
분류: {doc.get("category", "unknown")}
내용: {doc.get("content", "")}
출처: {doc.get("source", "출처 없음")} {doc.get("source_page", "")}
""".strip()
        )

    return "\n\n".join(context_blocks)


def generate_answer_with_llm(
    query: str,
    docs: list[dict[str, Any]],
) -> dict[str, Any]:
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

    context = format_context(docs)

    system_prompt = """
너는 가천대학교 인공지능학과 학생을 돕는 친근한 학교생활 안내 Agent다.

답변 원칙:
1. 반드시 제공된 [검색된 자료]만 근거로 답변한다.
2. 자료에 없는 내용은 추측하지 않는다.
3. 사용자가 이해하기 쉽게 핵심부터 짧게 정리한다.
4. 학사일정, 졸업요건, 수강신청 등 중요한 정보는 최종적으로 학교 공식 시스템 또는 담당 부서 확인이 필요하다고 안내한다.
5. 답변은 한국어로 작성한다.
""".strip()

    user_prompt = f"""
[사용자 질문]
{query}

[검색된 자료]
{context}

위 자료를 바탕으로 사용자 질문에 답변해줘.
답변 형식은 다음을 따라줘.

- 먼저 핵심 답변
- 필요하면 추가 설명
- 마지막에 "확인 필요" 또는 "유의사항"을 짧게 안내
""".strip()

    response = client.chat.completions.create(
        model="solar-pro3",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        stream=False,
    )

    answer = response.choices[0].message.content

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
        "answer": answer,
        "sources": sources,
        "requires_confirmation": False,
        "type": "rag_llm_answer",
    }