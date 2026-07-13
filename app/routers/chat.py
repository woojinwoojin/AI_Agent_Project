"""/api/chat — LangGraph 에이전트 (router → rag/tool → response)."""

import json

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage
from pydantic import BaseModel

from app import llm
from app.graph.edges import route_by_intent
from app.graph.graph import get_graph
from app.graph.nodes import build_response_inputs, rag_node, router_node, tool_node
from app.graph.state import create_initial_state

router = APIRouter(prefix="/api", tags=["chat"])


class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None


def _rag_sources(docs: list[dict]) -> list[dict]:
    """검색 문서를 출처별로 묶어 프론트가 쓰는 {source, page, score} 형태로."""
    best: dict[str, dict] = {}
    for d in docs:
        src = d.get("source") or "출처 없음"
        score = d.get("score", 0)
        if src not in best or score > best[src]["score"]:
            best[src] = {
                "source": src,
                "page": d.get("page"),
                "score": round(float(score), 3),
            }
    return sorted(best.values(), key=lambda s: s["score"], reverse=True)


def _contact_sources(contact: dict | None) -> list[dict]:
    """가드레일 안내의 문의처/링크를 프론트 sources({source}) 형태로."""
    if not contact:
        return []
    srcs: list[dict] = []
    if contact.get("matched"):
        phone = (contact.get("담당") or {}).get("전화") or contact.get("대표전화")
        label = contact["부서"] + (f" ☎ {phone}" if phone else "")
        srcs.append({"source": label})
    for link in contact.get("링크", []):
        srcs.append({"source": f"{link['이름']} ({link['URL']})"})
    return srcs


@router.post("/chat")
async def chat(req: ChatRequest):
    graph = get_graph()
    state = create_initial_state(
        session_id=req.session_id or "default",
        messages=[HumanMessage(content=req.message)],
    )
    result = await graph.ainvoke(state)

    answer = result["messages"][-1].content
    intent = result.get("intent")

    # 출처 표기 (프론트 chat.js는 {source, page, score} 객체 배열을 기대)
    sources: list[dict] = []
    response_type = "chat_answer"

    contact = result.get("contact")

    if intent == "rag" and result.get("guardrail"):
        response_type = "guardrail"
        sources = _contact_sources(contact)
    elif intent == "rag":
        docs = result.get("retrieved_docs", [])
        sources = _rag_sources(docs)
        response_type = "rag_llm_answer"
    elif intent == "tool":
        tr = result.get("tool_result") or {}
        data = tr.get("data") if isinstance(tr, dict) else None
        if isinstance(data, dict) and data.get("출처"):
            # page/score는 생략 → 프론트에서 undefined로 처리되어 표기 안 됨
            sources = [{"source": data["출처"]}]
        response_type = "tool_answer"

    return {
        "answer": answer,
        "type": response_type,
        "intent": intent,
        "tool_name": result.get("tool_name"),
        "sources": sources,
        "contact": contact,
    }


def sse_event(event: str, data: dict) -> str:
    return f"event: {event}\n" f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


async def prepare_state_without_response(req: ChatRequest):
    """LangGraph의 response_node 직전까지 실행한다.

    즉,
    router → rag/tool
    까지만 수행하고, 최종 LLM 응답 생성은 stream=True로 따로 처리한다.
    """
    state = create_initial_state(
        session_id=req.session_id or "default",
        messages=[HumanMessage(content=req.message)],
    )

    router_update = await router_node(state)
    state.update(router_update)

    route = route_by_intent(state)

    if route == "rag":
        rag_update = await rag_node(state)
        state.update(rag_update)

    elif route == "tool":
        tool_update = await tool_node(state)
        state.update(tool_update)

    return state


@router.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    async def event_generator():
        yield sse_event("status", {"message": "질문을 분석하는 중이에요."})

        state = await prepare_state_without_response(req)

        intent = state.get("intent")
        sources: list[dict] = []
        response_type = "chat_answer"

        contact = state.get("contact")

        if intent == "rag" and state.get("guardrail"):
            response_type = "guardrail"
            sources = _contact_sources(contact)

        elif intent == "rag":
            docs = state.get("retrieved_docs", [])
            sources = _rag_sources(docs)
            response_type = "rag_llm_answer"

        elif intent == "tool":
            tr = state.get("tool_result") or {}
            data = tr.get("data") if isinstance(tr, dict) else None
            if isinstance(data, dict) and data.get("출처"):
                sources = [{"source": data["출처"]}]
            response_type = "tool_answer"

        yield sse_event(
            "meta",
            {
                "type": response_type,
                "intent": intent,
                "tool_name": state.get("tool_name"),
                "sources": sources,
                "contact": contact,
            },
        )

        system_prompt, user_input = build_response_inputs(state)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input},
        ]

        try:
            for token in llm.chat_stream(messages):
                yield sse_event("delta", {"text": token})

            yield sse_event("done", {"message": "complete"})

        except Exception as e:
            yield sse_event(
                "error",
                {"message": f"스트리밍 중 오류가 발생했습니다: {str(e)}"},
            )

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


"""
@router.get("/chat/stream-test")
async def chat_stream_test():
    async def event_generator():
        for token in ["안녕", "하세요. ", "이건 ", "SSE ", "테스트입니다."]:
            print("SEND TOKEN:", token)
            yield sse_event("delta", {"text": token})
            await asyncio.sleep(1)

        yield sse_event("done", {"message": "complete"})

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
"""
