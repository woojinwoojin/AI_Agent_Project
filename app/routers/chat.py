"""/api/chat — LangGraph 에이전트 (router → rag/tool → response)."""
from fastapi import APIRouter
from langchain_core.messages import HumanMessage
from pydantic import BaseModel

from app.graph.graph import get_graph
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
