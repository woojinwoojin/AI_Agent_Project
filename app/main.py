from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from app import db, llm, retrieval


app = FastAPI(title="AI 학과 길잡이 MVP")

app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")


SYSTEM_PROMPT = """
너는 가천대학교 인공지능학과 학생을 돕는 학사 안내 AI야.

답변 원칙:
1. 반드시 [참고자료]에 있는 내용만 근거로 답변해.
2. 참고자료에 없는 내용은 절대 지어내지 마.
3. 자료에서 확인되지 않으면 "제 자료에서 확인되지 않습니다"라고 말해.
4. 학사일정, 졸업요건, 수강신청, 휴학/복학 등 중요한 내용은 학교 공식 시스템 또는 담당 부서 확인이 필요하다고 안내해.
5. 답변은 친근한 선배 말투로 하되, 너무 장황하지 않게 정리해.
6. 가능하면 핵심 답변 → 추가 설명 → 유의사항 순서로 답변해.
7. 출처 목록은 시스템이 따로 표시하므로 본문에 출처를 길게 반복하지 않아도 돼.
""".strip()


class ChatRequest(BaseModel):
    message: str


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={},
    )


@app.post("/api/chat")
def chat(req: ChatRequest):
    conn = db.connect()

    try:
        hits = retrieval.search(conn, req.message, k=4)
    finally:
        conn.close()

    # 관련도 기준
    # vector_score는 순수 의미 유사도, score는 키워드 보정 포함 점수
    if not hits:
        return {
            "answer": (
                "제 자료에서 확인되지 않습니다.\n\n"
                "정확한 내용은 학과사무실, 교무처 또는 관련 담당 부서에 문의하는 것을 권장해요."
            ),
            "sources": [],
            "requires_confirmation": False,
            "type": "guardrail",
        }

    best_hit = hits[0]
    best_vector_score = best_hit.get("vector_score", 0)
    best_score = best_hit.get("score", 0)

    if best_vector_score < 0.25 and best_score < 0.32:
        return {
            "answer": (
                "제 자료에서 확인되지 않습니다.\n\n"
                "제가 가진 자료는 학사일정, 수강신청, 졸업요건, 사회봉사, 휴학/복학, 전과/재입학 중심이라서 "
                "해당 질문은 정확히 답변하기 어려워요. 관련 부서나 학과사무실에 문의하는 것을 권장해요."
            ),
            "sources": [],
            "requires_confirmation": False,
            "type": "guardrail",
        }

    context = "\n\n".join(
        f"""
[자료 {i + 1}]
출처: {h["source"]} {h.get("page") or ""}
내용:
{h["content"][:1200]}
""".strip()
        for i, h in enumerate(hits)
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"[참고자료]\n{context}\n\n[사용자 질문]\n{req.message}",
        },
    ]

    answer = llm.chat(messages)

    sources = [
        {
            "source": h["source"],
            "page": h.get("page"),
            "score": round(h["score"], 3),
            "vector_score": round(h.get("vector_score", 0), 3),
        }
        for h in hits
    ]

    return {
        "answer": answer,
        "sources": sources,
        "requires_confirmation": False,
        "type": "rag_llm_answer",
    }


@app.get("/health")
def health():
    status = {"status": "ok", "db": "unknown"}

    try:
        conn = db.connect()
        count = conn.execute("SELECT count(*) FROM documents").fetchone()[0]
        conn.close()
        status["db"] = "ok"
        status["documents"] = count
    except Exception as e:
        status["status"] = "degraded"
        status["db"] = f"error: {e}"

    return status