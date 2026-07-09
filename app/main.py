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
아래 참고자료에 근거해서만 답변해.
자료에 없는 내용은 지어내지 말고 '제 자료에서 확인되지 않습니다'라고 답해.
학사일정, 졸업요건, 수강신청 등 중요한 정보는 학교 공식 시스템 또는 담당 부서 확인이 필요하다고 안내해.
친근한 선배 말투로 간결하게 답변해.
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

    # 낮은 유사도일 경우 가드레일 처리
    if not hits or hits[0]["score"] < 0.25:
        return {
            "answer": (
                "제 자료에서 확인되지 않습니다.\n\n"
                "정확한 내용은 학과사무실, 교무처 또는 관련 담당 부서에 문의하는 것을 권장해요."
            ),
            "sources": [],
            "requires_confirmation": False,
            "type": "guardrail",
        }

    context = "\n\n".join(
        f"[자료 {i + 1}]\n출처: {h['source']}\n내용: {h['content']}"
        for i, h in enumerate(hits)
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"[참고자료]\n{context}\n\n[질문]\n{req.message}",
        },
    ]

    answer = llm.chat(messages)

    sources = [
        {
            "source": h["source"],
            "page": h.get("page"),
            "score": h["score"],
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