from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app import db
from app.routers import chat

BASE_DIR = Path(__file__).resolve().parent

app = FastAPI(title="AI 학과 길잡이 MVP")

app.mount(
    "/static",
    StaticFiles(directory=BASE_DIR / "static"),
    name="static",
)

templates = Jinja2Templates(directory=BASE_DIR / "templates")
# /api/chat 은 LangGraph 에이전트 라우터가 담당 (router → rag/tool → response)
app.include_router(chat.router)

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={},
    )


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
