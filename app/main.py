import logging
import sys
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app import db
from app.routers import chat

# Windows 콘솔/리다이렉트 시 로케일 코드페이지(cp949)로 인코딩돼 로그의 한글이
# 깨지는 문제 방지 — 로그(질문·문서 source 등 한글 포함)를 그대로 읽을 수 있어야 함.
if sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

# app.rag / app.retrieval 로거(router 판단, 검색 범위, rerank 결과, guardrail)를 출력.
# basicConfig는 이 시점엔 이미 다른 라이브러리(langchain 등) import로 root 로거에
# 핸들러가 붙어있어 no-op이 될 수 있어, "app" 로거에 직접 핸들러를 붙인다.
_app_logger = logging.getLogger("app")
if not _app_logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
    _app_logger.addHandler(_handler)
_app_logger.setLevel(logging.INFO)
_app_logger.propagate = False

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
