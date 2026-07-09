from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel


app = FastAPI(title="AI 학과 길잡이 MVP")

app.mount("/static", StaticFiles(directory="app/static"), name="static")

templates = Jinja2Templates(directory="app/templates")


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
async def chat(request: ChatRequest):
    user_message = request.message

    # 1단계에서는 Agent 없이 임시 응답만 반환
    return {
        "answer": f"입력한 질문을 확인했어요: {user_message}\n\n다음 단계에서 이 질문을 Agent가 분류하도록 만들 예정입니다.",
        "sources": [],
        "requires_confirmation": False,
    }


@app.get("/health")
async def health_check():
    return {"status": "ok"}