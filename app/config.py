"""애플리케이션 설정 (.env 로드)."""
import os
import pathlib
from dotenv import load_dotenv

ROOT = pathlib.Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

# Upstage (Solar) — OpenAI 호환 엔드포인트
UPSTAGE_API_KEY = os.getenv("UPSTAGE_API_KEY", "")
UPSTAGE_BASE_URL = os.getenv("UPSTAGE_BASE_URL", "https://api.upstage.ai/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "solar-pro3")

# Upstage 임베딩 모델 (solar-embedding-1-large 계열, 4096차원)
EMBED_MODEL_PASSAGE = os.getenv("EMBED_MODEL_PASSAGE", "embedding-passage")
EMBED_MODEL_QUERY = os.getenv("EMBED_MODEL_QUERY", "embedding-query")
EMBED_DIM = int(os.getenv("EMBED_DIM", "4096"))

# PostgreSQL + pgvector
DATABASE_URL = os.getenv(
    "DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/gachon_ai"
)

# 데이터 경로
STRUCTURED_DIR = ROOT / "output" / "structured"
PARSED_DIR = ROOT / "output" / "parsed"

# 가드레일: RAG 최고 점수가 이 값보다 낮으면 '자료 없음'으로 보고 문의처 안내
# (score = 코사인유사도 + 키워드보너스. 관련 문서는 대략 0.4~0.5+)
GUARDRAIL_MIN_SCORE = float(os.getenv("GUARDRAIL_MIN_SCORE", "0.35"))

DEBUG = os.getenv("DEBUG", "false").lower() == "true"
