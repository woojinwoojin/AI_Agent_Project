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
# score = reranker 가중합(vector*0.65+keyword*0.15+category*0.10+priority*0.05+recency*0.05).
# 실측: 관련 질문 0.62+, 무관 질문 ≤0.37 → 0.40 으로 분리. (리랭크 도입 전 0.35에서 상향)
GUARDRAIL_MIN_SCORE = float(os.getenv("GUARDRAIL_MIN_SCORE", "0.40"))

DEBUG = os.getenv("DEBUG", "false").lower() == "true"
