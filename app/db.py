"""PostgreSQL + pgvector 연결 및 스키마."""
import psycopg
from pgvector.psycopg import register_vector

from app import config


def connect():
    """pgvector 등록된 psycopg 연결 반환."""
    conn = psycopg.connect(config.DATABASE_URL, autocommit=True)
    conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    register_vector(conn)
    return conn


def init_schema(conn):
    """RAG 문서/과목/졸업요건 테이블 생성."""
    dim = config.EMBED_DIM
    conn.execute(
        f"""
        CREATE TABLE IF NOT EXISTS documents (
            id          BIGSERIAL PRIMARY KEY,
            source      TEXT NOT NULL,
            page        INT,
            category    TEXT,
            category_l1 TEXT,
            category_l2 TEXT,
            content     TEXT NOT NULL,
            metadata    JSONB DEFAULT '{{}}'::jsonb,
            embedding   VECTOR({dim})
        )
        """
    )
    # 기존 DB(컬럼 없이 생성된 경우) 대응: 카테고리 컬럼을 안전하게 추가한다.
    conn.execute("ALTER TABLE documents ADD COLUMN IF NOT EXISTS category_l1 TEXT")
    conn.execute("ALTER TABLE documents ADD COLUMN IF NOT EXISTS category_l2 TEXT")
    # 카테고리 필터 검색(2단계) 대비 인덱스
    conn.execute("CREATE INDEX IF NOT EXISTS idx_documents_category_l1 ON documents (category_l1)")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS courses (
            id            BIGSERIAL PRIMARY KEY,
            교과목명       TEXT NOT NULL,
            이수구분       TEXT,
            학점          INT,
            이론          INT,
            실습          INT,
            개설학년       TEXT,
            개설학기       TEXT,
            트랙          TEXT,
            교육과정_연도   INT
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS graduation_requirements (
            id            BIGSERIAL PRIMARY KEY,
            교육과정_연도   INT,
            data          JSONB NOT NULL
        )
        """
    )
    # 주의: pgvector HNSW/IVFFlat 인덱스는 최대 2000차원까지만 지원한다.
    # Upstage 임베딩은 4096차원이고 문서 수가 적으므로, 인덱스 없이
    # 정확(exact) 코사인 검색(<=>)을 사용한다. (데이터가 커지면 halfvec 전환 고려)
