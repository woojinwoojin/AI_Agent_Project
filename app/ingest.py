"""
데이터 적재: 정형 카탈로그 + 졸업요건 + RAG 청크 → PostgreSQL/pgvector

사용법:
    python -m app.ingest
"""
import json

from app import config, db, embeddings

CHUNK_TARGET = 700   # 청크 목표 길이(문자)
CHUNK_MIN = 200      # 이보다 짧으면 다음 블록과 합침


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def chunk_markdown(md: str) -> list[str]:
    """빈 줄 기준 블록 → 목표 길이로 병합. 표(| ... |)는 연속 유지."""
    blocks, cur = [], []
    for line in md.splitlines():
        if line.strip() == "":
            if cur:
                blocks.append("\n".join(cur))
                cur = []
        else:
            cur.append(line)
    if cur:
        blocks.append("\n".join(cur))

    chunks, buf = [], ""
    for b in blocks:
        if len(buf) + len(b) + 1 <= CHUNK_TARGET or len(buf) < CHUNK_MIN:
            buf = (buf + "\n" + b).strip()
        else:
            chunks.append(buf)
            buf = b
    if buf:
        chunks.append(buf)
    return [c for c in chunks if c.strip()]


def ingest_documents(conn, source: str):
    """파싱 마크다운을 청킹·임베딩하여 documents에 적재."""
    md_path = config.PARSED_DIR / f"{source}.md"
    md = md_path.read_text(encoding="utf-8")
    chunks = chunk_markdown(md)
    print(f"[documents] {source}: {len(chunks)}개 청크 임베딩...")

    vectors = embeddings.embed_passages(chunks)
    with conn.cursor() as cur:
        for content, emb in zip(chunks, vectors):
            cur.execute(
                "INSERT INTO documents (source, content, metadata, embedding) "
                "VALUES (%s, %s, %s, %s)",
                (source, content, json.dumps({"source": source}), emb),
            )
    print(f"[documents] {source}: {len(chunks)}건 적재 완료 (dim={len(vectors[0])})")


def ingest_all_documents(conn):
    """output/parsed/ 안의 모든 *.md 를 RAG 문서로 적재.
    팀원은 자기 문서를 parse_pdf.py로 파싱해 .md만 넣으면 자동 포함된다."""
    md_files = sorted(config.PARSED_DIR.glob("*.md"))
    if not md_files:
        print("[documents] output/parsed/*.md 없음 — 건너뜀")
        return
    for md_path in md_files:
        ingest_documents(conn, md_path.stem)


def ingest_courses(conn):
    catalog = load_json(config.STRUCTURED_DIR / "course_catalog.json")
    with conn.cursor() as cur:
        for c in catalog:
            cur.execute(
                """INSERT INTO courses
                   (교과목명, 이수구분, 학점, 이론, 실습, 개설학년, 개설학기, 트랙, 교육과정_연도)
                   VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
                (
                    c["교과목명"], c["이수구분"], c["학점"], c["이론"], c["실습"],
                    str(c["개설학년"]), str(c["개설학기"]), c["트랙"], c["교육과정_연도"],
                ),
            )
    print(f"[courses] {len(catalog)}과목 적재 완료")


def ingest_graduation(conn):
    grad = load_json(config.STRUCTURED_DIR / "graduation_requirements.json")
    conn.execute(
        "INSERT INTO graduation_requirements (교육과정_연도, data) VALUES (%s, %s)",
        (grad.get("교육과정_연도"), json.dumps(grad, ensure_ascii=False)),
    )
    print("[graduation_requirements] 1건 적재 완료")


def main():
    conn = db.connect()
    db.init_schema(conn)

    # 재적재를 위해 초기화 (idempotent)
    conn.execute("TRUNCATE documents, courses, graduation_requirements RESTART IDENTITY")

    ingest_courses(conn)
    ingest_graduation(conn)
    ingest_all_documents(conn)

    n_doc = conn.execute("SELECT count(*) FROM documents").fetchone()[0]
    n_course = conn.execute("SELECT count(*) FROM courses").fetchone()[0]
    print(f"\n✅ 적재 완료 — documents={n_doc}, courses={n_course}")
    conn.close()


if __name__ == "__main__":
    main()
