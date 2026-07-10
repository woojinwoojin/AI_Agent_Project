"""
데이터 적재: 정형 카탈로그 + 졸업요건 + RAG 청크 → PostgreSQL/pgvector

사용법:
    python -m app.ingest
"""
import json
import re

from app import config, db, embeddings

CHUNK_TARGET = 700   # 청크 목표 길이(문자)
CHUNK_MIN = 200      # 이보다 짧으면 다음 블록과 합침

MANIFEST = config.PARSED_DIR / "_crawl_manifest.json"
_CATEGORY_HEADER = re.compile(r"^>\s*카테고리:\s*([A-Za-z_]+)\s*/\s*([A-Za-z_]+)", re.M)


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_category_map() -> dict[str, tuple[str, str]]:
    """_crawl_manifest.json → {source: (category_l1, category_l2)}."""
    if not MANIFEST.exists():
        return {}
    out = {}
    for m in load_json(MANIFEST):
        if m.get("category_l1"):
            out[m["source"]] = (m["category_l1"], m.get("category_l2"))
    return out


def category_for(source: str, md: str, cat_map: dict) -> tuple[str | None, str | None]:
    """카테고리 결정: manifest 우선, 없으면 .md 헤더의 `> 카테고리: l1/l2` 파싱."""
    if source in cat_map:
        return cat_map[source]
    m = _CATEGORY_HEADER.search(md)
    if m:
        return m.group(1), m.group(2)
    return None, None


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


def ingest_documents(conn, source: str, cat_map: dict | None = None):
    """파싱 마크다운을 청킹·임베딩하여 documents에 적재. 카테고리(l1/l2)도 함께 저장."""
    md_path = config.PARSED_DIR / f"{source}.md"
    md = md_path.read_text(encoding="utf-8")
    cat_l1, cat_l2 = category_for(source, md, cat_map or {})
    chunks = chunk_markdown(md)
    tag = f"{cat_l1}/{cat_l2}" if cat_l1 else "미분류"
    print(f"[documents] {source} [{tag}]: {len(chunks)}개 청크 임베딩...")

    vectors = embeddings.embed_passages(chunks)
    meta = {"source": source, "category_l1": cat_l1, "category_l2": cat_l2}
    with conn.cursor() as cur:
        for content, emb in zip(chunks, vectors):
            cur.execute(
                "INSERT INTO documents "
                "(source, category_l1, category_l2, content, metadata, embedding) "
                "VALUES (%s, %s, %s, %s, %s, %s)",
                (source, cat_l1, cat_l2, content, json.dumps(meta), emb),
            )
    print(f"[documents] {source}: {len(chunks)}건 적재 완료 (dim={len(vectors[0])})")


def ingest_all_documents(conn):
    """output/parsed/ 안의 모든 *.md 를 RAG 문서로 적재.
    팀원은 자기 문서를 parse_pdf.py로 파싱해 .md만 넣으면 자동 포함된다."""
    md_files = sorted(config.PARSED_DIR.glob("*.md"))
    if not md_files:
        print("[documents] output/parsed/*.md 없음 — 건너뜀")
        return
    cat_map = load_category_map()
    uncategorized = []
    for md_path in md_files:
        source = md_path.stem
        ingest_documents(conn, source, cat_map)
        if source not in cat_map and not _CATEGORY_HEADER.search(
            md_path.read_text(encoding="utf-8")
        ):
            uncategorized.append(source)
    if uncategorized:
        print(
            "[documents] [주의] 카테고리 미분류(NULL로 적재됨, manifest/헤더에 추가 필요): "
            + ", ".join(uncategorized)
        )


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


def _when(c: dict) -> str:
    """과목 개설 시점 문자열. (부트캠프는 매학기/계절학기)"""
    yr, sem = c["개설학년"], c["개설학기"]
    if isinstance(sem, int):
        return f"{yr}학년 {sem}학기"
    return (f"{yr}학년 " if yr else "") + str(sem)


def synthesize_structured_docs(conn):
    """정형 카탈로그/졸업요건을 '깨끗한 자연어 문서'로 합성해 RAG에 적재.
    2단 인터리빙 표에서 오는 부정확성을 제거하고 과목/학점 질의 정확도를 높인다."""
    from collections import defaultdict

    catalog = load_json(config.STRUCTURED_DIR / "course_catalog.json")
    grad = load_json(config.STRUCTURED_DIR / "graduation_requirements.json")
    SRC = "2026 인공지능학과 교육과정(정형)"
    docs: list[str] = []

    # A) 과목별 사실 문서
    for c in catalog:
        docs.append(
            f"[교육과정] '{c['교과목명']}'은(는) 가천대 인공지능학과 {c['트랙']} 트랙 "
            f"{_when(c)} 개설 {c['이수구분']} 과목이며 {c['학점']}학점"
            f"(이론 {c['이론']}, 실습 {c['실습']})이다."
        )

    # B) 공통 트랙 (학년,학기)별 개설 과목 (이수구분 묶음)
    grp = defaultdict(lambda: defaultdict(list))
    for c in catalog:
        if c["트랙"] == "공통" and isinstance(c["개설학기"], int):
            grp[(c["개설학년"], c["개설학기"])][c["이수구분"]].append(c["교과목명"])
    for (yr, sem), gubuns in sorted(grp.items()):
        parts = "; ".join(f"{g}: {', '.join(ns)}" for g, ns in gubuns.items())
        docs.append(f"[교육과정] 인공지능학과 {yr}학년 {sem}학기 개설 과목 — {parts}.")

    # C) 트랙별 과목 목록
    for trk in ["Intelligent SW", "AIoT", "Vision & Language", "AI부트캠프"]:
        items = [f"{c['교과목명']}({_when(c)})" for c in catalog if c["트랙"] == trk]
        if items:
            docs.append(f"[교육과정] 인공지능학과 {trk} 트랙 과목: " + ", ".join(items) + ".")

    # D) 이수구분별 전체 목록
    for g in ["전공필수", "전공선택", "공통필수"]:
        names = [c["교과목명"] for c in catalog if c["이수구분"] == g]
        docs.append(
            f"[교육과정] 인공지능학과 {g} 과목 전체({len(names)}과목): " + ", ".join(names) + "."
        )

    # E) 졸업요건
    req = grad["이수구분별_최소학점"]
    docs.append(
        f"[졸업요건] {grad['교육과정_연도']} 가천대 인공지능학과 졸업 이수학점은 총 "
        f"{grad['총_졸업학점']}학점이다. 전공필수 {req['전공필수']}학점, 전공선택 {req['전공선택']}학점, "
        f"공통필수 {req['공통필수']}학점, 공통선택 {req['공통선택']}학점을 이수해야 한다. "
        f"전공(전공필수+전공선택)은 {req['전공필수'] + req['전공선택']}학점이다."
    )

    print(f"[synth] 정형 문서 {len(docs)}건 임베딩...")
    vectors = embeddings.embed_passages(docs)
    with conn.cursor() as cur:
        for content, emb in zip(docs, vectors):
            cur.execute(
                "INSERT INTO documents (source, content, metadata, embedding) "
                "VALUES (%s, %s, %s, %s)",
                (SRC, content, json.dumps({"source": SRC, "kind": "structured"}), emb),
            )
    print(f"[synth] {len(docs)}건 적재 완료")


def main():
    conn = db.connect()
    db.init_schema(conn)

    # 재적재를 위해 초기화 (idempotent)
    conn.execute("TRUNCATE documents, courses, graduation_requirements RESTART IDENTITY")

    ingest_courses(conn)
    ingest_graduation(conn)
    synthesize_structured_docs(conn)
    ingest_all_documents(conn)

    n_doc = conn.execute("SELECT count(*) FROM documents").fetchone()[0]
    n_course = conn.execute("SELECT count(*) FROM courses").fetchone()[0]
    print(f"\n✅ 적재 완료 — documents={n_doc}, courses={n_course}")
    conn.close()


if __name__ == "__main__":
    main()
