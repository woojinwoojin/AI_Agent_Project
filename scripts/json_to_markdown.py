import json
from pathlib import Path


INPUT_PATH = Path("data/docs.json")
OUTPUT_PATH = Path("output/parsed/gachon_academic_docs.md")


def main():
    docs = json.loads(INPUT_PATH.read_text(encoding="utf-8"))

    blocks = []

    for doc in docs:
        title = doc.get("title", "제목 없음")
        area = doc.get("area", "")
        category = doc.get("category", "")
        content = doc.get("content", "")
        source = doc.get("source", "")
        source_page = doc.get("source_page", "")
        keywords = ", ".join(doc.get("keywords", []))

        block = f"""# {title}

영역: {area}
분류: {category}
키워드: {keywords}

{content}

출처: {source} {source_page}
"""
        blocks.append(block.strip())

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text("\n\n---\n\n".join(blocks), encoding="utf-8")

    print(f"변환 완료: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()