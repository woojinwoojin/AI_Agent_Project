"""
Stage 2b: 학번(입학년도)별 졸업 이수학점 기준 → 정형 JSON (다년도)

가천대 요람 '교육과정표' 하단의 '졸업 이수학점' 박스를 학번(입학년도)별로 정형화한다.
원문 PDF(2021~2025 요람 발췌 + 2026 build_catalog)를 사람이 직접 대조 확인한 상수를
담고, 불변식(전공필수+전공선택=72, 총=120)으로 검증한 뒤 JSON으로 내보낸다.

한국 대학은 입학년도 기준으로 졸업요건이 적용된다 → 학번별로 다른 값을 봇이
학번-aware하게 인용해야 오답이 없다. (앞서 Phase 1에서 학번-aware 검색/되묻기 완료)

관찰된 사실:
  - 졸업요건은 두 체제: 2021~2022(전필38/전선34, 기초교양17/융합교양7),
    2023~2026(전필35/전선37, 교양11+13). 전공(필수+선택)은 전 학번 72로 동일.
  - 학과명 전환: 2021~2024 'AI·소프트웨어학부 소프트웨어전공' → 2025~ '인공지능학과'.
    (인공지능학과 = 구 소프트웨어전공. 재학생 검색이 '인공지능학과'로 매칭되도록 학과명 병기)
  - 교양 명칭 변화: 2021~2025 '기초교양/융합교양' → 2026 '공통선택/공통필수'
    (수치 매핑: 기초교양13=공통선택13, 융합교양11=공통필수11).

출력:
  output/structured/graduation_by_year.json  (연도 오름차순 리스트)
"""

import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "output" / "structured" / "graduation_by_year.json"

_SW = "AI·소프트웨어학부 소프트웨어전공"
_AI = "인공지능학과"


def rec(year, 학부전공, 전공필수, 전공선택, 교양, 비고, 총=120):
    """학번별 졸업요건 레코드. 학과는 재학생 검색 매칭을 위해 항상 '인공지능학과'로
    표기하고, 원문 학부/전공명은 학부전공_원문에 보존한다."""
    return {
        "교육과정_연도": year,
        "학과": _AI,
        "학부전공_원문": 학부전공,
        "총_졸업학점": 총,
        "전공필수": 전공필수,
        "전공선택": 전공선택,
        "전공_합": 전공필수 + 전공선택,
        "교양": 교양,
        "비고": 비고,
    }


# 원문(요람 교육과정표 '졸업 이수학점' 박스) 대조 확인값.
GRADUATION_BY_YEAR = [
    rec(
        2021,
        _SW,
        38,
        34,
        {"기초교양": 17, "융합교양": 7, "계열교양": None},
        "융합교양은 4개 영역 중 서로 다른 3개 영역을 이수해야 함. (구 소프트웨어전공)",
    ),
    rec(
        2022,
        _SW,
        38,
        34,
        {"기초교양": 17, "융합교양": 7, "계열교양": None},
        "융합교양은 4개 영역 중 서로 다른 3개 영역을 이수해야 함. (구 소프트웨어전공)",
    ),
    rec(
        2023,
        _SW,
        35,
        37,
        {"기초교양": 13, "융합교양": 11, "계열교양": None},
        "구 소프트웨어전공. 전공선택 37학점은 Big Data/Smart Systems 트랙 과목에서 이수.",
    ),
    rec(
        2024,
        _SW,
        35,
        37,
        {"기초교양": 13, "융합교양": 11, "계열교양": None},
        "구 소프트웨어전공. 전공선택 37학점은 Big Data/Smart Systems 트랙 과목에서 이수.",
    ),
    rec(
        2025,
        _AI,
        35,
        37,
        {"기초교양": 13, "융합교양": 11, "계열교양": None},
        "전공선택 37학점은 Intelligent SW/AIoT/Vision & Language 트랙 과목에서 이수.",
    ),
    rec(
        2026,
        _AI,
        35,
        37,
        {"공통필수": 11, "공통선택": 13, "계열기초": None},
        "트랙 무관 공통 기준. 전공선택 37학점은 3개 트랙/부트캠프 과목에서 이수. "
        "(교양 명칭 개편: 2025 기초교양13→공통선택13, 융합교양11→공통필수11)",
    ),
]


def validate():
    errs = []
    seen = set()
    for r in GRADUATION_BY_YEAR:
        y = r["교육과정_연도"]
        if y in seen:
            errs.append(f"{y}: 연도 중복")
        seen.add(y)
        # 불변식: 전공(필수+선택) = 72, 총 = 120
        if r["전공_합"] != 72:
            errs.append(f"{y}: 전공필수+전공선택={r['전공_합']} ≠ 72")
        if r["총_졸업학점"] != 120:
            errs.append(f"{y}: 총_졸업학점={r['총_졸업학점']} ≠ 120")
        for k in ("전공필수", "전공선택"):
            if not isinstance(r[k], int) or r[k] <= 0:
                errs.append(f"{y}: {k} 값 이상 ({r[k]})")
    return errs


def main():
    errs = validate()
    print("=== 검증 ===")
    if errs:
        print("❌ 오류:")
        for e in errs:
            print("  -", e)
        raise SystemExit(1)
    print(f"✅ {len(GRADUATION_BY_YEAR)}개 연도 불변식 통과 (전공합=72, 총=120)")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(GRADUATION_BY_YEAR, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[저장] {OUT.relative_to(ROOT)}")
    for r in GRADUATION_BY_YEAR:
        print(
            f"  {r['교육과정_연도']}: 전필 {r['전공필수']} / 전선 {r['전공선택']} / "
            f"교양 {r['교양']} / 총 {r['총_졸업학점']}"
        )


if __name__ == "__main__":
    main()
