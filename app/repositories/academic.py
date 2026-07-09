"""학사 정형 데이터 접근 (courses / graduation_requirements)."""
import json

from app import config, db


class AcademicRepository:
    def recommend_courses(self, 학년: int, 학기: int, 트랙: str | None = None) -> list[dict]:
        """해당 학년/학기 개설 과목. 트랙 지정 시 공통+해당트랙만."""
        conn = db.connect()
        try:
            sql = (
                "SELECT 교과목명, 이수구분, 학점, 트랙 FROM courses "
                "WHERE 개설학년 = %s AND 개설학기 = %s"
            )
            params: list = [str(학년), str(학기)]
            if 트랙:
                sql += " AND 트랙 IN ('공통', %s)"
                params.append(트랙)
            sql += " ORDER BY 이수구분, 교과목명"
            rows = conn.execute(sql, params).fetchall()
            return [
                {"교과목명": r[0], "이수구분": r[1], "학점": r[2], "트랙": r[3]} for r in rows
            ]
        finally:
            conn.close()

    def get_graduation_requirements(self) -> dict:
        """졸업 이수학점 기준 (정형 JSON = 신뢰 원본)."""
        path = config.STRUCTURED_DIR / "graduation_requirements.json"
        return json.loads(path.read_text(encoding="utf-8"))
