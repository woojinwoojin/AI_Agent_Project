"""이메일 리마인드 예약 데이터 접근 (reminder_requests)."""

from datetime import datetime

from app import db


class ReminderRepository:
    def create(
        self, email: str, content: str, remind_at: datetime, session_id: str | None = None
    ) -> int:
        conn = db.connect()
        try:
            row = conn.execute(
                """
                INSERT INTO reminder_requests (session_id, email, content, remind_at, created_at)
                VALUES (%s, %s, %s, %s, %s)
                RETURNING id
                """,
                (session_id, email, content, remind_at, datetime.now()),
            ).fetchone()
            return row[0]
        finally:
            conn.close()

    def fetch_due(self, now: datetime, limit: int = 50) -> list[dict]:
        """status='pending'이고 remind_at이 지난 항목을 발송 대상으로 조회."""
        conn = db.connect()
        try:
            rows = conn.execute(
                """
                SELECT id, email, content
                FROM reminder_requests
                WHERE status = 'pending' AND remind_at <= %s
                ORDER BY remind_at
                LIMIT %s
                """,
                (now, limit),
            ).fetchall()
            return [{"id": r[0], "email": r[1], "content": r[2]} for r in rows]
        finally:
            conn.close()

    def mark_sent(self, reminder_id: int) -> None:
        conn = db.connect()
        try:
            conn.execute(
                "UPDATE reminder_requests SET status = 'sent', sent_at = %s WHERE id = %s",
                (datetime.now(), reminder_id),
            )
        finally:
            conn.close()

    def mark_failed(self, reminder_id: int, error: str) -> None:
        conn = db.connect()
        try:
            conn.execute(
                "UPDATE reminder_requests SET status = 'failed', error = %s WHERE id = %s",
                (error, reminder_id),
            )
        finally:
            conn.close()

    def list_recent(self, limit: int = 100) -> list[dict]:
        """관리용 리마인드 로그 조회 (최근 생성순)."""
        conn = db.connect()
        try:
            rows = conn.execute(
                """
                SELECT id, email, content, remind_at, status, error, created_at, sent_at
                FROM reminder_requests
                ORDER BY created_at DESC
                LIMIT %s
                """,
                (limit,),
            ).fetchall()
            return [
                {
                    "id": r[0],
                    "email": r[1],
                    "content": r[2],
                    "remind_at": r[3],
                    "status": r[4],
                    "error": r[5],
                    "created_at": r[6],
                    "sent_at": r[7],
                }
                for r in rows
            ]
        finally:
            conn.close()


_repo: ReminderRepository | None = None


def get_reminder_repository() -> ReminderRepository:
    global _repo
    if _repo is None:
        _repo = ReminderRepository()
    return _repo
