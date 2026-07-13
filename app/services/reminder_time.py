"""자연어 문장에서 리마인드 발송 시각을 규칙 기반으로 추출.

RAG 문서에서 실제 학사 일정을 자동으로 찾아 연결하는 기능은 아직 없다(별도
과제). 지금은 사용자가 메시지에 직접 적은 날짜/시간 표현만 해석하며, 날짜
표현이 전혀 없으면 "지금 바로"로 보고 즉시(= 다음 스케줄러 tick) 발송한다.
"""

import re
from datetime import datetime, timedelta

_MONTH_DAY = re.compile(r"(\d{1,2})\s*월\s*(\d{1,2})\s*일")
_N_DAYS_LATER = re.compile(r"(\d+)\s*일\s*(?:후|뒤)")
_TIME = re.compile(r"(오전|오후)?\s*(\d{1,2})\s*시(?:\s*(\d{1,2})\s*분)?")
_RELATIVE_DAYS = {"오늘": 0, "내일": 1, "모레": 2, "글피": 3}


def parse_remind_at(text: str, now: datetime | None = None) -> datetime:
    base = now or datetime.now()
    target_date = None

    m = _MONTH_DAY.search(text)
    if m:
        month, day = int(m.group(1)), int(m.group(2))
        try:
            target_date = base.replace(
                month=month, day=day, hour=0, minute=0, second=0, microsecond=0
            )
        except ValueError:
            target_date = None
        if target_date and target_date.date() < base.date():
            # 이미 지난 날짜면 내년으로 (예: 12월에 "1월 3일" 요청)
            target_date = target_date.replace(year=target_date.year + 1)

    if target_date is None:
        nd = _N_DAYS_LATER.search(text)
        if nd:
            target_date = (base + timedelta(days=int(nd.group(1)))).replace(
                hour=0, minute=0, second=0, microsecond=0
            )

    if target_date is None:
        for word, delta in _RELATIVE_DAYS.items():
            if word in text:
                target_date = (base + timedelta(days=delta)).replace(
                    hour=0, minute=0, second=0, microsecond=0
                )
                break

    if target_date is None:
        return base  # 날짜 표현 없음 -> 즉시 발송(Phase 1과 동일 동작)

    tm = _TIME.search(text)
    if tm:
        ampm, hour, minute = tm.group(1), int(tm.group(2)), int(tm.group(3) or 0)
        if ampm == "오후" and hour < 12:
            hour += 12
        if ampm == "오전" and hour == 12:
            hour = 0
        target_date = target_date.replace(hour=hour, minute=minute)
    else:
        target_date = target_date.replace(hour=9, minute=0)  # 시간 미지정 시 오전 9시

    return target_date
