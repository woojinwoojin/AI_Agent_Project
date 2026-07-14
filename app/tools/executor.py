"""Tool 실행기. Router가 고른 도구를 실행하고 {success, data|error}를 반환."""

import asyncio
from typing import Any

from app.repositories.academic import AcademicRepository
from app.repositories.reminders import get_reminder_repository
from app.services.reminder_time import parse_remind_at

# 계열기초(None) 제외한 학점 계산 대상
_CATS = ["전공필수", "전공선택", "공통필수", "공통선택"]


class ToolExecutor:
    def __init__(self):
        self.academic = AcademicRepository()
        self.reminders = get_reminder_repository()

    async def execute(
        self, tool_name: str, tool_args: dict, session_id: str | None = None
    ) -> dict[str, Any]:
        args = tool_args or {}
        try:
            match tool_name:
                case "calc_graduation_progress":
                    return await asyncio.to_thread(self._calc_graduation, args)
                case "recommend_courses":
                    return await asyncio.to_thread(self._recommend_courses, args)
                case "send_reminder_email":
                    return await asyncio.to_thread(self._send_reminder_email, args, session_id)
                case _:
                    return {"success": False, "error": f"알 수 없는 도구: {tool_name}"}
        except Exception as e:  # noqa: BLE001
            return {"success": False, "error": str(e)}

    # --- calc_graduation_progress ---
    def _calc_graduation(self, args: dict) -> dict:
        req = self.academic.get_graduation_requirements()
        mins = req["이수구분별_최소학점"]
        전공_필요 = mins["전공필수"] + mins["전공선택"]

        이수: dict[str, int] = {}
        남은: dict[str, int] = {}

        # 전공(전공필수+전공선택 통합)으로 물은 경우
        if args.get("전공") is not None:
            done = int(args["전공"])
            이수["전공"] = done
            남은["전공"] = max(0, 전공_필요 - done)

        # 세부 이수구분
        for k in _CATS:
            if args.get(k) is not None:
                done = int(args[k])
                이수[k] = done
                남은[k] = max(0, mins[k] - done)

        if not 이수:
            return {
                "success": False,
                "error": "이수 학점 정보가 필요합니다. 예: '전공 30학점 들었어'",
            }

        return {
            "success": True,
            "data": {
                "기준": {"총_졸업학점": req["총_졸업학점"], "전공_필요": 전공_필요, **mins},
                "이수": 이수,
                "남은": 남은,
                "출처": f"{req['교육과정_연도']} 인공지능학과 졸업요건",
            },
        }

    # --- recommend_courses ---
    def _recommend_courses(self, args: dict) -> dict:
        학년, 학기, 트랙 = args.get("학년"), args.get("학기"), args.get("트랙")
        if not 학년 or not 학기:
            return {"success": False, "error": "학년과 학기 정보가 필요합니다."}
        courses = self.academic.recommend_courses(int(학년), int(학기), 트랙)
        if not courses:
            return {
                "success": False,
                "error": f"{학년}학년 {학기}학기 개설 과목을 찾지 못했습니다.",
            }
        return {
            "success": True,
            "data": {
                "학년": 학년,
                "학기": 학기,
                "트랙": 트랙 or "전체",
                "과목수": len(courses),
                "과목": courses,
                "출처": "2026 인공지능학과 교육과정",
            },
        }

    # --- send_reminder_email ---
    # Phase 2: 즉시 발송하지 않고 reminder_requests에 예약만 등록한다.
    # 실제 발송은 app/scheduler.py가 주기적으로 마감된 예약을 조회해 처리한다.
    def _send_reminder_email(self, args: dict, session_id: str | None = None) -> dict:
        이메일, 내용 = args.get("이메일"), args.get("내용")
        if not 이메일:
            return {"success": False, "error": "리마인드를 보낼 이메일 주소가 필요합니다."}

        내용 = 내용 or "학사 일정 리마인드"
        # 멀티턴 확인 흐름은 '의도 파악 시점'에 이미 파싱한 발송예정시각을 그대로
        # 넘긴다(확인 턴에서 재파싱하면 "내일" 등 상대 표현이 다른 날로 밀리는
        # 드리프트가 생김). 단일턴 경로 등 미전달 시에만 내용에서 재파싱한다.
        remind_at = args.get("발송예정시각") or parse_remind_at(내용)
        self.reminders.create(
            email=이메일, content=내용, remind_at=remind_at, session_id=session_id
        )

        # ADR-007: 이메일 주소는 개인정보이므로 응답 생성 LLM에 넘기는 data에는 담지 않는다.
        return {
            "success": True,
            "data": {
                "예약상태": "등록완료",
                "발송예정시각": remind_at.strftime("%Y-%m-%d %H:%M"),
            },
        }
