"""평가 대시보드 생성기 — Tier1/Tier2 원자료(JSON) → 자체완결 HTML.

사용법:  python -m eval.build_dashboard
출력:    eval/results/dashboard.html  (외부 의존 없음, 브라우저로 바로 열림)

멘토 사전질문1 KPI를 발표용으로 시각화: 라우팅/검색/가드레일 정확도, 근거확보,
답변 품질(grounded/출처/관련도), 지연, 그리고 학번-게이트 수정 before/after.
"""

import json
import os
from datetime import datetime, timedelta, timezone

RESULT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

# 학번 게이트 수정 '전' Tier1 baseline (개선 스토리용, 세션 측정값).
BEFORE_T1 = {
    "intent": 82.1,
    "category": 92.3,
    "guardrail": 93.8,
    "retrieval": 100.0,
    "facts": 72.7,
}

# 팔레트: 명도 차 큰 2색(중립 slate / 강조 blue) + 상태색. 값 라벨을 항상 병기해
# 색만으로 정보를 전달하지 않는다(CVD 안전).
C_AFTER = "#2563eb"
C_BEFORE = "#94a3b8"
C_GOOD = "#16a34a"
C_WARN = "#d97706"
C_CRIT = "#dc2626"


def _load(name):
    path = os.path.join(RESULT_DIR, name)
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _rate(recs, key, subset=None):
    xs = [r[key] for r in (subset or recs) if r.get(key) is not None]
    return (sum(1 for x in xs if x) / len(xs) * 100, len(xs)) if xs else (None, 0)


def _stats(vals):
    if not vals:
        return None
    s = sorted(vals)

    def pct(q):
        return s[max(0, min(len(s) - 1, int(round((len(s) - 1) * q))))]

    return {"avg": sum(s) / len(s), "p50": pct(0.5), "p95": pct(0.95)}


def agg_tier1(recs):
    runs = len({r["run"] for r in recs})
    scen = len({r["scenario"] for r in recs})
    a = {}
    for label, key in [
        ("intent", "intent_ok"),
        ("category", "cat_ok"),
        ("guardrail", "guardrail_ok"),
        ("retrieval", "retrieval_ok"),
        ("facts", "facts_hit"),
        ("tool", "tool_ok"),
    ]:
        a[label], _ = _rate(recs, key)
    router = _stats([r["router_ms"] for r in recs if r.get("router_ms") is not None])
    stage = _stats([r["stage_ms"] for r in recs if r.get("stage_ms") is not None])
    return {
        "runs": runs,
        "scen": scen,
        "turns": len(recs),
        "acc": a,
        "router": router,
        "stage": stage,
    }


def agg_tier2(recs):
    judged = [r for r in recs if r.get("judged")]
    runs = len({r["run"] for r in recs})
    grounded, ng = _rate(judged, "grounded")
    src, ns = _rate(judged, "source_cited", [r for r in judged if r.get("expect_source")])
    rels = [r["relevance"] for r in judged if r.get("relevance") is not None]
    guard = [r for r in judged if not r["answerable"]]
    gguard, _ = _rate(judged, "grounded", guard)
    e2e = _stats([r["e2e_ms"] for r in recs if r.get("e2e_ms") is not None])
    bad = [r for r in judged if r.get("grounded") is False or (r.get("relevance") or 5) <= 2]
    return {
        "runs": runs,
        "judged": len(judged),
        "grounded": grounded,
        "ng": ng,
        "source": src,
        "ns": ns,
        "relevance": (sum(rels) / len(rels)) if rels else None,
        "guard_honesty": gguard,
        "n_guard": len(guard),
        "e2e": e2e,
        "bad": bad,
    }


def _tile(value, label, sub="", status="good"):
    color = {"good": C_GOOD, "warn": C_WARN, "crit": C_CRIT}[status]
    return f"""<div class="tile">
      <div class="tile-val" style="color:{color}">{value}</div>
      <div class="tile-label">{label}</div>
      <div class="tile-sub">{sub}</div>
    </div>"""


def _bar(label, pct, color, note=""):
    w = 0 if pct is None else max(2, pct)
    val = "n/a" if pct is None else f"{pct:.1f}%"
    return f"""<div class="bar-row">
      <div class="bar-label">{label}</div>
      <div class="bar-track"><div class="bar-fill" style="width:{w}%;background:{color}"></div></div>
      <div class="bar-val">{val}{note}</div>
    </div>"""


def _ba_bar(label, before, after):
    """before/after 페어 바."""

    def seg(pct, color, tag):
        w = 0 if pct is None else max(2, pct)
        v = "n/a" if pct is None else f"{pct:.1f}%"
        return (
            f'<div class="ba-seg"><span class="ba-tag">{tag}</span>'
            f'<div class="bar-track"><div class="bar-fill" style="width:{w}%;background:{color}"></div></div>'
            f'<span class="bar-val">{v}</span></div>'
        )

    delta = ""
    if before is not None and after is not None:
        d = after - before
        if abs(d) >= 0.05:
            delta = (
                f'<span class="delta">▲ +{d:.1f}p</span>'
                if d > 0
                else f'<span class="delta down">▼ {d:.1f}p</span>'
            )
    return f"""<div class="ba-row">
      <div class="bar-label">{label} {delta}</div>
      {seg(before, C_BEFORE, "전")}
      {seg(after, C_AFTER, "후")}
    </div>"""


def _lat_bars(title, stats, unit="ms"):
    if not stats:
        return ""
    mx = max(stats["avg"], stats["p50"], stats["p95"]) or 1
    rows = ""
    for k, c in [("p50", C_GOOD), ("avg", C_AFTER), ("p95", C_WARN)]:
        w = max(2, stats[k] / mx * 100)
        rows += (
            f'<div class="bar-row"><div class="bar-label lat">{k}</div>'
            f'<div class="bar-track"><div class="bar-fill" style="width:{w}%;background:{c}"></div></div>'
            f'<div class="bar-val">{stats[k]:.0f}{unit}</div></div>'
        )
    return f'<div class="lat-block"><div class="lat-title">{title}</div>{rows}</div>'


def build_html(t1, t2):
    kst = datetime.now(timezone(timedelta(hours=9))).strftime("%Y-%m-%d %H:%M KST")
    a = t1["acc"]

    def st(pct, hi=95, mid=85):
        if pct is None:
            return "warn"
        return "good" if pct >= hi else ("warn" if pct >= mid else "crit")

    tiles = "".join(
        [
            _tile(f"{a['intent']:.1f}%", "intent 라우팅", f"n={t1['turns']}", st(a["intent"])),
            _tile(f"{a['category']:.0f}%", "category 분류", "규칙 기반", st(a["category"])),
            _tile(f"{a['guardrail']:.1f}%", "가드레일 정확도", "범위밖 판별", st(a["guardrail"])),
            _tile(f"{a['retrieval']:.0f}%", "검색 성공률", "answerable RAG", st(a["retrieval"])),
            _tile(f"{a['facts']:.0f}%", "근거 확보율", "expect_facts", st(a["facts"])),
        ]
    )
    q_tiles = ""
    if t2:
        q_tiles = "".join(
            [
                _tile(
                    f"{t2['grounded']:.1f}%",
                    "grounded(환각 반대)",
                    f"환각율 {100 - t2['grounded']:.1f}%",
                    st(t2["grounded"], 90, 80),
                ),
                _tile(
                    f"{t2['source']:.1f}%", "출처 인용률", f"n={t2['ns']}", st(t2["source"], 90, 80)
                ),
                _tile(
                    f"{t2['relevance']:.2f}/5",
                    "평균 관련도",
                    f"{t2['relevance'] / 5 * 100:.0f}%",
                    st(t2["relevance"] / 5 * 100, 90, 75),
                ),
                _tile(
                    f"{t2['guard_honesty']:.0f}%",
                    "가드레일 정직도",
                    f"n={t2['n_guard']}",
                    st(t2["guard_honesty"], 90, 75),
                ),
            ]
        )

    ba = "".join(
        [
            _ba_bar("intent 라우팅", BEFORE_T1["intent"], a["intent"]),
            _ba_bar("category 분류", BEFORE_T1["category"], a["category"]),
            _ba_bar("가드레일 정확도", BEFORE_T1["guardrail"], a["guardrail"]),
            _ba_bar("근거 확보율", BEFORE_T1["facts"], a["facts"]),
        ]
    )

    lat = _lat_bars("Router (규칙 확정 시 LLM 생략 → p50 0ms)", t1["router"])
    lat += _lat_bars("검색/도구 (RAG retrieval+rerank)", t1["stage"])
    if t2 and t2["e2e"]:
        lat += _lat_bars("전체 응답 e2e (응답 LLM 포함)", t2["e2e"])

    bad_rows = ""
    if t2 and t2["bad"]:
        for r in t2["bad"]:
            bad_rows += (
                f"<tr><td>{r['scenario']}</td><td>{r['q']}</td>"
                f"<td>{'환각' if r.get('grounded') is False else ''} "
                f"rel={r.get('relevance')}</td><td>{r.get('judge_reason', '')}</td></tr>"
            )
        bad_rows = f"""<table class="tbl"><thead><tr><th>시나리오</th><th>질문</th>
          <th>플래그</th><th>심판 사유</th></tr></thead><tbody>{bad_rows}</tbody></table>
          <p class="muted">※ LLM 심판은 보수적으로 채점 — 위 케이스 상당수는 실제로는 정상
          (긴 목록 누락 오판, 정직한 거절을 저관련으로 평가 등).</p>"""

    t2_meta = f"Tier2 {t2['runs']}회·판정 {t2['judged']}턴" if t2 else "Tier2 미실행"

    return f"""<!doctype html><html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>AI 학과 길잡이 — 평가 대시보드</title>
<style>
:root {{
  --bg:#f8fafc; --surface:#ffffff; --ink:#0f172a; --ink2:#475569; --muted:#94a3b8;
  --border:#e2e8f0; --track:#eef2f6;
}}
@media (prefers-color-scheme: dark) {{
  :root {{ --bg:#0b1120; --surface:#111827; --ink:#e5e7eb; --ink2:#9ca3af;
    --muted:#6b7280; --border:#1f2937; --track:#1e293b; }}
}}
:root[data-theme="dark"] {{ --bg:#0b1120; --surface:#111827; --ink:#e5e7eb; --ink2:#9ca3af;
  --muted:#6b7280; --border:#1f2937; --track:#1e293b; }}
:root[data-theme="light"] {{ --bg:#f8fafc; --surface:#ffffff; --ink:#0f172a; --ink2:#475569;
  --muted:#94a3b8; --border:#e2e8f0; --track:#eef2f6; }}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:var(--bg); color:var(--ink);
  font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,"Malgun Gothic",sans-serif;
  line-height:1.5; }}
.wrap {{ max-width:1040px; margin:0 auto; padding:32px 20px 64px; }}
h1 {{ font-size:24px; margin:0 0 4px; }}
h2 {{ font-size:16px; margin:36px 0 14px; color:var(--ink); border-left:3px solid {C_AFTER};
  padding-left:10px; }}
.sub {{ color:var(--ink2); font-size:13px; margin:0; }}
.grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:12px; }}
.tile {{ background:var(--surface); border:1px solid var(--border); border-radius:12px;
  padding:16px; }}
.tile-val {{ font-size:26px; font-weight:700; letter-spacing:-.5px; }}
.tile-label {{ font-size:13px; font-weight:600; margin-top:2px; }}
.tile-sub {{ font-size:11px; color:var(--muted); margin-top:2px; }}
.card {{ background:var(--surface); border:1px solid var(--border); border-radius:12px;
  padding:18px 20px; }}
.bar-row, .ba-seg {{ display:flex; align-items:center; gap:10px; margin:7px 0; }}
.bar-label {{ width:150px; font-size:13px; flex-shrink:0; }}
.bar-label.lat {{ width:44px; color:var(--ink2); font-variant:tabular-nums; }}
.bar-track {{ flex:1; height:14px; background:var(--track); border-radius:7px; overflow:hidden; }}
.bar-fill {{ height:100%; border-radius:7px; }}
.bar-val {{ width:78px; text-align:right; font-size:13px; font-variant-numeric:tabular-nums;
  color:var(--ink2); flex-shrink:0; }}
.ba-row {{ padding:10px 0; border-bottom:1px solid var(--border); }}
.ba-row:last-child {{ border-bottom:0; }}
.ba-tag {{ width:20px; font-size:11px; color:var(--muted); flex-shrink:0; }}
.ba-seg {{ margin:4px 0 4px 0; }}
.delta {{ color:{C_GOOD}; font-size:12px; font-weight:600; }}
.delta.down {{ color:{C_CRIT}; }}
.lat-block {{ margin-bottom:16px; }}
.lat-title {{ font-size:13px; font-weight:600; margin-bottom:4px; }}
.tbl {{ width:100%; border-collapse:collapse; font-size:12px; margin-top:8px; }}
.tbl th, .tbl td {{ text-align:left; padding:6px 8px; border-bottom:1px solid var(--border);
  vertical-align:top; }}
.tbl th {{ color:var(--ink2); font-weight:600; }}
.muted {{ color:var(--muted); font-size:12px; }}
.method {{ font-size:13px; color:var(--ink2); }}
.method b {{ color:var(--ink); }}
.legend {{ font-size:12px; color:var(--muted); margin-top:6px; }}
.dot {{ display:inline-block; width:9px; height:9px; border-radius:2px; margin:0 3px 0 10px;
  vertical-align:middle; }}
</style></head><body><div class="wrap">
<h1>가천대 인공지능학과 길잡이 — 평가 대시보드</h1>
<p class="sub">Tier1 {t1['scen']}개 시나리오 × {t1['runs']}회 = {t1['turns']}턴 · {t2_meta} · 생성 {kst}</p>

<h2>핵심 KPI — 라우팅 &amp; 검색 (Tier1, 결정적)</h2>
<div class="grid">{tiles}</div>

<h2>답변 품질 (Tier2, LLM 심판)</h2>
<div class="grid">{q_tiles or '<p class="muted">Tier2 데이터 없음</p>'}</div>

<h2>개선 before / after — 학번 되묻기 게이트 수정</h2>
<div class="card">{ba}
<div class="legend"><span class="dot" style="background:{C_BEFORE}"></span>수정 전
<span class="dot" style="background:{C_AFTER}"></span>수정 후</div></div>

<h2>지연 (latency)</h2>
<div class="card">{lat}
<div class="legend"><span class="dot" style="background:{C_GOOD}"></span>p50
<span class="dot" style="background:{C_AFTER}"></span>avg
<span class="dot" style="background:{C_WARN}"></span>p95</div></div>

<h2>주목 케이스 (Tier2 심판 플래그)</h2>
<div class="card">{bad_rows or '<p class="muted">환각/저관련 플래그 없음</p>'}</div>

<h2>방법론 &amp; 할루시네이션 기준</h2>
<div class="card method">
<p><b>KPI 산정:</b> 시나리오 N회 반복 평균. Tier1은 응답 LLM 없이 라우팅/검색/가드레일을
결정적으로 측정, Tier2는 전체 그래프 답변을 LLM 심판으로 채점.</p>
<p><b>할루시네이션 기준(MVP):</b> ① 1차 신뢰 게이트 — reranker 점수(pgvector confidence
가중합) &lt; 0.40이면 답변 대신 문의처 안내(가드레일). ② 2차 grounding — 답변 사실이
검색 근거/도구 결과에 있는지 LLM 심판이 판정.</p>
<p><b>한계:</b> LLM 심판은 보수적으로 채점(긴 목록 항목 누락 오판, 정직한 거절을 저관련
평가 등) — 실제 환각률은 표기치보다 낮음. 데이터는 2026학년도 총람 및 확보 학사자료 기준.</p>
</div>
</div></body></html>"""


def main():
    t1_raw = _load("tier1_raw.json")
    t2_raw = _load("tier2_raw.json")
    if not t1_raw:
        print("tier1_raw.json 없음 — 먼저 eval.run_tier1 실행")
        return
    t1 = agg_tier1(t1_raw)
    t2 = agg_tier2(t2_raw) if t2_raw else None
    html = build_html(t1, t2)
    out = os.path.join(RESULT_DIR, "dashboard.html")
    with open(out, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"대시보드 생성: {out}")
    print(f"  Tier1: {t1['turns']}턴 / {t1['runs']}회, intent {t1['acc']['intent']:.1f}%")
    if t2:
        print(f"  Tier2: {t2['judged']}턴 / {t2['runs']}회, grounded {t2['grounded']:.1f}%")


if __name__ == "__main__":
    main()
