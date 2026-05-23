from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List


ROOT = Path(__file__).resolve().parent
PLUS_PATH = ROOT / "plus.py"
OUTPUT_DIR = ROOT / "paper_repro_outputs" / "cache_similarity_eval_v3_real_70"
OUTPUT_JSON = OUTPUT_DIR / "review_ui_phase1_static_check.json"
OUTPUT_MD = OUTPUT_DIR / "review_ui_phase1_static_check.md"


REQUIRED_TOKENS = [
    "cache_review",
    "showCacheReview",
    "review-card",
    "review_phase",
    "hint_only",
]

FORBIDDEN_TOKENS = [
    "cache_review_decision",
    "confirm_cache_reuse",
    "showModel(data.candidate_model_url)",
    "loadModel(data.candidate_model_url)",
    'fetch("',
    "fetch('",
]


def add_check(checks: List[Dict[str, Any]], name: str, passed: bool, detail: str = "") -> None:
    checks.append({"name": name, "passed": bool(passed), "detail": detail})


def build_markdown(result: Dict[str, Any]) -> str:
    def mark(value: bool) -> str:
        return "通过" if value else "失败"

    lines = [
        "# Review UI 第一阶段静态检查报告",
        "",
        "## 1. 检查目的",
        "",
        "确认 `review` 第一阶段只新增提示展示能力，没有新增真实复用接口、候选模型加载或阻塞确认逻辑。",
        "",
        "## 2. 检查范围",
        "",
        "- 只读取 `plus.py` 文本；",
        "- 不启动 Flask；",
        "- 不调用 Qwen；",
        "- 不调用 TripoSR；",
        "- 不训练 MLP；",
        "- 不重跑实验。",
        "",
        "## 3. 检查结果",
        "",
        f"- static_check_passed：`{result['static_check_passed']}`",
        "",
        "| 检查项 | 结果 | 说明 |",
        "| --- | --- | --- |",
    ]
    for item in result["checks"]:
        lines.append(f"| {item['name']} | {mark(item['passed'])} | {item.get('detail', '')} |")

    lines.extend(["", "## 4. 结论", ""])
    if result["static_check_passed"]:
        lines.append("Review UI 第一阶段静态检查通过：前端提示事件与提示卡片存在，且未发现真实复用接口、候选 GLB 加载或 `fetch` 调用。")
    else:
        lines.append("Review UI 第一阶段静态检查未通过，需要先处理以下失败项：")
        for failed in result["failed_checks"]:
            lines.append(f"- {failed}")
    return "\n".join(lines) + "\n"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    checks: List[Dict[str, Any]] = []

    plus_exists = PLUS_PATH.exists()
    add_check(checks, "plus.py exists", plus_exists, str(PLUS_PATH))
    text = PLUS_PATH.read_text(encoding="utf-8-sig", errors="replace") if plus_exists else ""

    for token in REQUIRED_TOKENS:
        add_check(checks, f"contains required token: {token}", token in text)

    for token in FORBIDDEN_TOKENS:
        add_check(checks, f"does not contain forbidden token: {token}", token not in text)

    static_check_passed = all(item["passed"] for item in checks)
    failed_checks = [item["name"] for item in checks if not item["passed"]]
    result: Dict[str, Any] = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "static_check_passed": static_check_passed,
        "failed_checks": failed_checks,
        "required_tokens": REQUIRED_TOKENS,
        "forbidden_tokens": FORBIDDEN_TOKENS,
        "checks": checks,
        "constraints": {
            "start_flask": False,
            "call_qwen": False,
            "call_triposr": False,
            "train_mlp": False,
            "rerun_experiment": False,
        },
    }

    OUTPUT_JSON.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    OUTPUT_MD.write_text(build_markdown(result), encoding="utf-8")

    print("=" * 72)
    print(f"static_check_passed: {static_check_passed}")
    print(f"failed_checks: {failed_checks}")
    print(f"review_ui_phase1_static_check.md: {OUTPUT_MD.relative_to(ROOT)}")
    print(f"review_ui_phase1_static_check.json: {OUTPUT_JSON.relative_to(ROOT)}")
    print("=" * 72)


if __name__ == "__main__":
    main()
