from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List


ROOT = Path(__file__).resolve().parent
PLUS_PATH = ROOT / "plus.py"
POLICY_PATH = ROOT / "runtime_assets" / "cache_policy.json"
OUTPUT_DIR = ROOT / "paper_repro_outputs" / "cache_similarity_eval_v3_real_70"
OUTPUT_MD = OUTPUT_DIR / "plus_policy_smoke_test.md"
OUTPUT_JSON = OUTPUT_DIR / "plus_policy_smoke_test.json"


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8-sig")


def add_check(checks: List[Dict[str, Any]], name: str, passed: bool, detail: str = "") -> None:
    checks.append(
        {
            "name": name,
            "passed": bool(passed),
            "detail": detail,
        }
    )


def run_decision_tests(policy: Dict[str, Any]) -> List[Dict[str, Any]]:
    from cache_policy_loader import decide_cache_policy

    cases = [
        {"name": "auto_hit", "text_score": 1.0, "image_score": 1.0, "expected": "auto_hit"},
        {"name": "review", "text_score": 0.75, "image_score": 0.75, "expected": "review"},
        {"name": "miss", "text_score": 0.5, "image_score": 0.5, "expected": "miss"},
    ]
    results: List[Dict[str, Any]] = []
    for case in cases:
        decision = decide_cache_policy(case["text_score"], case["image_score"], policy)
        actual = decision.get("policy_decision")
        results.append(
            {
                **case,
                "actual": actual,
                "policy_score": decision.get("policy_score"),
                "passed": actual == case["expected"],
            }
        )
    return results


def build_markdown(result: Dict[str, Any]) -> str:
    file_checks = result["file_checks"]
    decision_tests = result["decision_tests"]
    failed_checks = result["failed_checks"]
    policy = result.get("policy", {})

    def mark(passed: bool) -> str:
        return "通过" if passed else "失败"

    lines = [
        "# plus.py 策略接入无模型 Smoke Test",
        "",
        "## 1. 测试目的",
        "",
        "该测试用于验证策略接入的配置、开关和决策函数，不启动完整生成流程。",
        "",
        "## 2. 测试范围",
        "",
        "- 不启动 Flask；",
        "- 不调用 Qwen；",
        "- 不调用 TripoSR；",
        "- 不加载 3D 模型；",
        "- 不修改 plus.py。",
        "",
        "## 3. 检查结果",
        "",
        f"- plus.py 是否存在：{mark(result['plus_exists'])}",
        f"- ENABLE_POLICY_CACHE 是否默认关闭：{mark(result['default_disabled'])}",
        f"- runtime_assets/cache_policy.json 是否存在：{mark(result['policy_exists'])}",
        f"- policy loader 是否可用：{mark(result['policy_loader_available'])}",
        f"- auto_hit / review / miss 测试是否通过：{mark(result['decision_tests_passed'])}",
        "",
        "### 文件与静态配置",
        "",
        "| 检查项 | 结果 | 说明 |",
        "| --- | --- | --- |",
    ]

    for item in file_checks:
        lines.append(f"| {item['name']} | {mark(item['passed'])} | {item.get('detail', '')} |")

    lines.extend(
        [
            "",
            "### 策略配置",
            "",
            f"- text_weight：{policy.get('text_weight')}",
            f"- image_weight：{policy.get('image_weight')}",
            f"- weak_threshold：{policy.get('weak_threshold')}",
            f"- strong_threshold：{policy.get('strong_threshold')}",
            "",
            "### 决策函数测试",
            "",
            "| text_score | image_score | 期望 | 实际 | policy_score | 结果 |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
    )

    for item in decision_tests:
        lines.append(
            "| {text_score} | {image_score} | {expected} | {actual} | {policy_score} | {passed} |".format(
                text_score=item["text_score"],
                image_score=item["image_score"],
                expected=item["expected"],
                actual=item["actual"],
                policy_score=item["policy_score"],
                passed=mark(item["passed"]),
            )
        )

    lines.extend(["", "## 4. 结论", ""])
    if result["smoke_test_passed"]:
        lines.append("当前 plus.py 策略接入通过无模型 smoke test，可以进入小范围真实运行验证。")
    else:
        lines.append("当前 plus.py 策略接入未通过无模型 smoke test，需要先修复以下问题：")
        for failed in failed_checks:
            lines.append(f"- {failed}")

    lines.extend(
        [
            "",
            "## 5. 运行约束确认",
            "",
            "本次测试未 import plus.py，未启动 Flask 服务，未调用 Qwen，未调用 TripoSR，未加载 3D 模型，也未修改 labels.json。",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    checks: List[Dict[str, Any]] = []
    failed_checks: List[str] = []
    policy: Dict[str, Any] = {}
    decision_tests: List[Dict[str, Any]] = []

    plus_exists = PLUS_PATH.exists()
    add_check(checks, "plus.py exists", plus_exists, str(PLUS_PATH))

    plus_text = ""
    if plus_exists:
        plus_text = read_text(PLUS_PATH)
        required_tokens = [
            "ENABLE_POLICY_CACHE",
            "CACHE_POLICY_PATH",
            'os.getenv("ENABLE_POLICY_CACHE", "0") == "1"',
        ]
        for token in required_tokens:
            add_check(checks, f"plus.py contains {token}", token in plus_text)
    else:
        for token in ["ENABLE_POLICY_CACHE", "CACHE_POLICY_PATH", 'os.getenv("ENABLE_POLICY_CACHE", "0") == "1"']:
            add_check(checks, f"plus.py contains {token}", False, "plus.py missing")

    default_disabled = 'os.getenv("ENABLE_POLICY_CACHE", "0") == "1"' in plus_text

    previous_enable = os.environ.pop("ENABLE_POLICY_CACHE", None)
    previous_policy_path = os.environ.pop("CACHE_POLICY_PATH", None)
    env_default_enabled = os.getenv("ENABLE_POLICY_CACHE", "0") == "1"
    add_check(
        checks,
        "environment default ENABLE_POLICY_CACHE is disabled",
        env_default_enabled is False,
        f"computed={env_default_enabled}",
    )

    policy_exists = POLICY_PATH.exists()
    add_check(checks, "runtime policy file exists", policy_exists, str(POLICY_PATH))

    policy_loader_available = False
    try:
        from cache_policy_loader import load_cache_policy

        policy_loader_available = True
        if policy_exists:
            policy = load_cache_policy(POLICY_PATH)
            for key in ["text_weight", "image_weight", "weak_threshold", "strong_threshold"]:
                add_check(checks, f"policy has {key}", key in policy, str(policy.get(key)))
            decision_tests = run_decision_tests(policy)
            for test in decision_tests:
                add_check(
                    checks,
                    f"decision {test['text_score']} / {test['image_score']} -> {test['expected']}",
                    test["passed"],
                    f"actual={test['actual']}, score={test['policy_score']}",
                )
        else:
            for key in ["text_weight", "image_weight", "weak_threshold", "strong_threshold"]:
                add_check(checks, f"policy has {key}", False, "policy file missing")
    except Exception as exc:
        add_check(checks, "policy loader import and decision execution", False, repr(exc))

    add_check(checks, "policy loader available", policy_loader_available)

    os.environ["ENABLE_POLICY_CACHE"] = "1"
    os.environ["CACHE_POLICY_PATH"] = str(POLICY_PATH)
    env_enable_readable = os.getenv("ENABLE_POLICY_CACHE") == "1"
    env_path_readable = os.getenv("CACHE_POLICY_PATH") == str(POLICY_PATH)
    env_policy_readable = False
    try:
        if policy_loader_available and policy_exists:
            from cache_policy_loader import load_cache_policy

            env_policy = load_cache_policy(os.getenv("CACHE_POLICY_PATH", ""))
            env_policy_readable = bool(env_policy)
    except Exception:
        env_policy_readable = False
    add_check(checks, "ENABLE_POLICY_CACHE=1 can be read", env_enable_readable)
    add_check(checks, "CACHE_POLICY_PATH can be read", env_path_readable)
    add_check(checks, "policy can be loaded when env is enabled", env_policy_readable)

    if previous_enable is None:
        os.environ.pop("ENABLE_POLICY_CACHE", None)
    else:
        os.environ["ENABLE_POLICY_CACHE"] = previous_enable
    if previous_policy_path is None:
        os.environ.pop("CACHE_POLICY_PATH", None)
    else:
        os.environ["CACHE_POLICY_PATH"] = previous_policy_path

    failed_checks = [item["name"] for item in checks if not item["passed"]]
    smoke_test_passed = not failed_checks
    result = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "smoke_test_passed": smoke_test_passed,
        "failed_checks": failed_checks,
        "plus_exists": plus_exists,
        "default_disabled": default_disabled and not env_default_enabled,
        "policy_exists": policy_exists,
        "policy_loader_available": policy_loader_available,
        "decision_tests_passed": bool(decision_tests) and all(item["passed"] for item in decision_tests),
        "policy_path": str(POLICY_PATH),
        "policy": policy,
        "decision_tests": decision_tests,
        "file_checks": checks,
        "constraints": {
            "import_plus_py": False,
            "start_flask": False,
            "call_qwen": False,
            "call_triposr": False,
            "load_3d_model": False,
            "modify_plus_py": False,
            "modify_labels_json": False,
        },
    }

    OUTPUT_JSON.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    OUTPUT_MD.write_text(build_markdown(result), encoding="utf-8")

    print("=" * 72)
    print(f"smoke_test_passed: {smoke_test_passed}")
    print(f"failed_checks: {failed_checks}")
    print(f"plus_policy_smoke_test.md: {OUTPUT_MD.relative_to(ROOT)}")
    print(f"plus_policy_smoke_test.json: {OUTPUT_JSON.relative_to(ROOT)}")
    print("=" * 72)


if __name__ == "__main__":
    main()
