from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from cache_policy_loader import decide_cache_policy


OUTPUT_DIR = Path("paper_repro_outputs/cache_similarity_eval_v3_real_70")
MD_PATH = OUTPUT_DIR / "policy_integration_safety_check.md"
JSON_PATH = OUTPUT_DIR / "policy_integration_safety_check.json"


def check_exists(path: Path, name: str, checks: Dict[str, Any], failures: List[str]) -> None:
    ok = path.exists()
    checks[name] = ok
    if not ok:
        failures.append(f"{name} 不存在：{path}")


def read_policy(path: Path, failures: List[str]) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception as exc:
        failures.append(f"策略配置读取失败：{exc}")
        return {}


def run_decision_tests(policy: Dict[str, Any], failures: List[str]) -> Dict[str, str]:
    cases = [
        ("1.0 / 1.0", 1.0, 1.0, "auto_hit"),
        ("0.75 / 0.75", 0.75, 0.75, "review"),
        ("0.5 / 0.5", 0.5, 0.5, "miss"),
    ]
    results: Dict[str, str] = {}
    for label, text_score, image_score, expected in cases:
        try:
            actual = decide_cache_policy(text_score, image_score, policy)["policy_decision"]
        except Exception as exc:
            actual = f"error: {exc}"
        results[label] = actual
        if actual != expected:
            failures.append(f"决策测试失败：{label} 期望 {expected}，实际 {actual}")
    return results


def build_report(payload: Dict[str, Any]) -> str:
    policy = payload["policy"]
    decision_results = payload["decision_results"]
    failed_checks = payload["failed_checks"]
    safety_check_passed = payload["safety_check_passed"]

    conclusion = (
        "当前策略缓存接入满足默认关闭、配置化、可回退要求，可以进入小范围真实运行验证阶段。"
        if safety_check_passed
        else "当前策略缓存接入仍存在检查失败项，需要先修复后再进入真实运行验证。"
    )

    failure_lines = "\n".join(f"- {item}" for item in failed_checks) if failed_checks else "- 无"
    return f"""# 策略缓存接入安全检查报告

## 1. 检查目的

该检查用于确认 plus.py 的策略缓存接入是默认关闭、可回退、可配置的。检查过程不启动 plus.py 服务，不调用 Qwen，不调用 TripoSR，也不加载 3D 模型。

## 2. 文件检查

- plus.py: {payload['file_checks'].get('plus.py')}
- plus_backup_before_policy_integration.py: {payload['file_checks'].get('plus_backup_before_policy_integration.py')}
- runtime_assets/cache_policy.json: {payload['file_checks'].get('runtime_assets/cache_policy.json')}
- cache_policy_loader.py: {payload['file_checks'].get('cache_policy_loader.py')}

## 3. 配置检查

- ENABLE_POLICY_CACHE 默认状态: {payload['enable_policy_cache_default']}
- CACHE_POLICY_PATH: {payload['cache_policy_path']}
- weak_threshold: {policy.get('weak_threshold')}
- strong_threshold: {policy.get('strong_threshold')}
- text_weight: {policy.get('text_weight')}
- image_weight: {policy.get('image_weight')}

## 4. 决策函数测试

- 1.0 / 1.0 -> {decision_results.get('1.0 / 1.0')}
- 0.75 / 0.75 -> {decision_results.get('0.75 / 0.75')}
- 0.5 / 0.5 -> {decision_results.get('0.5 / 0.5')}

## 5. 安全结论

{conclusion}

失败项：

{failure_lines}
"""


def main() -> None:
    failures: List[str] = []
    file_checks: Dict[str, Any] = {}

    paths = {
        "plus.py": Path("plus.py"),
        "plus_backup_before_policy_integration.py": Path("plus_backup_before_policy_integration.py"),
        "runtime_assets/cache_policy.json": Path("runtime_assets/cache_policy.json"),
        "cache_policy_loader.py": Path("cache_policy_loader.py"),
    }
    for name, path in paths.items():
        check_exists(path, name, file_checks, failures)

    policy = read_policy(paths["runtime_assets/cache_policy.json"], failures)
    for key in ("text_weight", "image_weight", "weak_threshold", "strong_threshold"):
        if key not in policy:
            failures.append(f"策略配置缺少字段：{key}")

    plus_text = ""
    try:
        plus_text = paths["plus.py"].read_text(encoding="utf-8", errors="replace")
    except Exception as exc:
        failures.append(f"plus.py 读取失败：{exc}")

    required_tokens = [
        "ENABLE_POLICY_CACHE",
        "CACHE_POLICY_PATH",
        "cache_policy_loader",
        "policy_decision",
        "policy_score",
    ]
    token_checks: Dict[str, bool] = {}
    for token in required_tokens:
        ok = token in plus_text
        token_checks[token] = ok
        if not ok:
            failures.append(f"plus.py 缺少关键文本：{token}")

    default_false = 'os.getenv("ENABLE_POLICY_CACHE", "0") == "1"' in plus_text
    if not default_false:
        failures.append("ENABLE_POLICY_CACHE 默认状态不是 False 或未检测到默认关闭写法")

    cache_policy_path = "runtime_assets/cache_policy.json"
    if 'os.getenv("CACHE_POLICY_PATH", "runtime_assets/cache_policy.json")' not in plus_text:
        failures.append("CACHE_POLICY_PATH 默认路径未检测到 runtime_assets/cache_policy.json")

    decision_results = run_decision_tests(policy, failures)
    safety_check_passed = not failures

    payload = {
        "safety_check_passed": safety_check_passed,
        "failed_checks": failures,
        "file_checks": file_checks,
        "token_checks": token_checks,
        "enable_policy_cache_default": False if default_false else "unknown",
        "cache_policy_path": cache_policy_path,
        "policy": policy,
        "decision_results": decision_results,
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    JSON_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    MD_PATH.write_text(build_report(payload), encoding="utf-8")

    print("=" * 72)
    print(f"safety_check_passed: {safety_check_passed}")
    print(f"failed_checks: {failures}")
    print(f"policy_integration_safety_check.md: {MD_PATH}")
    print(f"policy_integration_safety_check.json: {JSON_PATH}")
    print("=" * 72)


if __name__ == "__main__":
    main()
