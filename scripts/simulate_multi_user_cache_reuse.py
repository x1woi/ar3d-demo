from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List

from cache_policy_loader import decide_cache_policy, load_cache_policy


DEFAULT_SUMMARY_CSV = Path("paper_repro_outputs/cache_similarity_eval_v3_real_70/summary.csv")
DEFAULT_POLICY_PATH = Path("runtime_assets/cache_policy.json")
DEFAULT_OUTPUT_DIR = Path(
    "paper_repro_outputs/cache_similarity_eval_v3_real_70/multi_user_cache_simulation"
)


def parse_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def parse_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def load_rows(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: Iterable[Dict[str, Any]], fieldnames: List[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def simulate(args: argparse.Namespace) -> Dict[str, Any]:
    summary_csv = Path(args.summary_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_rows(summary_csv)
    policy = load_cache_policy(args.policy_path)

    result_rows: List[Dict[str, Any]] = []
    for idx, row in enumerate(rows, start=1):
        text_score = parse_float(row.get("text_score"))
        image_score = parse_float(row.get("image_score"), text_score)
        elapsed_ms = parse_float(row.get("elapsed_ms"))
        should_hit = parse_bool(row.get("should_hit"))
        decision = decide_cache_policy(text_score, image_score, policy)
        policy_decision = decision["policy_decision"]
        best_model_path = str(row.get("best_model_path") or "")
        model_exists = Path(best_model_path).exists()

        baseline_latency_ms = args.generation_ms
        auto_reuse = policy_decision == "auto_hit" and model_exists
        review_candidate = policy_decision == "review" and model_exists
        miss = policy_decision == "miss" or (
            policy_decision in {"auto_hit", "review"} and not model_exists
        )

        if auto_reuse:
            simulated_latency_ms = elapsed_ms + args.network_delay_ms + args.model_load_ms
            triposr_called = False
            fallback_to_generation = False
            effective_action = "remote_cache_reuse"
        elif review_candidate:
            reuse_latency = elapsed_ms + args.network_delay_ms + args.model_load_ms
            fallback_latency = elapsed_ms + args.generation_ms
            simulated_latency_ms = (
                args.review_accept_rate * reuse_latency
                + (1.0 - args.review_accept_rate) * fallback_latency
            )
            triposr_called = args.review_accept_rate < 1.0
            fallback_to_generation = args.review_accept_rate < 1.0
            effective_action = "review_expected_reuse"
        else:
            simulated_latency_ms = elapsed_ms + args.generation_ms
            triposr_called = False
            fallback_to_generation = True
            effective_action = "fallback_generation_not_executed"

        saved_latency_ms = baseline_latency_ms - simulated_latency_ms
        result_rows.append(
            {
                "index": idx,
                "image": row.get("image", ""),
                "sample_type": row.get("sample_type", ""),
                "should_hit": should_hit,
                "text_score": text_score,
                "image_score": image_score,
                "policy_score": decision["policy_score"],
                "policy_decision": policy_decision,
                "best_keyword": row.get("best_keyword", ""),
                "best_model_path": best_model_path,
                "best_model_path_exists": model_exists,
                "auto_reuse": auto_reuse,
                "review_candidate": review_candidate,
                "fallback_to_generation": fallback_to_generation,
                "triposr_called_by_simulation": triposr_called,
                "effective_action": effective_action,
                "baseline_latency_ms": round(baseline_latency_ms, 3),
                "simulated_latency_ms": round(simulated_latency_ms, 3),
                "saved_latency_ms": round(saved_latency_ms, 3),
                "auto_false_hit": auto_reuse and not should_hit,
                "review_false_candidate": review_candidate and not should_hit,
                "false_miss": miss and should_hit,
            }
        )

    total = len(result_rows)
    auto_hit_rows = [r for r in result_rows if r["policy_decision"] == "auto_hit"]
    review_rows = [r for r in result_rows if r["policy_decision"] == "review"]
    miss_rows = [r for r in result_rows if r["policy_decision"] == "miss"]
    should_hit_rows = [r for r in result_rows if r["should_hit"]]
    should_not_hit_rows = [r for r in result_rows if not r["should_hit"]]
    auto_false_hit_rows = [r for r in result_rows if r["auto_false_hit"]]
    review_false_rows = [r for r in result_rows if r["review_false_candidate"]]
    false_miss_rows = [r for r in result_rows if r["false_miss"]]
    true_reuse_rows = [r for r in result_rows if r["should_hit"] and (r["auto_reuse"] or r["review_candidate"])]

    baseline_total = sum(float(r["baseline_latency_ms"]) for r in result_rows)
    simulated_total = sum(float(r["simulated_latency_ms"]) for r in result_rows)
    saved_total = baseline_total - simulated_total

    summary = {
        "summary_csv": str(summary_csv),
        "policy_path": str(args.policy_path),
        "total_samples": total,
        "generation_ms": args.generation_ms,
        "network_delay_ms": args.network_delay_ms,
        "model_load_ms": args.model_load_ms,
        "review_accept_rate": args.review_accept_rate,
        "baseline_total_latency_ms": round(baseline_total, 3),
        "multi_user_simulated_total_latency_ms": round(simulated_total, 3),
        "saved_latency_ms": round(saved_total, 3),
        "saved_latency_seconds": round(saved_total / 1000.0, 3),
        "speedup_ratio": round(baseline_total / simulated_total, 4) if simulated_total else None,
        "avg_saved_latency_per_sample_ms": round(saved_total / total, 3) if total else 0,
        "should_hit_count": len(should_hit_rows),
        "should_not_hit_count": len(should_not_hit_rows),
        "auto_hit_count": len(auto_hit_rows),
        "review_count": len(review_rows),
        "miss_count": len(miss_rows),
        "auto_false_hit_count": len(auto_false_hit_rows),
        "review_false_candidate_count": len(review_false_rows),
        "false_miss_count": len(false_miss_rows),
        "auto_false_hit_rate": round(
            len(auto_false_hit_rows) / len(should_not_hit_rows), 6
        )
        if should_not_hit_rows
        else 0,
        "recall_if_review_accepted": round(len(true_reuse_rows) / len(should_hit_rows), 6)
        if should_hit_rows
        else 0,
        "avg_similarity_elapsed_ms": round(
            mean(parse_float(r.get("elapsed_ms")) for r in rows), 3
        )
        if rows
        else 0,
        "notes": (
            "This is an offline multi-user cache reuse simulation. It does not call Qwen, "
            "does not call TripoSR, and does not transfer or load GLB files."
        ),
    }

    fieldnames = list(result_rows[0].keys()) if result_rows else []
    detail_csv = output_dir / "multi_user_cache_simulation_results.csv"
    summary_json = output_dir / "multi_user_cache_simulation_summary.json"
    report_md = output_dir / "multi_user_cache_simulation_report.md"
    write_csv(detail_csv, result_rows, fieldnames)
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    report = f"""# 多用户缓存复用仿真实验报告

## 1. 实验目的

本实验用于回应会议中提出的“多用户 AR 缓存共享”方向：当一个用户已经生成过 3D 模型时，另一个用户是否可以通过图文相似度判断复用该缓存模型，从而用传输延迟换取生成算力，减少重复 TripoSR 调用。

## 2. 仿真设置

- 输入结果：`{summary_csv}`
- 策略配置：`{args.policy_path}`
- generation_ms = {args.generation_ms}
- network_delay_ms = {args.network_delay_ms}
- model_load_ms = {args.model_load_ms}
- review_accept_rate = {args.review_accept_rate}

本轮为离线仿真，不调用 Qwen，不调用 TripoSR，不真实传输 GLB。

## 3. 核心结果

- total_samples = {summary['total_samples']}
- auto_hit_count = {summary['auto_hit_count']}
- review_count = {summary['review_count']}
- miss_count = {summary['miss_count']}
- auto_false_hit_count = {summary['auto_false_hit_count']}
- review_false_candidate_count = {summary['review_false_candidate_count']}
- recall_if_review_accepted = {summary['recall_if_review_accepted']}
- baseline_total_latency_ms = {summary['baseline_total_latency_ms']}
- multi_user_simulated_total_latency_ms = {summary['multi_user_simulated_total_latency_ms']}
- saved_latency_seconds = {summary['saved_latency_seconds']}
- speedup_ratio = {summary['speedup_ratio']}
- avg_saved_latency_per_sample_ms = {summary['avg_saved_latency_per_sample_ms']}

## 4. 阶段结论

在当前参数下，多用户缓存复用仿真显示：如果远端缓存命中能够通过图文相似度策略筛出，并且传输延迟低于重新生成耗时，则该机制具备降低总等待时间的潜力。

该实验目前只验证“时间账”和“误复用风险统计”，下一步可以把它扩展为单机双进程或局域网双设备验证。

## 5. 下一步

1. 单机启动两个 AR 实例，模拟 user_A / user_B；
2. 固定网络传输延迟，例如 20 秒；
3. 记录远端缓存命中时是否跳过 TripoSR；
4. 统计生成耗时、传输耗时、总耗时和误复用；
5. 若稳定，再考虑真实局域网双设备实验。
"""
    report_md.write_text(report, encoding="utf-8")

    summary.update(
        {
            "detail_csv": str(detail_csv),
            "summary_json": str(summary_json),
            "report_md": str(report_md),
        }
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline multi-user cache reuse simulation.")
    parser.add_argument("--summary-csv", default=str(DEFAULT_SUMMARY_CSV))
    parser.add_argument("--policy-path", default=str(DEFAULT_POLICY_PATH))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--generation-ms", type=float, default=51236)
    parser.add_argument("--network-delay-ms", type=float, default=20000)
    parser.add_argument("--model-load-ms", type=float, default=1000)
    parser.add_argument("--review-accept-rate", type=float, default=1.0)
    args = parser.parse_args()
    result = simulate(args)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
