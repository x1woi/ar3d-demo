import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


DEFAULT_SUMMARY_CSV = Path("paper_repro_outputs/cache_similarity_eval_v2_hard/summary.csv")
DEFAULT_ORIGINAL_DUAL_JSON = Path(
    "paper_repro_outputs/cache_similarity_eval_v2_hard/dual_threshold_analysis/recommended_dual_threshold.json"
)
DEFAULT_WEIGHTED_DUAL_JSON = Path(
    "paper_repro_outputs/cache_similarity_eval_v2_hard/weighted_dual_threshold_analysis/recommended_weighted_dual_threshold.json"
)
DEFAULT_OUTPUT_DIR = Path("paper_repro_outputs/cache_similarity_eval_v2_hard/weighted_latency_comparison")
DEFAULT_GENERATION_MS = 51236.0
DEFAULT_MODEL_LOAD_MS = 1000.0
DEFAULT_SIMILARITY_LATENCY_MS = 7.077


def read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception:
        return {}


def parse_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    if value is None or str(value).strip() == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def parse_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def metric_from_json(data: Dict[str, Any], key: str, default: Any) -> Any:
    if key in data:
        return data.get(key, default)
    key_metrics = data.get("key_metrics", {})
    if isinstance(key_metrics, dict):
        return key_metrics.get(key, default)
    return default


def read_samples(summary_csv: Path) -> List[Dict[str, Any]]:
    if not summary_csv.exists():
        raise FileNotFoundError(f"summary.csv not found: {summary_csv}")

    with summary_csv.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        required = {"text_score", "image_score", "fused_score", "should_hit", "sample_type"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"summary.csv missing required columns: {sorted(missing)}")

        samples = []
        for line_no, row in enumerate(reader, start=2):
            text_score = parse_float(row.get("text_score"))
            fused_score = parse_float(row.get("fused_score"))
            if text_score is None:
                raise ValueError(f"text_score is missing or invalid at CSV line {line_no}")
            if fused_score is None:
                raise ValueError(f"fused_score is missing or invalid at CSV line {line_no}")
            image_score = parse_float(row.get("image_score"), text_score)
            elapsed_ms = parse_float(row.get("elapsed_ms"), None)
            samples.append(
                {
                    "sample_type": row.get("sample_type", "").strip() or "unknown",
                    "should_hit": parse_bool(row.get("should_hit", "")),
                    "text_score": text_score,
                    "image_score": image_score,
                    "fused_score": fused_score,
                    "elapsed_ms": elapsed_ms,
                }
            )
    return samples


def average_similarity_latency(samples: List[Dict[str, Any]]) -> float:
    elapsed = [sample["elapsed_ms"] for sample in samples if sample.get("elapsed_ms") is not None]
    if not elapsed:
        return DEFAULT_SIMILARITY_LATENCY_MS
    return round(sum(float(value) for value in elapsed) / len(elapsed), 3)


def decide(score: float, weak_threshold: float, strong_threshold: float) -> str:
    if score >= strong_threshold:
        return "auto_hit"
    if score >= weak_threshold:
        return "review"
    return "miss"


def round3(value: float) -> float:
    return round(value, 3)


def round4(value: float) -> float:
    return round(value, 4)


def evaluate_strategy(
    name: str,
    samples: List[Dict[str, Any]],
    score_key: str,
    weak_threshold: float,
    strong_threshold: float,
    generation_ms: float,
    model_load_ms: float,
    similarity_latency_ms: float,
) -> Dict[str, Any]:
    total = len(samples)
    baseline_total_latency_ms = total * generation_ms
    decisions = []
    for sample in samples:
        decision = decide(float(sample[score_key]), weak_threshold, strong_threshold)
        decisions.append((sample, decision))

    auto_hit = [(sample, decision) for sample, decision in decisions if decision == "auto_hit"]
    review = [(sample, decision) for sample, decision in decisions if decision == "review"]
    miss = [(sample, decision) for sample, decision in decisions if decision == "miss"]

    auto_hit_count = len(auto_hit)
    review_count = len(review)
    miss_count = len(miss)
    auto_true_hit_count = sum(1 for sample, _ in auto_hit if sample["should_hit"])
    auto_false_hit_count = sum(1 for sample, _ in auto_hit if not sample["should_hit"])
    review_true_candidate_count = sum(1 for sample, _ in review if sample["should_hit"])
    review_false_candidate_count = sum(1 for sample, _ in review if not sample["should_hit"])
    false_miss_count = sum(1 for sample, _ in miss if sample["should_hit"])
    expected_hit_count = sum(1 for sample in samples if sample["should_hit"])

    estimated_total_latency_ms = (
        (auto_hit_count + review_count) * (similarity_latency_ms + model_load_ms)
        + miss_count * (similarity_latency_ms + generation_ms)
    )
    saved_latency_ms = baseline_total_latency_ms - estimated_total_latency_ms

    def count(sample_type: str, decision_name: str, false_only: bool = False) -> int:
        return sum(
            1
            for sample, decision in decisions
            if sample["sample_type"] == sample_type
            and decision == decision_name
            and (not false_only or not sample["should_hit"])
        )

    return {
        "strategy": name,
        "weak_threshold": weak_threshold,
        "strong_threshold": strong_threshold,
        "total_samples": total,
        "generation_ms": generation_ms,
        "model_load_ms": model_load_ms,
        "similarity_latency_ms": similarity_latency_ms,
        "auto_hit_count": auto_hit_count,
        "review_count": review_count,
        "miss_count": miss_count,
        "auto_true_hit_count": auto_true_hit_count,
        "auto_false_hit_count": auto_false_hit_count,
        "auto_false_hit_rate": round4(auto_false_hit_count / auto_hit_count) if auto_hit_count else 0.0,
        "review_true_candidate_count": review_true_candidate_count,
        "review_false_candidate_count": review_false_candidate_count,
        "false_miss_count": false_miss_count,
        "false_miss_rate": round4(false_miss_count / expected_hit_count) if expected_hit_count else 0.0,
        "estimated_total_latency_ms": round3(estimated_total_latency_ms),
        "saved_latency_ms": round3(saved_latency_ms),
        "saved_latency_seconds": round3(saved_latency_ms / 1000.0),
        "speedup_ratio": round3(baseline_total_latency_ms / estimated_total_latency_ms)
        if estimated_total_latency_ms
        else 0.0,
        "avg_latency_ms": round3(estimated_total_latency_ms / total) if total else 0.0,
        "avg_saved_latency_per_sample_ms": round3(saved_latency_ms / total) if total else 0.0,
        "near_positive_auto_hit_count": count("near_positive", "auto_hit"),
        "near_positive_review_count": count("near_positive", "review"),
        "near_positive_miss_count": count("near_positive", "miss"),
        "hard_negative_auto_false_hit_count": count("hard_negative", "auto_hit", false_only=True),
        "hard_negative_review_count": count("hard_negative", "review"),
    }


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    fieldnames = [
        "strategy",
        "weak_threshold",
        "strong_threshold",
        "total_samples",
        "generation_ms",
        "model_load_ms",
        "similarity_latency_ms",
        "auto_hit_count",
        "review_count",
        "miss_count",
        "auto_true_hit_count",
        "auto_false_hit_count",
        "auto_false_hit_rate",
        "review_true_candidate_count",
        "review_false_candidate_count",
        "false_miss_count",
        "false_miss_rate",
        "estimated_total_latency_ms",
        "saved_latency_ms",
        "saved_latency_seconds",
        "speedup_ratio",
        "avg_latency_ms",
        "avg_saved_latency_per_sample_ms",
        "near_positive_auto_hit_count",
        "near_positive_review_count",
        "near_positive_miss_count",
        "hard_negative_auto_false_hit_count",
        "hard_negative_review_count",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def write_report(path: Path, original: Dict[str, Any], weighted: Dict[str, Any], text_weight: float, image_weight: float) -> None:
    lines = [
        "# 新融合权重下的延迟收益对比",
        "",
        "## 1. 实验目的",
        "",
        "该实验用于比较原 0.45 / 0.55 双阈值策略和新 0.3 / 0.7 双阈值策略在延迟收益上的差异。",
        "",
        "## 2. 对比策略",
        "",
        "- 原策略：使用原 fused_score，weak=0.75，strong=0.8。",
        f"- 新策略：weighted_score = {text_weight} * text_score + {image_weight} * image_score，weak=0.75，strong=0.8。",
        "",
        "## 3. 核心结果",
        "",
        "| 指标 | 原双阈值策略 | 新权重双阈值策略 |",
        "| --- | ---: | ---: |",
        f"| auto_hit_count | {original['auto_hit_count']} | {weighted['auto_hit_count']} |",
        f"| review_count | {original['review_count']} | {weighted['review_count']} |",
        f"| miss_count | {original['miss_count']} | {weighted['miss_count']} |",
        f"| auto_false_hit_rate | {original['auto_false_hit_rate']} | {weighted['auto_false_hit_rate']} |",
        f"| saved_latency_seconds | {original['saved_latency_seconds']} | {weighted['saved_latency_seconds']} |",
        f"| speedup_ratio | {original['speedup_ratio']} | {weighted['speedup_ratio']} |",
        f"| avg_saved_latency_per_sample_ms | {original['avg_saved_latency_per_sample_ms']} | {weighted['avg_saved_latency_per_sample_ms']} |",
        "",
        "## 4. 结果解释",
        "",
    ]

    if weighted["auto_hit_count"] > original["auto_hit_count"] and weighted["auto_false_hit_rate"] == 0:
        lines.append(
            "新融合权重在保持自动复用安全性的同时，提高了自动复用比例，从而进一步减少确认或重新生成带来的等待时间。"
        )
    if weighted["saved_latency_seconds"] > original["saved_latency_seconds"]:
        lines.append("新融合权重在当前数据集上带来了更高延迟收益。")
    if weighted["auto_hit_count"] <= original["auto_hit_count"] and weighted["saved_latency_seconds"] <= original["saved_latency_seconds"]:
        lines.append("新融合权重与原策略收益接近，仍需更多真实样本验证稳定性。")

    lines.extend(
        [
            "",
            "## 5. 注意事项",
            "",
            "当前数据集样本量较小，且 generation_ms 基于当前已有 TripoSR 记录，后续需要继续补充样本和真实耗时日志。review 阶段仍按“用户确认后复用”估计；如果用户选择不复用，则需要补充更保守的延迟估计。",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare latency savings between original and weighted dual-threshold strategies.")
    parser.add_argument("--summary-csv", default=str(DEFAULT_SUMMARY_CSV))
    parser.add_argument("--original-dual-json", default=str(DEFAULT_ORIGINAL_DUAL_JSON))
    parser.add_argument("--weighted-dual-json", default=str(DEFAULT_WEIGHTED_DUAL_JSON))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--generation-ms", type=float, default=DEFAULT_GENERATION_MS)
    parser.add_argument("--model-load-ms", type=float, default=DEFAULT_MODEL_LOAD_MS)
    args = parser.parse_args()

    samples = read_samples(Path(args.summary_csv))
    similarity_latency_ms = average_similarity_latency(samples)
    original_dual = read_json(Path(args.original_dual_json))
    weighted_dual = read_json(Path(args.weighted_dual_json))

    original_weak = float(metric_from_json(original_dual, "recommended_weak_threshold", 0.75))
    original_strong = float(metric_from_json(original_dual, "recommended_strong_threshold", 0.8))
    text_weight = float(metric_from_json(weighted_dual, "text_weight", 0.3))
    image_weight = float(metric_from_json(weighted_dual, "image_weight", 0.7))
    weighted_weak = float(metric_from_json(weighted_dual, "recommended_weak_threshold", 0.75))
    weighted_strong = float(metric_from_json(weighted_dual, "recommended_strong_threshold", 0.8))

    for sample in samples:
        sample["weighted_score"] = round4(
            text_weight * float(sample["text_score"]) + image_weight * float(sample["image_score"])
        )

    original = evaluate_strategy(
        "original_dual_threshold",
        samples,
        "fused_score",
        original_weak,
        original_strong,
        args.generation_ms,
        args.model_load_ms,
        similarity_latency_ms,
    )
    weighted = evaluate_strategy(
        "weighted_dual_threshold",
        samples,
        "weighted_score",
        weighted_weak,
        weighted_strong,
        args.generation_ms,
        args.model_load_ms,
        similarity_latency_ms,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "weighted_latency_comparison.csv"
    json_path = output_dir / "weighted_latency_comparison.json"
    report_path = output_dir / "weighted_latency_comparison_report.md"

    rows = [original, weighted]
    write_csv(csv_path, rows)
    json_path.write_text(json.dumps({"strategies": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(report_path, original, weighted, text_weight, image_weight)

    print("=" * 72)
    print(f"original_saved_latency_seconds: {original['saved_latency_seconds']}")
    print(f"weighted_saved_latency_seconds: {weighted['saved_latency_seconds']}")
    print(f"original_speedup_ratio: {original['speedup_ratio']}")
    print(f"weighted_speedup_ratio: {weighted['speedup_ratio']}")
    print(f"weighted_auto_hit_count: {weighted['auto_hit_count']}")
    print(f"weighted_review_count: {weighted['review_count']}")
    print(f"weighted_latency_comparison_report.md: {report_path}")
    print("=" * 72)


if __name__ == "__main__":
    main()
