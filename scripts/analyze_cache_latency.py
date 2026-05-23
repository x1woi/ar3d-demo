import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List


DEFAULT_EVAL_DIR = Path("paper_repro_outputs/cache_similarity_eval_v2_hard")
DEFAULT_OUTPUT_DIR = DEFAULT_EVAL_DIR / "latency_analysis"


def read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8-sig"))


def parse_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def parse_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def read_samples(summary_csv: Path) -> List[Dict[str, Any]]:
    if not summary_csv.exists():
        raise FileNotFoundError(f"summary.csv not found: {summary_csv}")

    samples: List[Dict[str, Any]] = []
    with summary_csv.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        required = {"fused_score", "elapsed_ms"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"summary.csv missing required columns: {sorted(missing)}")

        for row in reader:
            samples.append(
                {
                    "decision": row.get("decision", ""),
                    "elapsed_ms": parse_float(row.get("elapsed_ms")),
                    "fused_score": parse_float(row.get("fused_score")),
                    "should_hit": parse_bool(row.get("should_hit", "")),
                    "sample_type": row.get("sample_type", ""),
                }
            )
    return samples


def get_dual_thresholds(dual: Dict[str, Any], summary: Dict[str, Any]) -> Dict[str, float]:
    key_metrics = dual.get("key_metrics", {}) if isinstance(dual.get("key_metrics", {}), dict) else {}

    weak = dual.get("recommended_weak_threshold", key_metrics.get("weak_threshold"))
    strong = dual.get("recommended_strong_threshold", key_metrics.get("strong_threshold"))

    if weak is None:
        weak = summary.get("recommended_threshold", summary.get("threshold", 0.75))
    if strong is None:
        strong = summary.get("recommended_threshold", summary.get("threshold", 0.8))

    weak_value = parse_float(weak, 0.75)
    strong_value = parse_float(strong, 0.8)
    if weak_value >= strong_value:
        weak_value = max(0.0, strong_value - 0.05)

    return {"weak_threshold": weak_value, "strong_threshold": strong_value}


def classify(fused_score: float, weak_threshold: float, strong_threshold: float) -> str:
    if fused_score >= strong_threshold:
        return "auto_hit"
    if fused_score >= weak_threshold:
        return "review"
    return "miss"


def round3(value: float) -> float:
    return round(value, 3)


def analyze_latency(
    samples: List[Dict[str, Any]],
    weak_threshold: float,
    strong_threshold: float,
    generation_ms: float,
    model_load_ms: float,
) -> Dict[str, Any]:
    total_samples = len(samples)
    avg_similarity_latency_ms = (
        sum(parse_float(sample.get("elapsed_ms")) for sample in samples) / total_samples
        if total_samples
        else 0.0
    )

    decisions = [
        classify(parse_float(sample.get("fused_score")), weak_threshold, strong_threshold)
        for sample in samples
    ]
    auto_hit_count = decisions.count("auto_hit")
    review_count = decisions.count("review")
    miss_count = decisions.count("miss")

    baseline_total_latency_ms = total_samples * generation_ms
    dual_threshold_estimated_latency_ms = (
        auto_hit_count * (avg_similarity_latency_ms + model_load_ms)
        + review_count * (avg_similarity_latency_ms + model_load_ms)
        + miss_count * (avg_similarity_latency_ms + generation_ms)
    )
    saved_latency_ms = baseline_total_latency_ms - dual_threshold_estimated_latency_ms

    return {
        "total_samples": total_samples,
        "weak_threshold": weak_threshold,
        "strong_threshold": strong_threshold,
        "generation_ms": generation_ms,
        "model_load_ms": model_load_ms,
        "avg_similarity_latency_ms": round3(avg_similarity_latency_ms),
        "auto_hit_count": auto_hit_count,
        "review_count": review_count,
        "miss_count": miss_count,
        "baseline_total_latency_ms": round3(baseline_total_latency_ms),
        "dual_threshold_estimated_latency_ms": round3(dual_threshold_estimated_latency_ms),
        "saved_latency_ms": round3(saved_latency_ms),
        "saved_latency_seconds": round3(saved_latency_ms / 1000.0),
        "speedup_ratio": round3(baseline_total_latency_ms / dual_threshold_estimated_latency_ms)
        if dual_threshold_estimated_latency_ms
        else 0.0,
        "avg_latency_baseline_ms": round3(baseline_total_latency_ms / total_samples)
        if total_samples
        else 0.0,
        "avg_latency_dual_threshold_ms": round3(dual_threshold_estimated_latency_ms / total_samples)
        if total_samples
        else 0.0,
        "avg_saved_latency_per_sample_ms": round3(saved_latency_ms / total_samples)
        if total_samples
        else 0.0,
    }


def write_summary_csv(path: Path, metrics: Dict[str, Any]) -> None:
    fieldnames = [
        "total_samples",
        "generation_ms",
        "model_load_ms",
        "avg_similarity_latency_ms",
        "auto_hit_count",
        "review_count",
        "miss_count",
        "baseline_total_latency_ms",
        "dual_threshold_estimated_latency_ms",
        "saved_latency_ms",
        "saved_latency_seconds",
        "speedup_ratio",
        "avg_latency_baseline_ms",
        "avg_latency_dual_threshold_ms",
        "avg_saved_latency_per_sample_ms",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow({key: metrics.get(key, "") for key in fieldnames})


def write_report(path: Path, metrics: Dict[str, Any]) -> None:
    lines = [
        "# 缓存复用延迟对比分析",
        "",
        "## 1. 分析目的",
        "",
        "仅有命中率指标还不够，还需要对比缓存复用机制对系统等待时间的影响。对于 AR 阅读系统，缓存判断如果能避免重复 3D 生成，就可以直接降低用户等待时间。",
        "",
        "## 2. 参数设置",
        "",
        f"- generation_ms: {metrics['generation_ms']}",
        f"- model_load_ms: {metrics['model_load_ms']}",
        f"- avg_similarity_latency_ms: {metrics['avg_similarity_latency_ms']}",
        f"- weak_threshold: {metrics['weak_threshold']}",
        f"- strong_threshold: {metrics['strong_threshold']}",
        "",
        "## 3. 延迟对比结果",
        "",
        f"- total_samples: {metrics['total_samples']}",
        f"- auto_hit_count: {metrics['auto_hit_count']}",
        f"- review_count: {metrics['review_count']}",
        f"- miss_count: {metrics['miss_count']}",
        f"- baseline_total_latency_ms: {metrics['baseline_total_latency_ms']}",
        f"- dual_threshold_estimated_latency_ms: {metrics['dual_threshold_estimated_latency_ms']}",
        f"- saved_latency_seconds: {metrics['saved_latency_seconds']}",
        f"- speedup_ratio: {metrics['speedup_ratio']}",
        f"- avg_saved_latency_per_sample_ms: {metrics['avg_saved_latency_per_sample_ms']}",
        "",
        "## 4. 结论",
        "",
        "图文融合相似度判断本身为毫秒级，而 3D 生成通常为秒级或分钟级。因此缓存强命中或 review 复用可以显著减少重复生成带来的等待时间。",
        "",
        "在当前估计参数下，双阈值策略将 auto_hit 和 review 样本按复用缓存估计，只有 miss 样本进入重新生成流程，因此总等待时间相比全部重新生成明显下降。",
        "",
        "## 5. 注意事项",
        "",
        "当前 generation_ms 为估计值，后续可以从 plus.py 实际运行日志中统计真实 TripoSR 平均生成耗时后替换。review 阶段当前按“用户确认后可复用”估计，如果前端确认后选择不复用，则需要另外计算保守版本。",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze latency savings from cache reuse experiments.")
    parser.add_argument("--eval-dir", default=str(DEFAULT_EVAL_DIR))
    parser.add_argument("--generation-ms", type=float, default=120000)
    parser.add_argument("--model-load-ms", type=float, default=1000)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args()

    eval_dir = Path(args.eval_dir)
    output_dir = Path(args.output_dir)
    summary_csv = eval_dir / "summary.csv"
    summary_json = eval_dir / "summary.json"
    dual_json = eval_dir / "dual_threshold_analysis" / "recommended_dual_threshold.json"

    samples = read_samples(summary_csv)
    summary = read_json(summary_json)
    dual = read_json(dual_json)
    thresholds = get_dual_thresholds(dual, summary)

    metrics = analyze_latency(
        samples=samples,
        weak_threshold=thresholds["weak_threshold"],
        strong_threshold=thresholds["strong_threshold"],
        generation_ms=args.generation_ms,
        model_load_ms=args.model_load_ms,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "latency_summary.json"
    csv_path = output_dir / "latency_summary.csv"
    report_path = output_dir / "latency_report.md"

    json_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    write_summary_csv(csv_path, metrics)
    write_report(report_path, metrics)

    print("=" * 72)
    print(f"latency_report.md: {report_path}")
    print(f"avg_similarity_latency_ms: {metrics['avg_similarity_latency_ms']}")
    print(f"saved_latency_seconds: {metrics['saved_latency_seconds']}")
    print(f"speedup_ratio: {metrics['speedup_ratio']}")
    print(f"avg_saved_latency_per_sample_ms: {metrics['avg_saved_latency_per_sample_ms']}")
    print("=" * 72)


if __name__ == "__main__":
    main()
