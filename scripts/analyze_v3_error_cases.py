import argparse
import csv
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List


DEFAULT_EVAL_DIR = Path("paper_repro_outputs/cache_similarity_eval_v3_real")
DEFAULT_SUMMARY_CSV = DEFAULT_EVAL_DIR / "summary.csv"
DEFAULT_SUMMARY_JSON = DEFAULT_EVAL_DIR / "summary.json"
DEFAULT_WEIGHTED_DUAL_JSON = DEFAULT_EVAL_DIR / "weighted_dual_threshold_analysis/recommended_weighted_dual_threshold.json"
DEFAULT_OUTPUT_DIR = DEFAULT_EVAL_DIR / "error_analysis"


def read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8-sig"))


def parse_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def parse_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or str(value).strip() == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def metric(data: Dict[str, Any], key: str, default: Any = None) -> Any:
    if key in data:
        return data.get(key, default)
    key_metrics = data.get("key_metrics", {})
    if isinstance(key_metrics, dict):
        return key_metrics.get(key, default)
    return default


def read_rows(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"summary.csv not found: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        rows = []
        for row in reader:
            row["should_hit_bool"] = parse_bool(row.get("should_hit"))
            row["text_score_float"] = parse_float(row.get("text_score"))
            row["image_score_float"] = parse_float(row.get("image_score"))
            row["fused_score_float"] = parse_float(row.get("fused_score"))
            rows.append(row)
        return rows


def write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def false_miss_reason(row: Dict[str, Any], threshold: float) -> str:
    text_score = row["text_score_float"]
    image_score = row["image_score_float"]
    fused_score = row["fused_score_float"]
    reasons: List[str] = []

    if text_score < 0.5 and image_score >= 0.6:
        reasons.append("可能是文本相似度不足")
    elif image_score < 0.5 and text_score >= 0.6:
        reasons.append("可能是图像相似度不足")
    elif text_score < 0.5 and image_score < 0.5:
        reasons.append("文本和图像相似度均不足")
    else:
        reasons.append("融合分数不足")

    if max(0.0, threshold - 0.1) <= fused_score < threshold:
        reasons.append("接近阈值的边界漏命中，可考虑进入 review 区")

    return "；".join(reasons)


def borderline_action(row: Dict[str, Any], weak_threshold: float, strong_threshold: float) -> str:
    should_hit = row["should_hit_bool"]
    fused_score = row["fused_score_float"]
    if should_hit and weak_threshold - 0.1 <= fused_score < strong_threshold:
        return "建议进入 review 区或降低 weak_threshold"
    if (not should_hit) and weak_threshold - 0.1 <= fused_score < strong_threshold:
        return "危险负样本，需要保持保守阈值"
    if should_hit and fused_score >= strong_threshold:
        return "应自动复用"
    if (not should_hit) and fused_score >= strong_threshold:
        return "高风险误命中样本"
    return "观察样本"


def make_false_miss_rows(rows: List[Dict[str, Any]], threshold: float) -> List[Dict[str, Any]]:
    result = []
    for row in rows:
        if row["should_hit_bool"] and row["fused_score_float"] < threshold:
            result.append(
                {
                    "image": row.get("image", ""),
                    "category": row.get("category", ""),
                    "sample_type": row.get("sample_type", ""),
                    "query_text": row.get("query_text", ""),
                    "text_score": row.get("text_score", ""),
                    "image_score": row.get("image_score", ""),
                    "fused_score": row.get("fused_score", ""),
                    "best_keyword": row.get("best_keyword", ""),
                    "best_filename": row.get("best_filename", ""),
                    "reason_guess": false_miss_reason(row, threshold),
                }
            )
    return result


def make_false_hit_rows(rows: List[Dict[str, Any]], threshold: float) -> List[Dict[str, Any]]:
    result = []
    for row in rows:
        if (not row["should_hit_bool"]) and row["fused_score_float"] >= threshold:
            result.append(
                {
                    "image": row.get("image", ""),
                    "category": row.get("category", ""),
                    "sample_type": row.get("sample_type", ""),
                    "query_text": row.get("query_text", ""),
                    "text_score": row.get("text_score", ""),
                    "image_score": row.get("image_score", ""),
                    "fused_score": row.get("fused_score", ""),
                    "best_keyword": row.get("best_keyword", ""),
                    "best_filename": row.get("best_filename", ""),
                }
            )
    return result


def make_borderline_rows(rows: List[Dict[str, Any]], weak_threshold: float, strong_threshold: float) -> List[Dict[str, Any]]:
    low = weak_threshold - 0.1
    high = strong_threshold + 0.05
    result = []
    for row in rows:
        if low <= row["fused_score_float"] < high:
            result.append(
                {
                    "image": row.get("image", ""),
                    "category": row.get("category", ""),
                    "sample_type": row.get("sample_type", ""),
                    "should_hit": row.get("should_hit", ""),
                    "text_score": row.get("text_score", ""),
                    "image_score": row.get("image_score", ""),
                    "fused_score": row.get("fused_score", ""),
                    "suggested_action": borderline_action(row, weak_threshold, strong_threshold),
                }
            )
    return result


def median(values: List[float]) -> float:
    if not values:
        return 0.0
    return round(statistics.median(values), 4)


def avg(values: List[float]) -> float:
    if not values:
        return 0.0
    return round(sum(values) / len(values), 4)


def make_distribution_rows(rows: List[Dict[str, Any]], threshold: float) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row.get("sample_type", "unknown")].append(row)

    result = []
    for sample_type in sorted(grouped):
        items = grouped[sample_type]
        fused_scores = [item["fused_score_float"] for item in items]
        result.append(
            {
                "sample_type": sample_type,
                "count": len(items),
                "avg_text_score": avg([item["text_score_float"] for item in items]),
                "avg_image_score": avg([item["image_score_float"] for item in items]),
                "avg_fused_score": avg(fused_scores),
                "min_fused_score": round(min(fused_scores), 4) if fused_scores else 0.0,
                "max_fused_score": round(max(fused_scores), 4) if fused_scores else 0.0,
                "median_fused_score": median(fused_scores),
                "hit_count": sum(1 for item in items if item["fused_score_float"] >= threshold),
                "miss_count": sum(1 for item in items if item["fused_score_float"] < threshold),
            }
        )
    return result


def reason_counts(false_miss_rows: List[Dict[str, Any]]) -> Counter:
    counts: Counter = Counter()
    for row in false_miss_rows:
        reason = row.get("reason_guess", "")
        if "文本相似度不足" in reason:
            counts["文本相似度不足"] += 1
        if "图像相似度不足" in reason:
            counts["图像相似度不足"] += 1
        if "文本和图像相似度均不足" in reason:
            counts["文本和图像相似度均不足"] += 1
        if "边界漏命中" in reason:
            counts["接近阈值的边界漏命中"] += 1
        if not any(key in reason for key in ("文本相似度不足", "图像相似度不足", "文本和图像相似度均不足", "边界漏命中")):
            counts["融合分数不足"] += 1
    return counts


def write_report(
    path: Path,
    summary: Dict[str, Any],
    threshold: float,
    weak_threshold: float,
    strong_threshold: float,
    false_miss_rows: List[Dict[str, Any]],
    false_hit_rows: List[Dict[str, Any]],
    borderline_rows: List[Dict[str, Any]],
    distribution_rows: List[Dict[str, Any]],
) -> None:
    reason_counter = reason_counts(false_miss_rows)
    near_row = next((row for row in distribution_rows if row["sample_type"] == "near_positive"), {})
    near_avg = float(near_row.get("avg_fused_score") or 0)
    false_miss_count = len(false_miss_rows)
    false_hit_count = len(false_hit_rows)
    borderline_should_hit = sum(1 for row in borderline_rows if parse_bool(row.get("should_hit")))
    borderline_should_miss = len(borderline_rows) - borderline_should_hit

    lines = [
        "# v3_real 错误样本分析报告",
        "",
        "## 1. 分析目的",
        "",
        "本报告用于解释 v3_real 中 recall 偏低、false_miss_rate 偏高和 review 区间为空的原因，重点查看 false miss、false hit、边界样本和不同 sample_type 的分数分布。",
        "",
        "## 2. 总体情况",
        "",
        f"- total_samples = {summary.get('total_samples')}",
        f"- false_hit_rate = {summary.get('false_hit_rate')}",
        f"- recall = {summary.get('recall')}",
        f"- false_miss_rate = {summary.get('false_miss_rate')}",
        f"- recommended_threshold = {threshold}",
        f"- weak_threshold = {weak_threshold}",
        f"- strong_threshold = {strong_threshold}",
        "",
        "## 3. False Miss 样本分析",
        "",
        f"- false_miss_count = {false_miss_count}",
        f"- false_miss 中 positive 数量 = {sum(1 for row in false_miss_rows if row.get('sample_type') == 'positive')}",
        f"- false_miss 中 near_positive 数量 = {sum(1 for row in false_miss_rows if row.get('sample_type') == 'near_positive')}",
        "",
        "常见原因统计：",
        "",
        f"- 文本相似度不足 = {reason_counter.get('文本相似度不足', 0)}",
        f"- 图像相似度不足 = {reason_counter.get('图像相似度不足', 0)}",
        f"- 两者均不足 = {reason_counter.get('文本和图像相似度均不足', 0)}",
        f"- 接近阈值的边界漏命中 = {reason_counter.get('接近阈值的边界漏命中', 0)}",
        "",
        "## 4. False Hit 样本分析",
        "",
    ]

    if false_hit_count == 0:
        lines.append("当前没有 false hit，说明系统误复用控制较好。")
    else:
        lines.append(f"当前存在 {false_hit_count} 个 false hit，需要重点检查 hard_negative / negative 样本。")

    lines.extend(
        [
            "",
            "## 5. Borderline 样本分析",
            "",
            f"- borderline_count = {len(borderline_rows)}",
            f"- borderline should_hit=true 数量 = {borderline_should_hit}",
            f"- borderline should_hit=false 数量 = {borderline_should_miss}",
            "",
        ]
    )

    if len(borderline_rows) == 0:
        lines.append("0.6~0.75 附近样本较少，review_rate=0 可能是因为分数分布两极化。")
    else:
        lines.append("0.6~0.75 附近存在边界样本，可用于分析是否需要调整 weak_threshold 或扩大 review 区间。")
    if borderline_should_hit:
        lines.append("该区间存在 should_hit=true 样本，可考虑降低 weak_threshold 或将其纳入 review。")
    if borderline_should_miss:
        lines.append("该区间存在 should_hit=false 样本，阈值不能过低，否则会增加误复用风险。")

    lines.extend(
        [
            "",
            "## 6. 分数分布分析",
            "",
            "| sample_type | count | avg_text_score | avg_image_score | avg_fused_score | min_fused_score | max_fused_score | median_fused_score |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in distribution_rows:
        lines.append(
            f"| {row['sample_type']} | {row['count']} | {row['avg_text_score']} | "
            f"{row['avg_image_score']} | {row['avg_fused_score']} | {row['min_fused_score']} | "
            f"{row['max_fused_score']} | {row['median_fused_score']} |"
        )

    lines.extend(["", "## 7. 阶段结论", ""])
    if false_hit_count == 0:
        lines.append("- 当前策略在误复用控制方面较稳定。")
    if false_miss_count > 0:
        lines.append("- 当前主要问题是漏命中，应重点优化 positive / near_positive 的召回。")
    if len(borderline_rows) == 0:
        lines.append("- 当前确认区没有样本，说明阈值区间或样本分布需要进一步调整。")
    if near_avg < threshold:
        lines.append("- near_positive 的平均 fused_score 偏低，需要继续补充和细分 near_positive 样本，或改进图像/文本相似度计算。")
    lines.append("- 下一步建议优先检查 false_miss_cases.csv，并补充更多落在 weak/strong 阈值附近的边界样本。")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze v3_real false miss, false hit, borderline, and score distributions.")
    parser.add_argument("--summary-csv", default=str(DEFAULT_SUMMARY_CSV))
    parser.add_argument("--summary-json", default=str(DEFAULT_SUMMARY_JSON))
    parser.add_argument("--weighted-dual-json", default=str(DEFAULT_WEIGHTED_DUAL_JSON))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args()

    summary = read_json(Path(args.summary_json))
    weighted_dual = read_json(Path(args.weighted_dual_json))
    rows = read_rows(Path(args.summary_csv))

    threshold = float(summary.get("recommended_threshold") or 0.6)
    weak_threshold = float(metric(weighted_dual, "recommended_weak_threshold", 0.7))
    strong_threshold = float(metric(weighted_dual, "recommended_strong_threshold", 0.75))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    false_miss_rows = make_false_miss_rows(rows, threshold)
    false_hit_rows = make_false_hit_rows(rows, threshold)
    borderline_rows = make_borderline_rows(rows, weak_threshold, strong_threshold)
    distribution_rows = make_distribution_rows(rows, threshold)

    false_miss_fields = [
        "image",
        "category",
        "sample_type",
        "query_text",
        "text_score",
        "image_score",
        "fused_score",
        "best_keyword",
        "best_filename",
        "reason_guess",
    ]
    false_hit_fields = [
        "image",
        "category",
        "sample_type",
        "query_text",
        "text_score",
        "image_score",
        "fused_score",
        "best_keyword",
        "best_filename",
    ]
    borderline_fields = [
        "image",
        "category",
        "sample_type",
        "should_hit",
        "text_score",
        "image_score",
        "fused_score",
        "suggested_action",
    ]
    distribution_fields = [
        "sample_type",
        "count",
        "avg_text_score",
        "avg_image_score",
        "avg_fused_score",
        "min_fused_score",
        "max_fused_score",
        "median_fused_score",
        "hit_count",
        "miss_count",
    ]

    false_miss_path = output_dir / "false_miss_cases.csv"
    false_hit_path = output_dir / "false_hit_cases.csv"
    borderline_path = output_dir / "borderline_cases.csv"
    distribution_path = output_dir / "score_distribution_by_type.csv"
    report_path = output_dir / "v3_error_analysis_report.md"

    write_csv(false_miss_path, false_miss_rows, false_miss_fields)
    write_csv(false_hit_path, false_hit_rows, false_hit_fields)
    write_csv(borderline_path, borderline_rows, borderline_fields)
    write_csv(distribution_path, distribution_rows, distribution_fields)
    write_report(
        report_path,
        summary,
        threshold,
        weak_threshold,
        strong_threshold,
        false_miss_rows,
        false_hit_rows,
        borderline_rows,
        distribution_rows,
    )

    print("=" * 72)
    print(f"false_miss_count: {len(false_miss_rows)}")
    print(f"false_hit_count: {len(false_hit_rows)}")
    print(f"borderline_count: {len(borderline_rows)}")
    print(f"score_distribution_by_type.csv: {distribution_path}")
    print(f"v3_error_analysis_report.md: {report_path}")
    print("=" * 72)


if __name__ == "__main__":
    main()
