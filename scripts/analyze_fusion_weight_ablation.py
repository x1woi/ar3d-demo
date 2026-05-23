import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


DEFAULT_SUMMARY_CSV = Path("paper_repro_outputs/cache_similarity_eval_v2_hard/summary.csv")
DEFAULT_OUTPUT_DIR = Path("paper_repro_outputs/cache_similarity_eval_v2_hard/fusion_weight_ablation")
DEFAULT_WEIGHTS = "0.0,0.3,0.4,0.45,0.5,0.6,0.7,1.0"
DEFAULT_THRESHOLDS = "0.5,0.6,0.7,0.75,0.8,0.82,0.85,0.9"
SAMPLE_TYPES = ("positive", "near_positive", "hard_negative", "negative")


@dataclass
class Sample:
    image: str
    category: str
    sample_type: str
    should_hit: bool
    text_score: float
    image_score: float
    image_score_missing: bool


def parse_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    if value is None or str(value).strip() == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def parse_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def parse_float_list(raw: str) -> List[float]:
    values = []
    for part in raw.split(","):
        part = part.strip()
        if part:
            values.append(float(part))
    return sorted(set(values))


def round4(value: float) -> float:
    return round(value, 4)


def read_samples(summary_csv: Path) -> List[Sample]:
    if not summary_csv.exists():
        raise FileNotFoundError(f"summary.csv not found: {summary_csv}")

    with summary_csv.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        columns = set(reader.fieldnames or [])
        required = {"text_score", "image_score", "should_hit", "sample_type"}
        missing = required - columns
        if missing:
            raise ValueError(f"summary.csv missing required columns: {sorted(missing)}")

        samples: List[Sample] = []
        for index, row in enumerate(reader, start=2):
            text_score = parse_float(row.get("text_score"))
            if text_score is None:
                raise ValueError(f"text_score is missing or invalid at CSV line {index}")

            raw_image_score = parse_float(row.get("image_score"))
            image_score_missing = raw_image_score is None
            # Current cache scoring falls back to text-only behavior when image score
            # is unavailable. Keep that behavior so ablation is comparable.
            image_score = text_score if image_score_missing else raw_image_score

            samples.append(
                Sample(
                    image=row.get("image", ""),
                    category=row.get("category", ""),
                    sample_type=row.get("sample_type", "").strip() or "unknown",
                    should_hit=parse_bool(row.get("should_hit", "")),
                    text_score=text_score,
                    image_score=image_score,
                    image_score_missing=image_score_missing,
                )
            )
    return samples


def fused_score(sample: Sample, text_weight: float) -> float:
    image_weight = 1.0 - text_weight
    return text_weight * sample.text_score + image_weight * sample.image_score


def grouped_metrics(samples: List[Sample], hits: List[bool]) -> Dict[str, Dict[str, Any]]:
    grouped: Dict[str, Dict[str, Any]] = {}
    for sample_type in sorted(set(SAMPLE_TYPES) | {sample.sample_type for sample in samples}):
        indexed = [(sample, hit) for sample, hit in zip(samples, hits) if sample.sample_type == sample_type]
        count = len(indexed)
        if count == 0:
            grouped[sample_type] = {
                "count": 0,
                "hit_count": 0,
                "miss_count": 0,
                "accuracy": 0.0,
                "false_hit_rate": 0.0,
                "false_miss_rate": 0.0,
            }
            continue
        hit_count = sum(1 for _sample, hit in indexed if hit)
        miss_count = count - hit_count
        correct = sum(1 for sample, hit in indexed if hit == sample.should_hit)
        expected_hit = sum(1 for sample, _hit in indexed if sample.should_hit)
        expected_miss = count - expected_hit
        false_hit = sum(1 for sample, hit in indexed if hit and not sample.should_hit)
        false_miss = sum(1 for sample, hit in indexed if (not hit) and sample.should_hit)
        grouped[sample_type] = {
            "count": count,
            "hit_count": hit_count,
            "miss_count": miss_count,
            "accuracy": round4(correct / count) if count else 0.0,
            "false_hit_rate": round4(false_hit / expected_miss) if expected_miss else 0.0,
            "false_miss_rate": round4(false_miss / expected_hit) if expected_hit else 0.0,
        }
    return grouped


def evaluate(samples: List[Sample], text_weight: float, threshold: float) -> Dict[str, Any]:
    image_weight = round4(1.0 - text_weight)
    scores = [fused_score(sample, text_weight) for sample in samples]
    hits = [score >= threshold for score in scores]

    tp = sum(1 for sample, hit in zip(samples, hits) if hit and sample.should_hit)
    tn = sum(1 for sample, hit in zip(samples, hits) if (not hit) and (not sample.should_hit))
    fp = sum(1 for sample, hit in zip(samples, hits) if hit and not sample.should_hit)
    fn = sum(1 for sample, hit in zip(samples, hits) if (not hit) and sample.should_hit)
    total = len(samples)

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    expected_miss = tn + fp
    expected_hit = tp + fn
    groups = grouped_metrics(samples, hits)

    row: Dict[str, Any] = {
        "text_weight": round4(text_weight),
        "image_weight": image_weight,
        "threshold": threshold,
        "total_samples": total,
        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "accuracy": round4((tp + tn) / total) if total else 0.0,
        "precision": round4(precision),
        "recall": round4(recall),
        "f1": round4(f1),
        "false_hit_rate": round4(fp / expected_miss) if expected_miss else 0.0,
        "false_miss_rate": round4(fn / expected_hit) if expected_hit else 0.0,
        "cost_score": round4(tp * 1.0 + tn * 0.5 - fp * 3.0 - fn * 1.0),
    }

    for sample_type in SAMPLE_TYPES:
        metrics = groups.get(sample_type, {})
        for metric_name in ("count", "hit_count", "miss_count", "accuracy", "false_hit_rate", "false_miss_rate"):
            row[f"{sample_type}_{metric_name}"] = metrics.get(metric_name, 0)

    return row


def recommend_threshold_for_weight(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    safe = [
        row
        for row in rows
        if row["false_hit_rate"] <= 0.05
        and row["recall"] > 0
        and row.get("positive_false_miss_rate", 0.0) < 1.0
    ]
    if safe:
        best = max(safe, key=lambda row: (row["cost_score"], row["recall"], row["f1"], -row["threshold"]))
        reason = (
            f"Selected threshold {best['threshold']} because false_hit_rate <= 0.05, "
            f"recall > 0, positive_false_miss_rate < 1.0, and cost_score is highest."
        )
    else:
        nonzero = [row for row in rows if row["recall"] > 0]
        if nonzero:
            best = min(nonzero, key=lambda row: (row["false_hit_rate"], -row["cost_score"], -row["recall"]))
            reason = (
                f"No threshold satisfied the strict safe condition. Selected threshold {best['threshold']} "
                f"because recall > 0 and false_hit_rate is lowest; ties use cost_score."
            )
        else:
            best = min(rows, key=lambda row: (row["false_hit_rate"], -row["cost_score"]))
            reason = (
                "All thresholds have recall = 0 under this fusion weight, so no effective reuse threshold "
                f"was found. Selected threshold {best['threshold']} only as a diagnostic fallback."
            )

    return {**best, "best_threshold": best["threshold"], "recommended_reason": reason}


def recommend_fusion_weight(best_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    candidates = [
        row for row in best_rows if row["false_hit_rate"] <= 0.05 and row["recall"] > 0
    ]
    pool = candidates or [row for row in best_rows if row["recall"] > 0] or best_rows

    def balanced_bonus(text_weight: float) -> int:
        return 1 if 0.4 <= text_weight <= 0.6 else 0

    best = max(
        pool,
        key=lambda row: (
            row["cost_score"],
            -row.get("near_positive_false_miss_rate", 1.0),
            row["recall"],
            -row["false_hit_rate"],
            balanced_bonus(row["text_weight"]),
            -abs(row["text_weight"] - 0.5),
        ),
    )

    reason = (
        f"Selected text_weight={best['text_weight']} and image_weight={best['image_weight']} "
        f"with threshold={best['best_threshold']} because it keeps false_hit_rate at "
        f"{best['false_hit_rate']}, recall at {best['recall']}, cost_score at "
        f"{best['cost_score']}, and near_positive_false_miss_rate at "
        f"{best.get('near_positive_false_miss_rate', 'N/A')}."
    )
    if 0.4 <= best["text_weight"] <= 0.6:
        reason += " It is also in the balanced fusion range 0.4-0.6."

    return {
        "recommended_text_weight": best["text_weight"],
        "recommended_image_weight": best["image_weight"],
        "best_threshold": best["best_threshold"],
        "recommended_reason": reason,
        "key_metrics": best,
    }


def write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def write_report(path: Path, recommendation: Dict[str, Any], best_rows: List[Dict[str, Any]], missing_image_count: int) -> None:
    key = recommendation.get("key_metrics", {})
    current = next((row for row in best_rows if abs(row["text_weight"] - 0.45) < 1e-9), None)
    pure_text = next((row for row in best_rows if abs(row["text_weight"] - 1.0) < 1e-9), None)
    pure_image = next((row for row in best_rows if abs(row["text_weight"] - 0.0) < 1e-9), None)

    def short(row: Optional[Dict[str, Any]]) -> str:
        if not row:
            return "N/A"
        return (
            f"threshold={row['best_threshold']}, precision={row['precision']}, "
            f"recall={row['recall']}, false_hit_rate={row['false_hit_rate']}, "
            f"near_positive_false_miss_rate={row.get('near_positive_false_miss_rate')}"
        )

    lines = [
        "# 图文融合权重消融实验",
        "",
        "## 1. 实验目的",
        "",
        "该实验用于比较纯文本、纯图像和不同图文融合权重下的缓存命中效果，判断当前规则融合是否优于单模态方案。",
        "",
        "## 2. 实验方法",
        "",
        "对已有 summary.csv 中的 text_score 和 image_score 重新计算融合分数：",
        "",
        "```text",
        "fused_score = text_weight * text_score + image_weight * image_score",
        "image_weight = 1.0 - text_weight",
        "```",
        "",
        f"注意：当前数据中有 {missing_image_count} 条样本缺少 image_score。为保持与现有缓存评分逻辑一致，这些样本使用 text_score 作为 image_score 的兜底值。",
        "",
        "## 3. 对比对象",
        "",
        "- 纯文本：text_weight = 1.0",
        "- 纯图像：text_weight = 0.0",
        "- 当前方案：text_weight = 0.45, image_weight = 0.55",
        "- 其他融合权重：0.3 / 0.4 / 0.5 / 0.6 / 0.7",
        "",
        "## 4. 主要结果",
        "",
        f"- recommended_text_weight: {recommendation.get('recommended_text_weight')}",
        f"- recommended_image_weight: {recommendation.get('recommended_image_weight')}",
        f"- best_threshold: {recommendation.get('best_threshold')}",
        f"- precision: {key.get('precision')}",
        f"- recall: {key.get('recall')}",
        f"- false_hit_rate: {key.get('false_hit_rate')}",
        f"- near_positive_false_miss_rate: {key.get('near_positive_false_miss_rate')}",
        f"- recommended_reason: {recommendation.get('recommended_reason')}",
        "",
        "对比摘要：",
        "",
        f"- 纯文本：{short(pure_text)}",
        f"- 纯图像：{short(pure_image)}",
        f"- 当前 0.45 / 0.55：{short(current)}",
        "",
        "## 5. 阶段性结论",
        "",
    ]

    rec_text_weight = recommendation.get("recommended_text_weight")
    if rec_text_weight in (0.0, 1.0):
        lines.append(
            "当前小规模 v2_hard 数据下，推荐结果偏向单模态方案。这说明现有样本分布或评分兜底逻辑仍会影响融合效果，后续需要扩充样本后再判断融合权重的稳定性。"
        )
    else:
        lines.append(
            "当前结果显示融合权重优于单一模态或至少能在安全性与复用率之间取得更稳妥的平衡，说明图文融合策略具备继续推进价值。"
        )

    lines.extend(
        [
            "",
            "## 6. 后续工作",
            "",
            "当前仍是规则融合 baseline。后续在样本量扩大后，可引入 Logistic Regression / RandomForest / MLP 等学习型方法，对文本相似度、图像相似度和样本类型特征进行联合判断。",
        ]
    )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run fusion weight ablation from existing cache similarity summary.csv.")
    parser.add_argument("--summary-csv", default=str(DEFAULT_SUMMARY_CSV))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--weights", default=DEFAULT_WEIGHTS)
    parser.add_argument("--thresholds", default=DEFAULT_THRESHOLDS)
    args = parser.parse_args()

    summary_csv = Path(args.summary_csv)
    output_dir = Path(args.output_dir)
    weights = parse_float_list(args.weights)
    thresholds = parse_float_list(args.thresholds)

    samples = read_samples(summary_csv)
    missing_image_count = sum(1 for sample in samples if sample.image_score_missing)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_rows: List[Dict[str, Any]] = []
    best_rows: List[Dict[str, Any]] = []
    for weight in weights:
        rows = [evaluate(samples, weight, threshold) for threshold in thresholds]
        all_rows.extend(rows)
        best_rows.append(recommend_threshold_for_weight(rows))

    recommendation = recommend_fusion_weight(best_rows)

    base_fields = [
        "text_weight",
        "image_weight",
        "threshold",
        "total_samples",
        "TP",
        "TN",
        "FP",
        "FN",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "false_hit_rate",
        "false_miss_rate",
        "cost_score",
    ]
    group_fields = [
        f"{sample_type}_{metric_name}"
        for sample_type in SAMPLE_TYPES
        for metric_name in ("count", "hit_count", "miss_count", "accuracy", "false_hit_rate", "false_miss_rate")
    ]
    threshold_summary_path = output_dir / "fusion_weight_threshold_summary.csv"
    best_summary_path = output_dir / "fusion_weight_best_summary.csv"
    recommendation_path = output_dir / "recommended_fusion_weight.json"
    report_path = output_dir / "fusion_weight_ablation_report.md"

    write_csv(threshold_summary_path, all_rows, base_fields + group_fields)

    best_fields = [
        "text_weight",
        "image_weight",
        "best_threshold",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "false_hit_rate",
        "false_miss_rate",
        "cost_score",
        "positive_false_miss_rate",
        "near_positive_false_miss_rate",
        "hard_negative_false_hit_rate",
        "recommended_reason",
    ]
    write_csv(best_summary_path, best_rows, best_fields)
    recommendation_path.write_text(json.dumps(recommendation, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(report_path, recommendation, best_rows, missing_image_count)

    key = recommendation.get("key_metrics", {})
    print("=" * 72)
    print(f"recommended_text_weight: {recommendation.get('recommended_text_weight')}")
    print(f"recommended_image_weight: {recommendation.get('recommended_image_weight')}")
    print(f"best_threshold: {recommendation.get('best_threshold')}")
    print(f"false_hit_rate: {key.get('false_hit_rate')}")
    print(f"recall: {key.get('recall')}")
    print(f"near_positive_false_miss_rate: {key.get('near_positive_false_miss_rate')}")
    print(f"fusion_weight_ablation_report.md: {report_path}")
    print("=" * 72)


if __name__ == "__main__":
    main()
