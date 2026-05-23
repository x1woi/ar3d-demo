import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


DEFAULT_SUMMARY_CSV = Path("paper_repro_outputs/cache_similarity_eval_v2_hard/summary.csv")
DEFAULT_RECOMMENDED_WEIGHT_JSON = Path(
    "paper_repro_outputs/cache_similarity_eval_v2_hard/fusion_weight_ablation/recommended_fusion_weight.json"
)
DEFAULT_ORIGINAL_DUAL_JSON = Path(
    "paper_repro_outputs/cache_similarity_eval_v2_hard/dual_threshold_analysis/recommended_dual_threshold.json"
)
DEFAULT_OUTPUT_DIR = Path("paper_repro_outputs/cache_similarity_eval_v2_hard/weighted_dual_threshold_analysis")
DEFAULT_WEAK_THRESHOLDS = "0.5,0.55,0.6,0.65,0.7,0.75"
DEFAULT_STRONG_THRESHOLDS = "0.75,0.8,0.82,0.85,0.9"
SAMPLE_TYPES = ("positive", "near_positive", "hard_negative", "negative")


@dataclass
class Sample:
    image: str
    category: str
    sample_type: str
    should_hit: bool
    text_score: float
    image_score: float
    weighted_score: float
    image_score_missing: bool


@dataclass
class WeightedDualMetrics:
    text_weight: float
    image_weight: float
    weak_threshold: float
    strong_threshold: float
    total_samples: int
    positive_count: int
    near_positive_count: int
    hard_negative_count: int
    negative_count: int
    auto_hit_count: int
    auto_hit_rate: float
    auto_true_hit_count: int
    auto_false_hit_count: int
    auto_false_hit_rate: float
    review_count: int
    review_rate: float
    review_true_candidate_count: int
    review_false_candidate_count: int
    miss_count: int
    false_miss_count: int
    false_miss_rate: float
    positive_auto_hit_count: int
    positive_review_count: int
    positive_miss_count: int
    near_positive_auto_hit_count: int
    near_positive_review_count: int
    near_positive_miss_count: int
    hard_negative_auto_false_hit_count: int
    hard_negative_review_count: int
    negative_auto_false_hit_count: int
    negative_review_count: int
    strong_false_hit_rate: float
    boundary_review_rate: float
    score: float


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


def parse_thresholds(raw: str) -> List[float]:
    values = []
    for part in raw.split(","):
        part = part.strip()
        if part:
            values.append(float(part))
    return sorted(set(values))


def round4(value: float) -> float:
    return round(value, 4)


def load_recommended_weights(path: Path) -> Dict[str, float]:
    data = read_json(path)
    text_weight = parse_float(data.get("recommended_text_weight"), None)
    image_weight = parse_float(data.get("recommended_image_weight"), None)
    if text_weight is None:
        text_weight = parse_float(data.get("key_metrics", {}).get("text_weight") if isinstance(data.get("key_metrics"), dict) else None, 0.3)
    if image_weight is None:
        image_weight = parse_float(data.get("key_metrics", {}).get("image_weight") if isinstance(data.get("key_metrics"), dict) else None, 1.0 - float(text_weight))

    text_weight = 0.3 if text_weight is None else float(text_weight)
    image_weight = 1.0 - text_weight if image_weight is None else float(image_weight)
    return {"text_weight": round4(text_weight), "image_weight": round4(image_weight)}


def read_samples(summary_csv: Path, text_weight: float, image_weight: float) -> List[Sample]:
    if not summary_csv.exists():
        raise FileNotFoundError(f"summary.csv not found: {summary_csv}")

    samples: List[Sample] = []
    with summary_csv.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        required = {"text_score", "image_score", "should_hit", "sample_type"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"summary.csv missing required columns: {sorted(missing)}")

        for line_no, row in enumerate(reader, start=2):
            text_score = parse_float(row.get("text_score"))
            if text_score is None:
                raise ValueError(f"text_score is missing or invalid at CSV line {line_no}")
            raw_image_score = parse_float(row.get("image_score"), None)
            image_score_missing = raw_image_score is None
            image_score = text_score if image_score_missing else float(raw_image_score)
            weighted_score = text_weight * text_score + image_weight * image_score
            samples.append(
                Sample(
                    image=row.get("image", ""),
                    category=row.get("category", ""),
                    sample_type=row.get("sample_type", "").strip() or "unknown",
                    should_hit=parse_bool(row.get("should_hit", "")),
                    text_score=text_score,
                    image_score=image_score,
                    weighted_score=round4(weighted_score),
                    image_score_missing=image_score_missing,
                )
            )
    return samples


def decide(score: float, weak_threshold: float, strong_threshold: float) -> str:
    if score >= strong_threshold:
        return "auto_hit"
    if score >= weak_threshold:
        return "review"
    return "miss"


def evaluate_pair(
    samples: List[Sample],
    text_weight: float,
    image_weight: float,
    weak_threshold: float,
    strong_threshold: float,
) -> WeightedDualMetrics:
    decisions = [(sample, decide(sample.weighted_score, weak_threshold, strong_threshold)) for sample in samples]
    total = len(samples)
    expected_hit_count = sum(1 for sample in samples if sample.should_hit)
    type_counts = {sample_type: sum(1 for sample in samples if sample.sample_type == sample_type) for sample_type in SAMPLE_TYPES}

    auto_hit = [(sample, decision) for sample, decision in decisions if decision == "auto_hit"]
    review = [(sample, decision) for sample, decision in decisions if decision == "review"]
    miss = [(sample, decision) for sample, decision in decisions if decision == "miss"]

    auto_true_hit_count = sum(1 for sample, _ in auto_hit if sample.should_hit)
    auto_false_hit_count = sum(1 for sample, _ in auto_hit if not sample.should_hit)
    review_true_candidate_count = sum(1 for sample, _ in review if sample.should_hit)
    review_false_candidate_count = sum(1 for sample, _ in review if not sample.should_hit)
    false_miss_count = sum(1 for sample, _ in miss if sample.should_hit)

    def count(sample_type: str, decision_name: str, false_only: bool = False) -> int:
        return sum(
            1
            for sample, decision in decisions
            if sample.sample_type == sample_type
            and decision == decision_name
            and (not false_only or not sample.should_hit)
        )

    score = (
        auto_true_hit_count * 2
        + review_true_candidate_count
        - auto_false_hit_count * 5
        - review_false_candidate_count
        - false_miss_count * 0.5
    )

    auto_hit_count = len(auto_hit)
    review_count = len(review)

    return WeightedDualMetrics(
        text_weight=text_weight,
        image_weight=image_weight,
        weak_threshold=weak_threshold,
        strong_threshold=strong_threshold,
        total_samples=total,
        positive_count=type_counts["positive"],
        near_positive_count=type_counts["near_positive"],
        hard_negative_count=type_counts["hard_negative"],
        negative_count=type_counts["negative"],
        auto_hit_count=auto_hit_count,
        auto_hit_rate=round4(auto_hit_count / total) if total else 0.0,
        auto_true_hit_count=auto_true_hit_count,
        auto_false_hit_count=auto_false_hit_count,
        auto_false_hit_rate=round4(auto_false_hit_count / auto_hit_count) if auto_hit_count else 0.0,
        review_count=review_count,
        review_rate=round4(review_count / total) if total else 0.0,
        review_true_candidate_count=review_true_candidate_count,
        review_false_candidate_count=review_false_candidate_count,
        miss_count=len(miss),
        false_miss_count=false_miss_count,
        false_miss_rate=round4(false_miss_count / expected_hit_count) if expected_hit_count else 0.0,
        positive_auto_hit_count=count("positive", "auto_hit"),
        positive_review_count=count("positive", "review"),
        positive_miss_count=count("positive", "miss"),
        near_positive_auto_hit_count=count("near_positive", "auto_hit"),
        near_positive_review_count=count("near_positive", "review"),
        near_positive_miss_count=count("near_positive", "miss"),
        hard_negative_auto_false_hit_count=count("hard_negative", "auto_hit", false_only=True),
        hard_negative_review_count=count("hard_negative", "review"),
        negative_auto_false_hit_count=count("negative", "auto_hit", false_only=True),
        negative_review_count=count("negative", "review"),
        strong_false_hit_rate=round4(auto_false_hit_count / auto_hit_count) if auto_hit_count else 0.0,
        boundary_review_rate=round4(review_count / total) if total else 0.0,
        score=round4(score),
    )


def recommend(metrics: List[WeightedDualMetrics]) -> Dict[str, Any]:
    if not metrics:
        return {
            "recommended_weak_threshold": None,
            "recommended_strong_threshold": None,
            "recommended_reason": "No threshold pairs were available.",
            "key_metrics": {},
        }

    safe = [item for item in metrics if item.strong_false_hit_rate <= 0.05]
    preferred = [item for item in safe if item.review_rate <= 0.5]
    pool = preferred or safe or metrics

    def key(item: WeightedDualMetrics) -> Any:
        near_positive_covered = item.near_positive_auto_hit_count + item.near_positive_review_count
        return (
            -item.strong_false_hit_rate,
            item.score,
            item.auto_true_hit_count,
            near_positive_covered,
            -item.review_false_candidate_count,
            -item.review_rate,
            -item.strong_threshold,
            item.weak_threshold,
        )

    best = max(pool, key=key)
    if preferred:
        reason = (
            f"Selected weak={best.weak_threshold}, strong={best.strong_threshold} because "
            f"strong_false_hit_rate is {best.strong_false_hit_rate} (<= 0.05), review_rate is "
            f"{best.review_rate} (<= 0.5), and the pair has the best score ({best.score}) "
            f"while covering {best.near_positive_auto_hit_count + best.near_positive_review_count} near_positive samples."
        )
    elif safe:
        reason = (
            f"No safe pair kept review_rate <= 0.5. Selected weak={best.weak_threshold}, "
            f"strong={best.strong_threshold} because strong_false_hit_rate is "
            f"{best.strong_false_hit_rate} and it has the best available score ({best.score})."
        )
    else:
        reason = (
            f"No pair satisfied strong_false_hit_rate <= 0.05. Selected weak={best.weak_threshold}, "
            f"strong={best.strong_threshold} only as a diagnostic fallback."
        )

    return {
        "text_weight": best.text_weight,
        "image_weight": best.image_weight,
        "recommended_weak_threshold": best.weak_threshold,
        "recommended_strong_threshold": best.strong_threshold,
        "recommended_reason": reason,
        "key_metrics": asdict(best),
    }


def write_csv(path: Path, metrics: List[WeightedDualMetrics]) -> None:
    rows = [asdict(item) for item in metrics]
    fieldnames = list(WeightedDualMetrics.__dataclass_fields__.keys())
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def get_metric(data: Dict[str, Any], key: str, default: Any = "N/A") -> Any:
    if key in data:
        return data.get(key, default)
    key_metrics = data.get("key_metrics", {})
    if isinstance(key_metrics, dict):
        return key_metrics.get(key, default)
    return default


def write_report(
    path: Path,
    recommendation: Dict[str, Any],
    original_dual: Dict[str, Any],
    original_dual_path: Path,
    missing_image_count: int,
) -> None:
    key = recommendation.get("key_metrics", {})
    text_weight = recommendation.get("text_weight")
    image_weight = recommendation.get("image_weight")
    original_exists = bool(original_dual)

    lines = [
        "# 基于推荐融合权重的双阈值分析",
        "",
        "## 1. 实验目的",
        "",
        f"该实验用于验证消融实验推荐的 {text_weight}/{image_weight} 权重在双阈值机制下是否更合适。",
        "",
        "## 2. 方法",
        "",
        "使用已有 summary.csv 中的 text_score 和 image_score 重新计算：",
        "",
        "```text",
        f"weighted_score = {text_weight} * text_score + {image_weight} * image_score",
        "```",
        "",
        f"当前有 {missing_image_count} 条样本缺少 image_score，按照前一轮消融实验口径使用 text_score 兜底。",
        "",
        "## 3. 双阈值规则",
        "",
        "- weighted_score >= strong_threshold：自动复用；",
        "- weak_threshold <= weighted_score < strong_threshold：进入确认区；",
        "- weighted_score < weak_threshold：重新生成。",
        "",
        "## 4. 推荐结果",
        "",
        f"- recommended_weak_threshold: {recommendation.get('recommended_weak_threshold')}",
        f"- recommended_strong_threshold: {recommendation.get('recommended_strong_threshold')}",
        f"- auto_false_hit_rate: {key.get('auto_false_hit_rate')}",
        f"- review_rate: {key.get('review_rate')}",
        f"- near_positive_auto_hit_count: {key.get('near_positive_auto_hit_count')}",
        f"- near_positive_review_count: {key.get('near_positive_review_count')}",
        f"- recommended_reason: {recommendation.get('recommended_reason')}",
        "",
        "## 5. 与原双阈值结果对比",
        "",
    ]

    if original_exists:
        lines.extend(
            [
                f"- 原 weak_threshold / strong_threshold: {get_metric(original_dual, 'recommended_weak_threshold', get_metric(original_dual, 'weak_threshold'))} / {get_metric(original_dual, 'recommended_strong_threshold', get_metric(original_dual, 'strong_threshold'))}",
                f"- 新 weak_threshold / strong_threshold: {recommendation.get('recommended_weak_threshold')} / {recommendation.get('recommended_strong_threshold')}",
                f"- 原 auto_false_hit_rate: {get_metric(original_dual, 'auto_false_hit_rate', get_metric(original_dual, 'strong_false_hit_rate'))}",
                f"- 新 auto_false_hit_rate: {key.get('auto_false_hit_rate')}",
                f"- 原 review_rate: {get_metric(original_dual, 'review_rate', get_metric(original_dual, 'boundary_review_rate'))}",
                f"- 新 review_rate: {key.get('review_rate')}",
                f"- 原 near_positive_review_count: {get_metric(original_dual, 'near_positive_review_count')}",
                f"- 新 near_positive_review_count: {key.get('near_positive_review_count')}",
            ]
        )
    else:
        lines.append(f"- WARNING: 原双阈值结果文件缺失或读取失败：{original_dual_path}")

    lines.extend(
        [
            "",
            "## 6. 阶段结论",
            "",
        ]
    )

    original_auto_false = get_metric(original_dual, "auto_false_hit_rate", get_metric(original_dual, "strong_false_hit_rate", None))
    original_near_review = get_metric(original_dual, "near_positive_review_count", None)
    new_auto_false = key.get("auto_false_hit_rate")
    new_near_covered = key.get("near_positive_auto_hit_count", 0) + key.get("near_positive_review_count", 0)

    if original_exists and new_auto_false == original_auto_false and original_near_review is not None and new_near_covered > original_near_review:
        lines.append(
            "在当前 v2_hard 小样本上，推荐融合权重下的双阈值策略保持了自动复用安全性，并覆盖了更多 near_positive 样本，可作为后续 plus.py 接入候选。"
        )
    elif original_exists:
        lines.append(
            "推荐融合权重下的结果与原双阈值结果接近，可暂时作为待验证 baseline；后续需要扩充真实样本确认其稳定性。"
        )
    else:
        lines.append(
            "由于缺少原双阈值对照文件，本轮仅能说明推荐融合权重下的双阈值表现，后续需要补充对照结果。"
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze dual thresholds using recommended fusion weights.")
    parser.add_argument("--summary-csv", default=str(DEFAULT_SUMMARY_CSV))
    parser.add_argument("--recommended-weight-json", default=str(DEFAULT_RECOMMENDED_WEIGHT_JSON))
    parser.add_argument("--original-dual-json", default=str(DEFAULT_ORIGINAL_DUAL_JSON))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--weak-thresholds", default=DEFAULT_WEAK_THRESHOLDS)
    parser.add_argument("--strong-thresholds", default=DEFAULT_STRONG_THRESHOLDS)
    args = parser.parse_args()

    summary_csv = Path(args.summary_csv)
    recommended_weight_json = Path(args.recommended_weight_json)
    original_dual_json = Path(args.original_dual_json)
    output_dir = Path(args.output_dir)
    weak_thresholds = parse_thresholds(args.weak_thresholds)
    strong_thresholds = parse_thresholds(args.strong_thresholds)

    weights = load_recommended_weights(recommended_weight_json)
    text_weight = weights["text_weight"]
    image_weight = weights["image_weight"]
    samples = read_samples(summary_csv, text_weight, image_weight)
    missing_image_count = sum(1 for sample in samples if sample.image_score_missing)

    metrics = [
        evaluate_pair(samples, text_weight, image_weight, weak, strong)
        for weak in weak_thresholds
        for strong in strong_thresholds
        if weak < strong
    ]
    metrics.sort(key=lambda item: (item.strong_threshold, item.weak_threshold))
    recommendation = recommend(metrics)
    original_dual = read_json(original_dual_json)

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "weighted_dual_threshold_summary.csv"
    recommendation_path = output_dir / "recommended_weighted_dual_threshold.json"
    report_path = output_dir / "weighted_dual_threshold_report.md"

    write_csv(summary_path, metrics)
    recommendation_path.write_text(json.dumps(recommendation, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(report_path, recommendation, original_dual, original_dual_json, missing_image_count)

    key = recommendation.get("key_metrics", {})
    print("=" * 72)
    print(f"text_weight: {recommendation.get('text_weight')}")
    print(f"image_weight: {recommendation.get('image_weight')}")
    print(f"recommended_weak_threshold: {recommendation.get('recommended_weak_threshold')}")
    print(f"recommended_strong_threshold: {recommendation.get('recommended_strong_threshold')}")
    print(f"auto_false_hit_rate: {key.get('auto_false_hit_rate')}")
    print(f"review_rate: {key.get('review_rate')}")
    print(f"near_positive_auto_hit_count: {key.get('near_positive_auto_hit_count')}")
    print(f"near_positive_review_count: {key.get('near_positive_review_count')}")
    print(f"weighted_dual_threshold_report.md: {report_path}")
    print("=" * 72)


if __name__ == "__main__":
    main()
