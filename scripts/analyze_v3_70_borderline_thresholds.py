from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Tuple


DEFAULT_EVAL_DIR = Path("paper_repro_outputs/cache_similarity_eval_v3_real_70")
DEFAULT_SUMMARY_CSV = DEFAULT_EVAL_DIR / "summary.csv"
DEFAULT_SUMMARY_JSON = DEFAULT_EVAL_DIR / "summary.json"
DEFAULT_BORDERLINE_CSV = DEFAULT_EVAL_DIR / "error_analysis" / "borderline_cases.csv"
DEFAULT_WEIGHTED_DUAL_JSON = (
    DEFAULT_EVAL_DIR / "weighted_dual_threshold_analysis" / "recommended_weighted_dual_threshold.json"
)
DEFAULT_OUTPUT_DIR = DEFAULT_EVAL_DIR / "borderline_threshold_analysis"

WEAK_THRESHOLDS = [0.5, 0.55, 0.6, 0.62, 0.65, 0.68, 0.7]
STRONG_THRESHOLDS = [0.7, 0.72, 0.75, 0.78, 0.8, 0.82]
SAMPLE_TYPES = ["positive", "near_positive", "hard_negative", "negative"]


def read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8-sig"))


def as_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def as_float(value: Any, default: float = 0.0) -> float:
    text = str(value if value is not None else "").strip()
    if not text:
        return default
    try:
        return float(text)
    except ValueError:
        return default


def round4(value: float) -> float:
    return round(value, 4)


def weighted_score(row: Dict[str, str], text_weight: float = 0.5, image_weight: float = 0.5) -> float:
    text_score = as_float(row.get("text_score"), 0.0)
    image_text = str(row.get("image_score") or "").strip()
    image_score = as_float(image_text, text_score) if image_text else text_score
    return text_weight * text_score + image_weight * image_score


def summarize_scores(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    scores = [as_float(row.get("fused_score")) for row in rows]
    text_scores = [as_float(row.get("text_score")) for row in rows]
    image_scores = [as_float(row.get("image_score"), as_float(row.get("text_score"))) for row in rows]
    return {
        "avg_text_score": round4(mean(text_scores)) if text_scores else 0.0,
        "avg_image_score": round4(mean(image_scores)) if image_scores else 0.0,
        "avg_fused_score": round4(mean(scores)) if scores else 0.0,
        "min_fused_score": round4(min(scores)) if scores else 0.0,
        "max_fused_score": round4(max(scores)) if scores else 0.0,
    }


def suggested_action(row: Dict[str, Any]) -> str:
    should_hit = as_bool(row.get("should_hit"))
    score = as_float(row.get("fused_score"))
    if should_hit and 0.6 <= score < 0.75:
        return "建议进入 review 区，提高召回。"
    if (not should_hit) and 0.6 <= score < 0.75:
        return "危险边界负样本，不能过度降低阈值。"
    if should_hit and score >= 0.75:
        return "应自动复用。"
    if (not should_hit) and score >= 0.75:
        return "高风险误命中，需要保持强阈值或增加负向保护。"
    return "非主要边界样本，保留复核。"


def build_borderline_review(
    borderline_rows: List[Dict[str, str]],
    summary_rows: List[Dict[str, str]],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    summary_by_image = {row.get("image", ""): row for row in summary_rows}
    review_rows: List[Dict[str, Any]] = []
    for row in borderline_rows:
        full = summary_by_image.get(row.get("image", ""), {})
        merged = dict(full)
        merged.update({k: v for k, v in row.items() if v not in (None, "")})
        if not str(merged.get("image_score") or "").strip() and full:
            merged["image_score"] = full.get("image_score", "")
        merged["fused_score"] = round4(weighted_score(merged, 0.5, 0.5))
        merged["suggested_action"] = suggested_action(merged)
        review_rows.append(
            {
                "image": merged.get("image", ""),
                "sample_type": merged.get("sample_type", ""),
                "should_hit": merged.get("should_hit", ""),
                "category": merged.get("category", ""),
                "text_score": merged.get("text_score", ""),
                "image_score": merged.get("image_score", ""),
                "fused_score": merged.get("fused_score", ""),
                "best_keyword": merged.get("best_keyword", ""),
                "best_filename": merged.get("best_filename", ""),
                "suggested_action": merged.get("suggested_action", ""),
            }
        )

    stats: Dict[str, Any] = {
        "borderline_count": len(review_rows),
        "borderline_should_hit_true_count": sum(1 for row in review_rows if as_bool(row.get("should_hit"))),
        "borderline_should_hit_false_count": sum(1 for row in review_rows if not as_bool(row.get("should_hit"))),
    }
    for sample_type in SAMPLE_TYPES:
        stats[f"{sample_type}_count"] = sum(1 for row in review_rows if row.get("sample_type") == sample_type)
    stats.update(summarize_scores(review_rows))
    return review_rows, stats


def evaluate_pair(rows: List[Dict[str, str]], weak: float, strong: float) -> Dict[str, Any]:
    should_hit_count = sum(1 for row in rows if as_bool(row.get("should_hit")))
    auto_hit: List[Dict[str, str]] = []
    review: List[Dict[str, str]] = []
    miss: List[Dict[str, str]] = []
    for row in rows:
        score = weighted_score(row, 0.5, 0.5)
        if score >= strong:
            auto_hit.append(row)
        elif score >= weak:
            review.append(row)
        else:
            miss.append(row)

    auto_true = sum(1 for row in auto_hit if as_bool(row.get("should_hit")))
    auto_false = len(auto_hit) - auto_true
    review_true = sum(1 for row in review if as_bool(row.get("should_hit")))
    review_false = len(review) - review_true
    false_miss = sum(1 for row in miss if as_bool(row.get("should_hit")))
    auto_false_hit_rate = auto_false / len(auto_hit) if auto_hit else 0.0
    recall_if_review_accepted = (auto_true + review_true) / should_hit_count if should_hit_count else 0.0
    false_miss_rate = false_miss / should_hit_count if should_hit_count else 0.0
    score = (
        auto_true * 2.0
        + review_true * 1.5
        - auto_false * 6.0
        - review_false * 1.5
        - false_miss * 0.5
    )

    def group_count(group: Iterable[Dict[str, str]], sample_type: str, should_hit_value: bool | None = None) -> int:
        total = 0
        for item in group:
            if item.get("sample_type") != sample_type:
                continue
            if should_hit_value is not None and as_bool(item.get("should_hit")) != should_hit_value:
                continue
            total += 1
        return total

    return {
        "weak_threshold": weak,
        "strong_threshold": strong,
        "total_samples": len(rows),
        "auto_hit_count": len(auto_hit),
        "review_count": len(review),
        "miss_count": len(miss),
        "auto_true_hit_count": auto_true,
        "auto_false_hit_count": auto_false,
        "auto_false_hit_rate": round4(auto_false_hit_rate),
        "review_true_candidate_count": review_true,
        "review_false_candidate_count": review_false,
        "false_miss_count": false_miss,
        "false_miss_rate": round4(false_miss_rate),
        "recall_if_review_accepted": round4(recall_if_review_accepted),
        "hard_negative_auto_false_hit_count": group_count(auto_hit, "hard_negative", False),
        "hard_negative_review_count": group_count(review, "hard_negative"),
        "negative_auto_false_hit_count": group_count(auto_hit, "negative", False),
        "negative_review_count": group_count(review, "negative"),
        "near_positive_auto_hit_count": group_count(auto_hit, "near_positive", True),
        "near_positive_review_count": group_count(review, "near_positive", True),
        "near_positive_miss_count": group_count(miss, "near_positive", True),
        "score": round4(score),
    }


def recommend(metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
    safe = [row for row in metrics if as_float(row["auto_false_hit_rate"]) <= 0.05]
    with_true_review = [row for row in safe if int(row["review_true_candidate_count"]) > 0]
    with_review = [row for row in safe if int(row["review_count"]) > 0]
    pool = with_true_review or with_review or safe or metrics

    def sort_key(row: Dict[str, Any]) -> Tuple[Any, ...]:
        width = as_float(row["strong_threshold"]) - as_float(row["weak_threshold"])
        return (
            as_float(row["auto_false_hit_rate"]) <= 0.05,
            int(row["review_true_candidate_count"]) > 0,
            as_float(row["recall_if_review_accepted"]),
            -int(row["review_false_candidate_count"]),
            as_float(row["recall_if_review_accepted"]),
            int(row["near_positive_review_count"]),
            as_float(row["score"]),
            -width,
            as_float(row["weak_threshold"]),
        )

    best = max(pool, key=sort_key)
    if int(best["review_true_candidate_count"]) > 0 and as_float(best["auto_false_hit_rate"]) <= 0.05:
        reason = (
            f"Selected weak={best['weak_threshold']}, strong={best['strong_threshold']} because "
            f"auto_false_hit_rate={best['auto_false_hit_rate']} stays <= 0.05 while review_count="
            f"{best['review_count']} brings {best['review_true_candidate_count']} true boundary samples into confirmation."
        )
    elif int(best["review_count"]) == 0:
        reason = (
            "No safe candidate produced review_count > 0; selected the safest/highest-score pair as fallback. "
            "This indicates the score distribution is still relatively polarized."
        )
    else:
        reason = "Selected fallback pair by score because no ideal safe pair was available."

    return {
        "recommended_weak_threshold": best["weak_threshold"],
        "recommended_strong_threshold": best["strong_threshold"],
        "recommended_reason": reason,
        "auto_false_hit_rate": best["auto_false_hit_rate"],
        "review_count": best["review_count"],
        "review_true_candidate_count": best["review_true_candidate_count"],
        "review_false_candidate_count": best["review_false_candidate_count"],
        "recall_if_review_accepted": best["recall_if_review_accepted"],
        "near_positive_review_count": best["near_positive_review_count"],
        "hard_negative_review_count": best["hard_negative_review_count"],
        "key_metrics": best,
    }


def write_report(
    path: Path,
    border_stats: Dict[str, Any],
    original: Dict[str, Any],
    recommendation: Dict[str, Any],
) -> None:
    lines = [
        "# v3_real_70 边界样本与双阈值区间重扫报告",
        "",
        "## 1. 分析目的",
        "",
        "当前 70 条样本下 false_hit_rate 为 0，但 recall 仍有提升空间，且原双阈值 review_rate 为 0。因此本报告复核 borderline 样本，并重新扫描 weak/strong 阈值区间。",
        "",
        "## 2. Borderline 样本统计",
        "",
        f"- borderline_count: {border_stats.get('borderline_count')}",
        f"- should_hit=true 数量: {border_stats.get('borderline_should_hit_true_count')}",
        f"- should_hit=false 数量: {border_stats.get('borderline_should_hit_false_count')}",
        f"- positive 数量: {border_stats.get('positive_count')}",
        f"- near_positive 数量: {border_stats.get('near_positive_count')}",
        f"- hard_negative 数量: {border_stats.get('hard_negative_count')}",
        f"- negative 数量: {border_stats.get('negative_count')}",
        f"- avg_text_score: {border_stats.get('avg_text_score')}",
        f"- avg_image_score: {border_stats.get('avg_image_score')}",
        f"- avg_fused_score: {border_stats.get('avg_fused_score')}",
        f"- min_fused_score: {border_stats.get('min_fused_score')}",
        f"- max_fused_score: {border_stats.get('max_fused_score')}",
        "",
        "## 3. 原双阈值问题",
        "",
        f"当前推荐 weak={original.get('recommended_weak_threshold', 0.7)}，strong={original.get('recommended_strong_threshold', 0.75)}，但 review_rate={original.get('key_metrics', {}).get('review_rate', 0.0)}，说明确认区未覆盖实际边界样本。",
        "",
        "## 4. 阈值重扫结果",
        "",
        f"- recommended_weak_threshold: {recommendation.get('recommended_weak_threshold')}",
        f"- recommended_strong_threshold: {recommendation.get('recommended_strong_threshold')}",
        f"- auto_false_hit_rate: {recommendation.get('auto_false_hit_rate')}",
        f"- review_count: {recommendation.get('review_count')}",
        f"- review_true_candidate_count: {recommendation.get('review_true_candidate_count')}",
        f"- review_false_candidate_count: {recommendation.get('review_false_candidate_count')}",
        f"- recall_if_review_accepted: {recommendation.get('recall_if_review_accepted')}",
        f"- near_positive_review_count: {recommendation.get('near_positive_review_count')}",
        f"- hard_negative_review_count: {recommendation.get('hard_negative_review_count')}",
        "",
        f"推荐理由：{recommendation.get('recommended_reason')}",
        "",
        "## 5. 阶段结论",
        "",
    ]
    if int(recommendation.get("review_count", 0)) > 0 and as_float(recommendation.get("auto_false_hit_rate")) <= 0.05:
        lines.append("本轮重扫找到了 review_count > 0 且 auto_false_hit_rate <= 0.05 的组合，说明双阈值机制仍然有价值，只是当前推荐区间需要调整。")
    else:
        lines.append("本轮重扫中 review_count 仍然较低，说明样本分布仍偏两极化，需要继续补充更接近 weak/strong 阈值之间的边界样本。")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze v3_real_70 borderline cases and rescan dual thresholds.")
    parser.add_argument("--summary-csv", default=str(DEFAULT_SUMMARY_CSV))
    parser.add_argument("--summary-json", default=str(DEFAULT_SUMMARY_JSON))
    parser.add_argument("--borderline-csv", default=str(DEFAULT_BORDERLINE_CSV))
    parser.add_argument("--weighted-dual-json", default=str(DEFAULT_WEIGHTED_DUAL_JSON))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args()

    summary_rows = read_csv(Path(args.summary_csv))
    borderline_rows = read_csv(Path(args.borderline_csv))
    original_dual = read_json(Path(args.weighted_dual_json))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    review_rows, border_stats = build_borderline_review(borderline_rows, summary_rows)
    threshold_rows = [
        evaluate_pair(summary_rows, weak, strong)
        for weak in WEAK_THRESHOLDS
        for strong in STRONG_THRESHOLDS
        if weak < strong
    ]
    recommendation = recommend(threshold_rows)

    review_fields = [
        "image",
        "sample_type",
        "should_hit",
        "category",
        "text_score",
        "image_score",
        "fused_score",
        "best_keyword",
        "best_filename",
        "suggested_action",
    ]
    threshold_fields = list(threshold_rows[0].keys()) if threshold_rows else []
    write_csv(output_dir / "borderline_case_review.csv", review_rows, review_fields)
    write_csv(output_dir / "threshold_rescan_summary.csv", threshold_rows, threshold_fields)
    (output_dir / "recommended_review_threshold.json").write_text(
        json.dumps(recommendation, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    report_path = output_dir / "borderline_threshold_report.md"
    write_report(report_path, border_stats, original_dual, recommendation)

    print("=" * 72)
    print(f"borderline_count: {border_stats['borderline_count']}")
    print(f"borderline_should_hit_true_count: {border_stats['borderline_should_hit_true_count']}")
    print(f"borderline_should_hit_false_count: {border_stats['borderline_should_hit_false_count']}")
    print(f"recommended_weak_threshold: {recommendation['recommended_weak_threshold']}")
    print(f"recommended_strong_threshold: {recommendation['recommended_strong_threshold']}")
    print(f"auto_false_hit_rate: {recommendation['auto_false_hit_rate']}")
    print(f"review_count: {recommendation['review_count']}")
    print(f"review_true_candidate_count: {recommendation['review_true_candidate_count']}")
    print(f"review_false_candidate_count: {recommendation['review_false_candidate_count']}")
    print(f"recall_if_review_accepted: {recommendation['recall_if_review_accepted']}")
    print(f"borderline_threshold_report.md: {report_path}")
    print("=" * 72)


if __name__ == "__main__":
    main()
