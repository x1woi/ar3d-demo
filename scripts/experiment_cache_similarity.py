from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from cache_similarity import build_similarity_index, score_cache_entries


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass
class LabelItem:
    image: str
    category: str
    should_hit: bool
    query_text: str = ""
    sample_type: str = "default"


@dataclass
class SampleResult:
    image: str
    category: str
    sample_type: str
    should_hit: bool
    predicted_hit: bool
    correct: bool
    best_keyword: str
    best_filename: str
    best_model_path: str
    text_score: float
    image_score: Optional[float]
    fused_score: float
    decision: str
    elapsed_ms: int


@dataclass
class ThresholdMetrics:
    threshold: float
    total_samples: int
    tp: int
    tn: int
    fp: int
    fn: int
    hit_count: int
    miss_count: int
    correct_count: int
    accuracy: float
    precision: float
    recall: float
    f1: float
    false_hit_rate: float
    false_miss_rate: float
    avg_elapsed_ms: float
    cost_score: float
    grouped_metrics: Dict[str, Dict[str, Any]]


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    text = str(value).strip().lower()
    return text in {"1", "true", "yes", "y", "hit", "should_hit", "命中", "是"}


def load_labels(path: Path) -> List[LabelItem]:
    raw = read_json(path)
    items: List[LabelItem] = []

    if isinstance(raw, dict) and "samples" in raw:
        raw_items = raw["samples"]
    elif isinstance(raw, list):
        raw_items = raw
    elif isinstance(raw, dict):
        raw_items = []
        for image, item in raw.items():
            if isinstance(item, dict):
                merged = dict(item)
                merged.setdefault("image", image)
                raw_items.append(merged)
            else:
                raw_items.append(
                    {
                        "image": image,
                        "category": str(item),
                        "should_hit": True,
                    }
                )
    else:
        raise ValueError("labels.json must be a list, a dict, or a dict with samples")

    for item in raw_items:
        if not isinstance(item, dict):
            raise ValueError(f"Invalid label item: {item}")
        image = str(item.get("image") or item.get("filename") or item.get("query_image") or "").strip()
        category = str(item.get("category") or item.get("label") or item.get("class") or "").strip()
        if not image:
            raise ValueError(f"Label item has no image field: {item}")
        if not category:
            category = Path(image).stem
        should_hit = parse_bool(item.get("should_hit", item.get("expected_hit", item.get("hit", True))))
        query_text = str(item.get("query_text") or item.get("text") or category).strip()
        sample_type = str(item.get("sample_type") or item.get("type") or "default").strip() or "default"
        items.append(
            LabelItem(
                image=image,
                category=category,
                should_hit=should_hit,
                query_text=query_text,
                sample_type=sample_type,
            )
        )

    return items


def resolve_image(dataset_dir: Path, image_name: str) -> Path:
    path = Path(image_name)
    if path.is_absolute():
        return path
    direct = dataset_dir / path
    if direct.exists():
        return direct
    for ext in IMAGE_EXTS:
        candidate = dataset_dir / f"{image_name}{ext}"
        if candidate.exists():
            return candidate
    return direct


def parse_thresholds(text: str, default: float) -> List[float]:
    if not text.strip():
        return [default]
    values: List[float] = []
    for part in text.replace(";", ",").split(","):
        part = part.strip()
        if not part:
            continue
        values.append(float(part))
    return sorted(set(values))


def compute_metrics(samples: List[SampleResult], threshold: float) -> ThresholdMetrics:
    total = len(samples)
    tp = tn = fp = fn = 0

    for item in samples:
        predicted_hit = item.fused_score >= threshold
        if predicted_hit and item.should_hit:
            tp += 1
        elif predicted_hit and not item.should_hit:
            fp += 1
        elif not predicted_hit and item.should_hit:
            fn += 1
        else:
            tn += 1

    hit_count = tp + fp
    miss_count = tn + fn
    correct_count = tp + tn
    expected_hit_count = tp + fn
    expected_miss_count = tn + fp

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    avg_elapsed_ms = (
        sum(item.elapsed_ms for item in samples) / total
        if total
        else 0.0
    )

    grouped_metrics = compute_grouped_metrics(samples, threshold)

    return ThresholdMetrics(
        threshold=threshold,
        total_samples=total,
        tp=tp,
        tn=tn,
        fp=fp,
        fn=fn,
        hit_count=hit_count,
        miss_count=miss_count,
        correct_count=correct_count,
        accuracy=round(correct_count / total, 4) if total else 0.0,
        precision=round(precision, 4),
        recall=round(recall, 4),
        f1=round(f1, 4),
        false_hit_rate=round(fp / expected_miss_count, 4) if expected_miss_count else 0.0,
        false_miss_rate=round(fn / expected_hit_count, 4) if expected_hit_count else 0.0,
        avg_elapsed_ms=round(avg_elapsed_ms, 3),
        cost_score=round(tp * 1.0 + tn * 0.5 - fp * 3.0 - fn * 1.0, 4),
        grouped_metrics=grouped_metrics,
    )


def compute_grouped_metrics(samples: List[SampleResult], threshold: float) -> Dict[str, Dict[str, Any]]:
    groups = sorted({item.sample_type or "default" for item in samples})
    grouped: Dict[str, Dict[str, Any]] = {}

    for group in groups:
        group_samples = [item for item in samples if (item.sample_type or "default") == group]
        count = len(group_samples)
        hit_count = sum(1 for item in group_samples if item.fused_score >= threshold)
        miss_count = count - hit_count
        correct_count = sum(
            1
            for item in group_samples
            if (item.fused_score >= threshold) == item.should_hit
        )
        expected_hit_count = sum(1 for item in group_samples if item.should_hit)
        expected_miss_count = count - expected_hit_count
        false_hit_count = sum(
            1
            for item in group_samples
            if item.fused_score >= threshold and not item.should_hit
        )
        false_miss_count = sum(
            1
            for item in group_samples
            if item.fused_score < threshold and item.should_hit
        )

        grouped[group] = {
            "count": count,
            "hit_count": hit_count,
            "miss_count": miss_count,
            "accuracy": round(correct_count / count, 4) if count else 0.0,
            "false_hit_rate": round(false_hit_count / expected_miss_count, 4) if expected_miss_count else 0.0,
            "false_miss_rate": round(false_miss_count / expected_hit_count, 4) if expected_hit_count else 0.0,
        }

    return grouped


def recommend_threshold(metrics: List[ThresholdMetrics]) -> Dict[str, Any]:
    if not metrics:
        return {
            "recommended_threshold": None,
            "recommended_reason": "no threshold metrics available",
        }

    safe = [
        item
        for item in metrics
        if item.false_hit_rate <= 0.05
        and item.recall > 0
        and item.grouped_metrics.get("positive", {}).get("false_miss_rate", 0.0) < 1.0
    ]
    if safe:
        best = max(safe, key=lambda item: (item.cost_score, item.recall, item.f1, -item.threshold))
        return {
            "recommended_threshold": best.threshold,
            "recommended_reason": (
                f"Selected threshold {best.threshold} because it satisfies "
                f"false_hit_rate <= 0.05 (actual {best.false_hit_rate}), recall > 0 "
                f"(actual {best.recall}), and positive_false_miss_rate < 1.0; among "
                f"those candidates it has the highest cost_score ({best.cost_score})."
            ),
        }

    nonzero_recall = [item for item in metrics if item.recall > 0]
    if nonzero_recall:
        best = min(nonzero_recall, key=lambda item: (item.false_hit_rate, -item.cost_score, -item.recall))
        return {
            "recommended_threshold": best.threshold,
            "recommended_reason": (
                f"No threshold satisfied false_hit_rate <= 0.05 with recall > 0 and "
                f"positive_false_miss_rate < 1.0. Selected threshold {best.threshold} "
                f"because recall is greater than 0 (actual {best.recall}) and it has "
                f"the lowest false_hit_rate ({best.false_hit_rate}); ties are broken "
                f"by cost_score ({best.cost_score})."
            ),
        }

    best = min(metrics, key=lambda item: (item.false_hit_rate, -item.cost_score))
    return {
        "recommended_threshold": best.threshold,
        "recommended_reason": (
            "All candidate thresholds have recall equal to 0. The current dataset cannot "
            "produce an effective cache reuse threshold; add positive / near_positive "
            "samples or evaluate lower thresholds. Selected the threshold with the lowest "
            f"false_hit_rate ({best.false_hit_rate}) only as a fallback."
        ),
    }


def evaluate_dataset(
    dataset_dir: Path,
    cache_dir: Path,
    output_dir: Path,
    labels_path: Optional[Path] = None,
    threshold: float = 0.82,
    thresholds: Optional[List[float]] = None,
    text_weight: float = 0.45,
    image_weight: float = 0.55,
) -> Dict[str, Any]:
    dataset_dir = Path(dataset_dir)
    cache_dir = Path(cache_dir)
    output_dir = Path(output_dir)
    labels_path = Path(labels_path) if labels_path else dataset_dir / "labels.json"

    labels = load_labels(labels_path)
    entries = build_similarity_index(
        cache_dir=cache_dir,
        reference_dir=cache_dir / "reference_images",
        output_path=cache_dir / "cache_similarity_index.json",
    )

    sample_results: List[SampleResult] = []
    eval_thresholds = thresholds or [threshold]
    scoring_threshold = min(eval_thresholds) if eval_thresholds else threshold

    for item in labels:
        query_image = resolve_image(dataset_dir, item.image)
        start = time.perf_counter()
        results = score_cache_entries(
            entries,
            query_text=item.query_text,
            query_image=query_image,
            text_weight=text_weight,
            image_weight=image_weight,
            threshold=scoring_threshold,
        )
        elapsed_ms = int(round((time.perf_counter() - start) * 1000))

        best = results[0] if results else None
        predicted_hit = bool(best and best.fused_score >= threshold)
        correct = predicted_hit == item.should_hit

        sample_results.append(
            SampleResult(
                image=str(query_image),
                category=item.category,
                sample_type=item.sample_type,
                should_hit=item.should_hit,
                predicted_hit=predicted_hit,
                correct=correct,
                best_keyword=best.keyword if best else "",
                best_filename=best.filename if best else "",
                best_model_path=best.model_path if best else "",
                text_score=best.text_score if best else 0.0,
                image_score=best.image_score if best else None,
                fused_score=best.fused_score if best else 0.0,
                decision="hit" if predicted_hit else "miss",
                elapsed_ms=elapsed_ms,
            )
        )

    threshold_metrics = [compute_metrics(sample_results, item) for item in eval_thresholds]
    base_metrics = compute_metrics(sample_results, threshold)
    recommendation = recommend_threshold(threshold_metrics)

    summary = {
        "schema": "ar_cache_similarity_eval.v1",
        "dataset_dir": str(dataset_dir),
        "cache_dir": str(cache_dir),
        "labels_path": str(labels_path),
        "threshold": threshold,
        "thresholds": eval_thresholds,
        "text_weight": text_weight,
        "image_weight": image_weight,
        "recommended_threshold": recommendation["recommended_threshold"],
        "recommended_reason": recommendation["recommended_reason"],
        "total_samples": base_metrics.total_samples,
        "hit_count": base_metrics.hit_count,
        "miss_count": base_metrics.miss_count,
        "correct_count": base_metrics.correct_count,
        "accuracy": base_metrics.accuracy,
        "precision": base_metrics.precision,
        "recall": base_metrics.recall,
        "f1": base_metrics.f1,
        "hit_rate": round(base_metrics.hit_count / base_metrics.total_samples, 4) if base_metrics.total_samples else 0.0,
        "false_hit_rate": base_metrics.false_hit_rate,
        "false_miss_rate": base_metrics.false_miss_rate,
        "avg_elapsed_ms": base_metrics.avg_elapsed_ms,
        "cost_score": base_metrics.cost_score,
        "grouped_metrics": base_metrics.grouped_metrics,
        "threshold_summary": [asdict(item) for item in threshold_metrics],
        "samples": [asdict(item) for item in sample_results],
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "summary.json", summary)

    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8-sig") as f:
        fieldnames = [
            "image",
            "category",
            "sample_type",
            "should_hit",
            "predicted_hit",
            "correct",
            "best_keyword",
            "best_filename",
            "text_score",
            "image_score",
            "fused_score",
            "decision",
            "elapsed_ms",
            "best_model_path",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for item in sample_results:
            row = asdict(item)
            writer.writerow({key: row.get(key, "") for key in fieldnames})

    with (output_dir / "threshold_summary.csv").open("w", newline="", encoding="utf-8-sig") as f:
        fieldnames = [
            "threshold",
            "total_samples",
            "tp",
            "tn",
            "fp",
            "fn",
            "hit_count",
            "miss_count",
            "correct_count",
            "accuracy",
            "precision",
            "recall",
            "f1",
            "false_hit_rate",
            "false_miss_rate",
            "avg_elapsed_ms",
            "cost_score",
        ]
        groups = sorted({group for item in threshold_metrics for group in item.grouped_metrics})
        for group in groups:
            for metric_name in (
                "count",
                "hit_count",
                "miss_count",
                "accuracy",
                "false_hit_rate",
                "false_miss_rate",
            ):
                fieldnames.append(f"{group}_{metric_name}")
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for item in threshold_metrics:
            row = asdict(item)
            grouped = row.pop("grouped_metrics", {})
            for group in groups:
                metrics = grouped.get(group, {})
                for metric_name in (
                    "count",
                    "hit_count",
                    "miss_count",
                    "accuracy",
                    "false_hit_rate",
                    "false_miss_rate",
                ):
                    row[f"{group}_{metric_name}"] = metrics.get(metric_name, "")
            writer.writerow({key: row.get(key, "") for key in fieldnames})

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate AR cache hit decisions with text/image/fused similarity. No TripoSR calls."
    )
    parser.add_argument("--dataset-dir", type=str, required=True)
    parser.add_argument("--labels", type=str, default="")
    parser.add_argument("--cache-dir", type=str, default="runtime_assets/model_cache")
    parser.add_argument("--output-dir", type=str, default="paper_repro_outputs/cache_similarity_eval")
    parser.add_argument("--threshold", type=float, default=0.82)
    parser.add_argument("--thresholds", type=str, default="")
    parser.add_argument("--text-weight", type=float, default=0.45)
    parser.add_argument("--image-weight", type=float, default=0.55)

    args = parser.parse_args()
    summary = evaluate_dataset(
        dataset_dir=Path(args.dataset_dir),
        cache_dir=Path(args.cache_dir),
        output_dir=Path(args.output_dir),
        labels_path=Path(args.labels) if args.labels else None,
        threshold=args.threshold,
        thresholds=parse_thresholds(args.thresholds, args.threshold),
        text_weight=args.text_weight,
        image_weight=args.image_weight,
    )

    print("=" * 72)
    print("Cache similarity evaluation completed")
    print(f"Total samples: {summary['total_samples']}")
    print(f"Hit count: {summary['hit_count']}")
    print(f"Miss count: {summary['miss_count']}")
    print(f"Correct count: {summary['correct_count']}")
    print(f"Hit rate: {summary['hit_rate']}")
    print(f"False hit rate: {summary['false_hit_rate']}")
    print(f"False miss rate: {summary['false_miss_rate']}")
    print(f"Average elapsed ms: {summary['avg_elapsed_ms']}")
    print(f"Recommended threshold: {summary['recommended_threshold']}")
    print(f"Reason: {summary['recommended_reason']}")
    print(f"Output dir: {args.output_dir}")
    print("=" * 72)


if __name__ == "__main__":
    main()
