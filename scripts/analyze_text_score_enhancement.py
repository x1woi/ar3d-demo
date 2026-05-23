import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple


DEFAULT_SUMMARY_CSV = Path("paper_repro_outputs/cache_similarity_eval_v3_real/summary.csv")
DEFAULT_SUMMARY_JSON = Path("paper_repro_outputs/cache_similarity_eval_v3_real/summary.json")
DEFAULT_OUTPUT_DIR = Path("paper_repro_outputs/cache_similarity_eval_v3_real/text_score_enhancement")
THRESHOLDS = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]
SAMPLE_TYPES = ["positive", "near_positive", "hard_negative", "negative"]

HEAD_GROUP = {"人头", "人脸", "头部", "脸部", "face", "head", "rentou", "renlian"}
GLASSES_GROUP = {"眼镜", "镜片", "镜框", "glasses", "eyeglasses", "yanjing"}
NEGATIVE_GUARD_WORDS = {
    "不应复用",
    "不应该复用",
    "无关",
    "背景",
    "非目标",
    "困难负样本",
    "遮挡",
    "mask",
    "键盘",
    "书",
    "桌面",
    "杯子",
}


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


def read_rows(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"summary.csv not found: {path}")

    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        required = {"image", "sample_type", "should_hit", "text_score", "image_score", "fused_score"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"summary.csv missing required columns: {sorted(missing)}")

        rows: List[Dict[str, Any]] = []
        for row in reader:
            text_score = parse_float(row.get("text_score"))
            image_score_missing = str(row.get("image_score", "")).strip() == ""
            image_score = parse_float(row.get("image_score"), text_score)
            row["query_text"] = row.get("query_text", "")
            row["should_hit_bool"] = parse_bool(row.get("should_hit"))
            row["text_score_float"] = text_score
            row["image_score_float"] = image_score
            row["image_score_missing"] = image_score_missing
            row["fused_score_float"] = parse_float(row.get("fused_score"))
            rows.append(row)
        return rows


def contains_any(text: str, words: set[str]) -> bool:
    lowered = text.lower()
    return any(word.lower() in lowered for word in words)


def group_for_text(text: str) -> str | None:
    lowered = text.lower()
    if any(word.lower() in lowered for word in HEAD_GROUP):
        return "head"
    if any(word.lower() in lowered for word in GLASSES_GROUP):
        return "glasses"
    return None


def same_synonym_group(row: Dict[str, Any]) -> bool:
    query_side = " ".join(
        [
            str(row.get("category", "")),
            str(row.get("query_text", "")),
            str(row.get("image", "")),
        ]
    )
    best_side = " ".join([str(row.get("best_keyword", "")), str(row.get("best_filename", ""))])
    query_group = group_for_text(query_side)
    best_group = group_for_text(best_side)
    return query_group is not None and query_group == best_group


def has_negative_guard(row: Dict[str, Any]) -> bool:
    text = " ".join([str(row.get("category", "")), str(row.get("query_text", ""))])
    return contains_any(text, NEGATIVE_GUARD_WORDS)


def enhanced_text_score(row: Dict[str, Any], strategy: str) -> float:
    original = row["text_score_float"]
    if strategy == "original":
        return original
    if strategy == "synonym_floor_0.6":
        return max(original, 0.6) if same_synonym_group(row) else original
    if strategy == "synonym_floor_0.8":
        return max(original, 0.8) if same_synonym_group(row) else original
    if strategy == "negative_guard_synonym_0.8":
        if has_negative_guard(row):
            return original
        return max(original, 0.8) if same_synonym_group(row) else original
    raise ValueError(f"unknown strategy: {strategy}")


def enhanced_fused_score(row: Dict[str, Any], strategy: str) -> Tuple[float, float]:
    text = enhanced_text_score(row, strategy)
    image = row["image_score_float"]
    if row["image_score_missing"]:
        image = text
    if strategy == "original":
        return row["fused_score_float"], text
    return round(0.5 * text + 0.5 * image, 4), text


def group_metrics(rows: List[Dict[str, Any]], hits: List[bool]) -> Dict[str, Dict[str, Any]]:
    grouped: Dict[str, List[Tuple[Dict[str, Any], bool]]] = defaultdict(list)
    for row, hit in zip(rows, hits):
        grouped[str(row.get("sample_type", "unknown"))].append((row, hit))

    metrics: Dict[str, Dict[str, Any]] = {}
    for sample_type in SAMPLE_TYPES:
        items = grouped.get(sample_type, [])
        count = len(items)
        expected_hit = sum(1 for row, _hit in items if row["should_hit_bool"])
        expected_miss = count - expected_hit
        hit_count = sum(1 for _row, hit in items if hit)
        miss_count = count - hit_count
        false_hit = sum(1 for row, hit in items if hit and not row["should_hit_bool"])
        false_miss = sum(1 for row, hit in items if (not hit) and row["should_hit_bool"])
        metrics[sample_type] = {
            "count": count,
            "hit_count": hit_count,
            "miss_count": miss_count,
            "false_hit_rate": round(false_hit / expected_miss, 4) if expected_miss else 0.0,
            "false_miss_rate": round(false_miss / expected_hit, 4) if expected_hit else 0.0,
        }
    return metrics


def evaluate(rows: List[Dict[str, Any]], strategy: str, threshold: float) -> Dict[str, Any]:
    scores = [enhanced_fused_score(row, strategy)[0] for row in rows]
    hits = [score >= threshold for score in scores]
    tp = sum(1 for row, hit in zip(rows, hits) if hit and row["should_hit_bool"])
    tn = sum(1 for row, hit in zip(rows, hits) if (not hit) and not row["should_hit_bool"])
    fp = sum(1 for row, hit in zip(rows, hits) if hit and not row["should_hit_bool"])
    fn = sum(1 for row, hit in zip(rows, hits) if (not hit) and row["should_hit_bool"])
    total = len(rows)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    expected_miss = tn + fp
    expected_hit = tp + fn
    groups = group_metrics(rows, hits)

    result: Dict[str, Any] = {
        "strategy": strategy,
        "threshold": threshold,
        "total_samples": total,
        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "accuracy": round((tp + tn) / total, 4) if total else 0.0,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "false_hit_rate": round(fp / expected_miss, 4) if expected_miss else 0.0,
        "false_miss_rate": round(fn / expected_hit, 4) if expected_hit else 0.0,
        "cost_score": round(tp * 1.0 + tn * 0.5 - fp * 3.0 - fn * 1.0, 4),
    }
    for sample_type in SAMPLE_TYPES:
        for key, value in groups[sample_type].items():
            result[f"{sample_type}_{key}"] = value
    return result


def recommend(summary_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    safe = [row for row in summary_rows if row["false_hit_rate"] <= 0.05 and row["recall"] > 0]
    pool = safe or [row for row in summary_rows if row["recall"] > 0] or summary_rows
    best = max(
        pool,
        key=lambda row: (
            -row["false_hit_rate"],
            row["recall"],
            -row.get("near_positive_false_miss_rate", 1.0),
            row["cost_score"],
            row["precision"],
        ),
    )
    reason = (
        f"Selected {best['strategy']} at threshold {best['threshold']} because false_hit_rate="
        f"{best['false_hit_rate']}, recall={best['recall']}, "
        f"near_positive_false_miss_rate={best.get('near_positive_false_miss_rate')}, "
        f"and cost_score={best['cost_score']}."
    )
    if best["strategy"] != "original" and best["false_hit_rate"] > 0:
        reason += " Recall improved but false_hit_rate increased, so this strategy has reuse risk."
    return {
        "recommended_strategy": best["strategy"],
        "recommended_threshold": best["threshold"],
        "recommended_reason": reason,
        "key_metrics": best,
    }


def write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def make_cases(rows: List[Dict[str, Any]], strategy: str, threshold: float, original_threshold: float) -> List[Dict[str, Any]]:
    cases: List[Dict[str, Any]] = []
    for row in rows:
        enhanced_score, enhanced_text = enhanced_fused_score(row, strategy)
        original_hit = row["fused_score_float"] >= original_threshold
        enhanced_hit = enhanced_score >= threshold
        changed = original_hit != enhanced_hit
        if row["should_hit_bool"] and (not original_hit) and enhanced_hit:
            change_type = "false_miss_fixed"
        elif (not row["should_hit_bool"]) and (not original_hit) and enhanced_hit:
            change_type = "false_hit_introduced"
        elif not changed:
            change_type = "unchanged"
        else:
            change_type = "other"
        cases.append(
            {
                "image": row.get("image", ""),
                "sample_type": row.get("sample_type", ""),
                "should_hit": row.get("should_hit", ""),
                "category": row.get("category", ""),
                "query_text": row.get("query_text", ""),
                "best_keyword": row.get("best_keyword", ""),
                "original_text_score": row.get("text_score", ""),
                "enhanced_text_score": enhanced_text,
                "image_score": row.get("image_score", ""),
                "original_fused_score": row.get("fused_score", ""),
                "enhanced_fused_score": enhanced_score,
                "original_hit": original_hit,
                "enhanced_hit": enhanced_hit,
                "changed": changed,
                "change_type": change_type,
            }
        )
    return cases


def write_report(
    path: Path,
    best_by_strategy: List[Dict[str, Any]],
    recommendation: Dict[str, Any],
    original_best: Dict[str, Any],
) -> None:
    key = recommendation["key_metrics"]
    lines = [
        "# v3_real 文本相似度增强离线实验",
        "",
        "## 1. 实验目的",
        "",
        "当前 v3_real 的主要问题是 false miss 偏多。错误分析显示，多数漏命中来自文本相似度不足，尤其是部分 positive / near_positive 样本的 text_score 为 0，但 image_score 并不低。",
        "",
        "## 2. 增强方法",
        "",
        "本实验比较四种离线策略：original、synonym_floor_0.6、synonym_floor_0.8、negative_guard_synonym_0.8。增强方法基于同义词归一化，如果 category/query_text 与 best_keyword 命中同一语义组，则给 text_score 设置 floor。negative_guard 策略会在负向词出现时跳过增强。",
        "",
        "## 3. 策略对比",
        "",
        "| strategy | best_threshold | accuracy | precision | recall | false_hit_rate | false_miss_rate | positive_false_miss_rate | near_positive_false_miss_rate | hard_negative_false_hit_rate | cost_score |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in best_by_strategy:
        lines.append(
            f"| {row['strategy']} | {row['threshold']} | {row['accuracy']} | {row['precision']} | "
            f"{row['recall']} | {row['false_hit_rate']} | {row['false_miss_rate']} | "
            f"{row.get('positive_false_miss_rate')} | {row.get('near_positive_false_miss_rate')} | "
            f"{row.get('hard_negative_false_hit_rate')} | {row['cost_score']} |"
        )
    lines.extend(
        [
            "",
            "## 4. 推荐策略",
            "",
            f"- recommended_strategy: {recommendation['recommended_strategy']}",
            f"- recommended_threshold: {recommendation['recommended_threshold']}",
            f"- recommended_reason: {recommendation['recommended_reason']}",
            "",
            "## 5. 结果解释",
            "",
        ]
    )
    if key["recall"] > original_best["recall"] and key["false_hit_rate"] <= 0.05:
        lines.append("文本相似度增强可以缓解 text_score=0 导致的漏命中问题，同时保持误复用安全性。")
    elif key["false_hit_rate"] > original_best["false_hit_rate"]:
        lines.append("文本增强会带来误命中风险，需要加入更严格的 negative guard 或保持保守阈值。")
    else:
        lines.append("在当前 v3_real 数据中，同义词增强没有明显改善 recall，说明问题可能来自标签语义过泛或同义词规则覆盖不足。")
    lines.extend(
        [
            "",
            "## 6. 后续建议",
            "",
            "如果离线增强有效，下一步可以考虑把同义词归一化逻辑轻量接入 cache_similarity.py，而不是直接上 MLP。若当前增强无效，应先细化 category/query_text 和同义词组，再扩大样本验证。",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline text score enhancement experiment for v3_real.")
    parser.add_argument("--summary-csv", default=str(DEFAULT_SUMMARY_CSV))
    parser.add_argument("--summary-json", default=str(DEFAULT_SUMMARY_JSON))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args()

    rows = read_rows(Path(args.summary_csv))
    summary = read_json(Path(args.summary_json))
    original_threshold = float(summary.get("recommended_threshold") or 0.6)
    strategies = ["original", "synonym_floor_0.6", "synonym_floor_0.8", "negative_guard_synonym_0.8"]

    summary_rows: List[Dict[str, Any]] = []
    best_by_strategy: List[Dict[str, Any]] = []
    for strategy in strategies:
        strategy_rows = [evaluate(rows, strategy, threshold) for threshold in THRESHOLDS]
        summary_rows.extend(strategy_rows)
        safe = [row for row in strategy_rows if row["false_hit_rate"] <= 0.05 and row["recall"] > 0]
        pool = safe or [row for row in strategy_rows if row["recall"] > 0] or strategy_rows
        best_by_strategy.append(
            max(
                pool,
                key=lambda row: (
                    row["recall"],
                    -row.get("near_positive_false_miss_rate", 1.0),
                    row["cost_score"],
                    -row["false_hit_rate"],
                ),
            )
        )

    recommendation = recommend(summary_rows)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_fields = [
        "strategy",
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
    for sample_type in SAMPLE_TYPES:
        for key in ("count", "hit_count", "miss_count", "false_hit_rate", "false_miss_rate"):
            summary_fields.append(f"{sample_type}_{key}")

    summary_path = output_dir / "text_score_enhancement_summary.csv"
    cases_path = output_dir / "text_score_enhancement_cases.csv"
    recommendation_path = output_dir / "recommended_text_enhancement.json"
    report_path = output_dir / "text_score_enhancement_report.md"

    write_csv(summary_path, summary_rows, summary_fields)
    cases = make_cases(rows, recommendation["recommended_strategy"], recommendation["recommended_threshold"], original_threshold)
    case_fields = [
        "image",
        "sample_type",
        "should_hit",
        "category",
        "query_text",
        "best_keyword",
        "original_text_score",
        "enhanced_text_score",
        "image_score",
        "original_fused_score",
        "enhanced_fused_score",
        "original_hit",
        "enhanced_hit",
        "changed",
        "change_type",
    ]
    write_csv(cases_path, cases, case_fields)
    recommendation_path.write_text(json.dumps(recommendation, ensure_ascii=False, indent=2), encoding="utf-8")
    original_best = next(row for row in best_by_strategy if row["strategy"] == "original")
    write_report(report_path, best_by_strategy, recommendation, original_best)

    false_miss_fixed_count = sum(1 for row in cases if row["change_type"] == "false_miss_fixed")
    false_hit_introduced_count = sum(1 for row in cases if row["change_type"] == "false_hit_introduced")
    print("=" * 72)
    print(f"recommended_strategy: {recommendation['recommended_strategy']}")
    print(f"recommended_threshold: {recommendation['recommended_threshold']}")
    print(f"original_recall: {original_best['recall']}")
    print(f"enhanced_recall: {recommendation['key_metrics']['recall']}")
    print(f"original_false_hit_rate: {original_best['false_hit_rate']}")
    print(f"enhanced_false_hit_rate: {recommendation['key_metrics']['false_hit_rate']}")
    print(f"false_miss_fixed_count: {false_miss_fixed_count}")
    print(f"false_hit_introduced_count: {false_hit_introduced_count}")
    print(f"text_score_enhancement_report.md: {report_path}")
    print("=" * 72)


if __name__ == "__main__":
    main()
