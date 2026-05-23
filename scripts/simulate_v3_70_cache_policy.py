from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List


DEFAULT_EVAL_DIR = Path("paper_repro_outputs/cache_similarity_eval_v3_real_70")
DEFAULT_SUMMARY_CSV = DEFAULT_EVAL_DIR / "summary.csv"
DEFAULT_REVIEW_AUDIT_CSV = DEFAULT_EVAL_DIR / "review_sample_audit" / "review_sample_audit.csv"
DEFAULT_OUTPUT_DIR = DEFAULT_EVAL_DIR / "policy_simulation"

POLICY = {
    "policy_name": "v3_real_70_rule_fusion_dual_threshold",
    "score_formula": "0.5 * text_score + 0.5 * image_score",
    "text_weight": 0.5,
    "image_weight": 0.5,
    "weak_threshold": 0.7,
    "strong_threshold": 0.78,
    "auto_hit_rule": "score >= 0.78",
    "review_rule": "0.7 <= score < 0.78",
    "miss_rule": "score < 0.7",
    "source_experiment": "v3_real_70",
    "manual_review_verified_count": 10,
    "notes": "该策略基于 v3_real_70 离线实验与 review 样本人工复核结果生成，尚未接入 plus.py。",
}


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


def policy_score(row: Dict[str, str], warnings: List[str]) -> float:
    text_score = as_float(row.get("text_score"), 0.0)
    image_raw = str(row.get("image_score") or "").strip()
    if not image_raw:
        warnings.append(f"image_score missing, fallback to text_score: {row.get('image', '')}")
        image_score = text_score
    else:
        image_score = as_float(image_raw, text_score)
    return 0.5 * text_score + 0.5 * image_score


def decide(score: float) -> str:
    if score >= 0.78:
        return "auto_hit"
    if score >= 0.7:
        return "review"
    return "miss"


def rate(numerator: int, denominator: int) -> float:
    return round4(numerator / denominator) if denominator else 0.0


def simulate(summary_csv: Path, review_audit_csv: Path, output_dir: Path) -> Dict[str, Any]:
    rows = read_csv(summary_csv)
    review_rows = read_csv(review_audit_csv)
    verified_review_count = sum(1 for row in review_rows if row.get("manual_decision") == "accept_review")

    policy = dict(POLICY)
    policy["manual_review_verified_count"] = verified_review_count

    warnings: List[str] = []
    result_rows: List[Dict[str, Any]] = []
    for row in rows:
        score = policy_score(row, warnings)
        decision = decide(score)
        should_hit = as_bool(row.get("should_hit"))
        sample_type = row.get("sample_type", "")
        is_correct_auto_hit = decision == "auto_hit" and should_hit
        is_false_auto_hit = decision == "auto_hit" and not should_hit
        is_review_true_candidate = decision == "review" and should_hit
        is_review_false_candidate = decision == "review" and not should_hit
        is_false_miss = decision == "miss" and should_hit

        result_rows.append(
            {
                "image": row.get("image", ""),
                "sample_type": sample_type,
                "should_hit": row.get("should_hit", ""),
                "category": row.get("category", ""),
                "query_text": row.get("query_text", ""),
                "text_score": row.get("text_score", ""),
                "image_score": row.get("image_score", ""),
                "policy_score": round4(score),
                "policy_decision": decision,
                "best_keyword": row.get("best_keyword", ""),
                "best_filename": row.get("best_filename", ""),
                "best_model_path": row.get("best_model_path", ""),
                "is_correct_auto_hit": is_correct_auto_hit,
                "is_false_auto_hit": is_false_auto_hit,
                "is_review_true_candidate": is_review_true_candidate,
                "is_review_false_candidate": is_review_false_candidate,
                "is_false_miss": is_false_miss,
            }
        )

    total = len(result_rows)
    should_hit_count = sum(1 for row in result_rows if as_bool(row["should_hit"]))
    should_not_hit_count = total - should_hit_count
    auto_hit_rows = [row for row in result_rows if row["policy_decision"] == "auto_hit"]
    review_rows_policy = [row for row in result_rows if row["policy_decision"] == "review"]
    miss_rows = [row for row in result_rows if row["policy_decision"] == "miss"]
    auto_true_hit_count = sum(1 for row in auto_hit_rows if as_bool(row["should_hit"]))
    auto_false_hit_count = len(auto_hit_rows) - auto_true_hit_count
    review_true_candidate_count = sum(1 for row in review_rows_policy if as_bool(row["should_hit"]))
    review_false_candidate_count = len(review_rows_policy) - review_true_candidate_count
    false_miss_count = sum(1 for row in miss_rows if as_bool(row["should_hit"]))

    def group_count(decision: str, sample_type: str, should_hit_value: bool | None = None) -> int:
        count = 0
        for row in result_rows:
            if row["policy_decision"] != decision or row["sample_type"] != sample_type:
                continue
            if should_hit_value is not None and as_bool(row["should_hit"]) != should_hit_value:
                continue
            count += 1
        return count

    summary = {
        "total_samples": total,
        "should_hit_count": should_hit_count,
        "should_not_hit_count": should_not_hit_count,
        "auto_hit_count": len(auto_hit_rows),
        "review_count": len(review_rows_policy),
        "miss_count": len(miss_rows),
        "auto_true_hit_count": auto_true_hit_count,
        "auto_false_hit_count": auto_false_hit_count,
        "auto_false_hit_rate": rate(auto_false_hit_count, len(auto_hit_rows)),
        "review_true_candidate_count": review_true_candidate_count,
        "review_false_candidate_count": review_false_candidate_count,
        "review_false_candidate_rate": rate(review_false_candidate_count, len(review_rows_policy)),
        "false_miss_count": false_miss_count,
        "false_miss_rate": rate(false_miss_count, should_hit_count),
        "recall_auto_only": rate(auto_true_hit_count, should_hit_count),
        "recall_if_review_accepted": rate(auto_true_hit_count + review_true_candidate_count, should_hit_count),
        "precision_auto_hit": rate(auto_true_hit_count, len(auto_hit_rows)),
        "near_positive_auto_hit_count": group_count("auto_hit", "near_positive", True),
        "near_positive_review_count": group_count("review", "near_positive", True),
        "near_positive_miss_count": group_count("miss", "near_positive", True),
        "hard_negative_auto_false_hit_count": group_count("auto_hit", "hard_negative", False),
        "hard_negative_review_count": group_count("review", "hard_negative"),
        "negative_auto_false_hit_count": group_count("auto_hit", "negative", False),
        "negative_review_count": group_count("review", "negative"),
        "warnings": warnings[:50],
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    policy_path = output_dir / "cache_policy_v3_real_70.json"
    results_path = output_dir / "policy_simulation_results.csv"
    summary_path = output_dir / "policy_simulation_summary.json"
    report_path = output_dir / "policy_simulation_report.md"

    policy_path.write_text(json.dumps(policy, ensure_ascii=False, indent=2), encoding="utf-8")
    fields = [
        "image",
        "sample_type",
        "should_hit",
        "category",
        "query_text",
        "text_score",
        "image_score",
        "policy_score",
        "policy_decision",
        "best_keyword",
        "best_filename",
        "best_model_path",
        "is_correct_auto_hit",
        "is_false_auto_hit",
        "is_review_true_candidate",
        "is_review_false_candidate",
        "is_false_miss",
    ]
    write_csv(results_path, result_rows, fields)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(report_path, summary, policy)

    return {
        "policy_path": policy_path,
        "results_path": results_path,
        "summary_path": summary_path,
        "report_path": report_path,
        "summary": summary,
    }


def write_report(path: Path, summary: Dict[str, Any], policy: Dict[str, Any]) -> None:
    safe_line = ""
    if summary["auto_false_hit_rate"] == 0 and summary["review_false_candidate_count"] == 0:
        safe_line = (
            "当前策略在离线模拟中保持自动误复用为 0，review 区间样本也经过人工确认，"
            "适合作为下一阶段小范围接入候选策略。"
        )
    else:
        safe_line = "当前策略仍存在误复用或 review 负样本风险，接入前需要继续复核。"

    text = f"""# v3_real_70 缓存复用策略离线模拟报告

## 1. 策略来源

该策略来自 v3_real_70 实验、边界样本重扫和 10 个 review 样本人工复核。当前仅作为离线配置和模拟结果，尚未接入 `plus.py`。

## 2. 策略规则

score = 0.5 * text_score + 0.5 * image_score

- score >= 0.78：自动复用
- 0.7 <= score < 0.78：用户确认
- score < 0.7：重新生成

## 3. 离线模拟结果

- total_samples: {summary['total_samples']}
- auto_hit_count: {summary['auto_hit_count']}
- review_count: {summary['review_count']}
- miss_count: {summary['miss_count']}
- auto_false_hit_rate: {summary['auto_false_hit_rate']}
- review_true_candidate_count: {summary['review_true_candidate_count']}
- review_false_candidate_count: {summary['review_false_candidate_count']}
- false_miss_count: {summary['false_miss_count']}
- recall_auto_only: {summary['recall_auto_only']}
- recall_if_review_accepted: {summary['recall_if_review_accepted']}

## 4. 人工复核支撑

10 个 review 样本已人工复核，`manual_decision=accept_review`。

## 5. 接入前结论

{safe_line}

## 6. 接入风险

- 当前样本量仍为 70；
- 还未在真实运行链路中验证；
- review 前端交互尚未接入；
- `plus.py` 暂未修改；
- 后续应先做小范围配置化接入，而不是直接写死阈值。
"""
    path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Simulate v3_real_70 cache policy offline.")
    parser.add_argument("--summary-csv", default=str(DEFAULT_SUMMARY_CSV))
    parser.add_argument("--review-audit-csv", default=str(DEFAULT_REVIEW_AUDIT_CSV))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args()

    result = simulate(Path(args.summary_csv), Path(args.review_audit_csv), Path(args.output_dir))
    summary = result["summary"]
    print("=" * 72)
    print(f"cache_policy_v3_real_70.json: {result['policy_path']}")
    print(f"policy_simulation_results.csv: {result['results_path']}")
    print(f"policy_simulation_summary.json: {result['summary_path']}")
    print(f"policy_simulation_report.md: {result['report_path']}")
    print(f"auto_hit_count: {summary['auto_hit_count']}")
    print(f"review_count: {summary['review_count']}")
    print(f"auto_false_hit_rate: {summary['auto_false_hit_rate']}")
    print(f"review_false_candidate_count: {summary['review_false_candidate_count']}")
    print(f"recall_if_review_accepted: {summary['recall_if_review_accepted']}")
    print("=" * 72)


if __name__ == "__main__":
    main()
