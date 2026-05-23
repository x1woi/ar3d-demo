#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Conservative weak/strong threshold sweep on real_eval_200.

No model training. No plus.py modification. Uses only existing real_eval_200
score features.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List


INPUT_CSV = Path("paper_repro_outputs/cache_similarity_eval_real_eval_200/real_eval_200_features.csv")
OUTPUT_DIR = Path("paper_repro_outputs/cache_similarity_eval_real_eval_200/threshold_sweep_conservative")
SWEEP_CSV = OUTPUT_DIR / "conservative_threshold_sweep.csv"
BEST_JSON = OUTPUT_DIR / "best_conservative_threshold.json"
REPORT_MD = OUTPUT_DIR / "conservative_threshold_sweep_report.md"

FIELDS = [
    "weak_threshold",
    "strong_threshold",
    "total_samples",
    "accuracy",
    "precision",
    "recall",
    "false_hit_rate",
    "false_miss_rate",
    "review_rate",
    "auto_hit_count",
    "review_count",
    "miss_count",
    "false_hit_count",
    "false_miss_count",
    "true_auto_hit_count",
    "review_true_candidate_count",
    "review_false_candidate_count",
]


def read_rows(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        out = []
        for row in csv.DictReader(f):
            text_score = float(row.get("text_score") or 0)
            image_score = float(row.get("image_score") or text_score)
            score = 0.5 * text_score + 0.5 * image_score
            out.append(
                {
                    "should_hit": 1 if str(row.get("should_hit")).strip() in {"1", "true", "True"} else 0,
                    "score": score,
                }
            )
        return out


def evaluate(rows: List[Dict[str, Any]], weak: float, strong: float) -> Dict[str, Any]:
    total = len(rows)
    positives = sum(1 for row in rows if row["should_hit"] == 1)
    negatives = total - positives
    auto_hit_count = review_count = miss_count = 0
    false_hit_count = false_miss_count = 0
    true_auto_hit_count = true_miss_count = 0
    review_true_candidate_count = review_false_candidate_count = 0

    for row in rows:
        should_hit = int(row["should_hit"])
        score = float(row["score"])
        if score >= strong:
            auto_hit_count += 1
            if should_hit:
                true_auto_hit_count += 1
            else:
                false_hit_count += 1
        elif score >= weak:
            review_count += 1
            if should_hit:
                review_true_candidate_count += 1
            else:
                review_false_candidate_count += 1
        else:
            miss_count += 1
            if should_hit:
                false_miss_count += 1
            else:
                true_miss_count += 1

    precision = true_auto_hit_count / auto_hit_count if auto_hit_count else 0.0
    recall = (true_auto_hit_count + review_true_candidate_count) / positives if positives else 0.0
    false_hit_rate = false_hit_count / negatives if negatives else 0.0
    false_miss_rate = false_miss_count / positives if positives else 0.0
    # Review is unresolved, so accuracy remains conservative.
    accuracy = (true_auto_hit_count + true_miss_count) / total if total else 0.0
    return {
        "weak_threshold": round(weak, 2),
        "strong_threshold": round(strong, 2),
        "total_samples": total,
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "false_hit_rate": round(false_hit_rate, 4),
        "false_miss_rate": round(false_miss_rate, 4),
        "review_rate": round(review_count / total, 4) if total else 0.0,
        "auto_hit_count": auto_hit_count,
        "review_count": review_count,
        "miss_count": miss_count,
        "false_hit_count": false_hit_count,
        "false_miss_count": false_miss_count,
        "true_auto_hit_count": true_auto_hit_count,
        "review_true_candidate_count": review_true_candidate_count,
        "review_false_candidate_count": review_false_candidate_count,
    }


def threshold_values(start: float, stop: float, step: float) -> List[float]:
    count = int(round((stop - start) / step)) + 1
    return [round(start + idx * step, 2) for idx in range(count)]


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows([{field: row.get(field, "") for field in FIELDS} for row in rows])


def make_report(rows: List[Dict[str, Any]], best: Dict[str, Any], baseline_060: Dict[str, Any], old_dual: Dict[str, Any]) -> str:
    top = sorted(
        [row for row in rows if row["false_hit_rate"] == 0.0],
        key=lambda row: (-row["recall"], row["review_rate"], row["strong_threshold"]),
    )[:10]
    table_lines = [
        "| weak | strong | recall | false_hit_rate | review_rate | auto_hit | review | miss |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in top:
        table_lines.append(
            f"| {row['weak_threshold']} | {row['strong_threshold']} | {row['recall']} | "
            f"{row['false_hit_rate']} | {row['review_rate']} | {row['auto_hit_count']} | "
            f"{row['review_count']} | {row['miss_count']} |"
        )

    text = f"""# real_eval_200 保守双阈值扫描报告

## 1. 实验目的

本实验基于 `real_eval_200_features.csv`，只扫描规则融合分数的 weak_threshold / strong_threshold，不训练新模型，不修改 plus.py。目标是在 `false_hit_rate = 0` 的前提下寻找 recall 最高的双阈值组合。

## 2. 方法

```text
score = 0.5 * text_score + 0.5 * image_score

score >= strong_threshold: auto_hit
weak_threshold <= score < strong_threshold: review
score < weak_threshold: miss
```

扫描范围：

- weak_threshold: 0.55 到 0.80，步长 0.01
- strong_threshold: 0.70 到 0.95，步长 0.01
- weak_threshold < strong_threshold

## 3. fixed@0.60 baseline

- threshold = 0.60
- false_hit_rate = {baseline_060['false_hit_rate']}
- recall = {baseline_060['recall']}
- false_hit_count = {baseline_060['false_hit_count']}

fixed@0.60 在 real_eval_200 上误复用偏高，不适合作为自动复用阈值。

## 4. 原双阈值

- weak = 0.70
- strong = 0.78
- false_hit_rate = {old_dual['false_hit_rate']}
- recall = {old_dual['recall']}
- review_rate = {old_dual['review_rate']}
- auto_hit_count = {old_dual['auto_hit_count']}
- review_count = {old_dual['review_count']}

双阈值的意义是把边界样本放入 review，而不是盲目降低自动复用阈值。

## 5. 最佳保守阈值

- recommended_weak_threshold = {best['weak_threshold']}
- recommended_strong_threshold = {best['strong_threshold']}
- false_hit_rate = {best['false_hit_rate']}
- recall = {best['recall']}
- review_rate = {best['review_rate']}
- auto_hit_count = {best['auto_hit_count']}
- review_count = {best['review_count']}
- miss_count = {best['miss_count']}
- false_hit_count = {best['false_hit_count']}
- false_miss_count = {best['false_miss_count']}

如果该组合相比 weak=0.70,strong=0.78 更稳，可以作为后续候选参数。本报告只给出推荐参数，不直接修改 plus.py。

## 6. false_hit_rate=0 的前 10 个候选

{chr(10).join(table_lines)}

## 7. 当前结论

保守阈值扫描再次说明，缓存复用任务不能只看 recall。自动复用必须优先保证 false_hit_rate 接近 0；边界样本应交给 review 分支承接。
"""
    REPORT_MD.write_text(text, encoding="utf-8-sig")
    return text


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = read_rows(INPUT_CSV)
    results: List[Dict[str, Any]] = []
    for weak in threshold_values(0.55, 0.80, 0.01):
        for strong in threshold_values(0.70, 0.95, 0.01):
            if weak >= strong:
                continue
            results.append(evaluate(rows, weak, strong))

    write_csv(SWEEP_CSV, results)
    candidates = [row for row in results if row["false_hit_rate"] == 0.0]
    if not candidates:
        raise RuntimeError("No false_hit_rate=0 threshold candidate found.")
    best = sorted(
        candidates,
        key=lambda row: (-row["recall"], row["review_rate"], row["strong_threshold"]),
    )[0]

    baseline_060 = evaluate(rows, 0.60, 0.60)
    old_dual = evaluate(rows, 0.70, 0.78)
    payload = {
        "selection_rule": [
            "false_hit_rate must be 0",
            "maximize recall",
            "tie-break lower review_rate",
            "tie-break lower strong_threshold",
        ],
        "best": best,
        "fixed_threshold_0_60": baseline_060,
        "old_dual_threshold": old_dual,
        "total_candidates": len(results),
        "zero_false_hit_candidates": len(candidates),
        "recommended_for_plus_py_change": False,
    }
    BEST_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    make_report(results, best, baseline_060, old_dual)

    print("=" * 72)
    print(f"total_samples: {len(rows)}")
    print(f"total_candidates: {len(results)}")
    print(f"zero_false_hit_candidates: {len(candidates)}")
    print(f"best_weak_threshold: {best['weak_threshold']}")
    print(f"best_strong_threshold: {best['strong_threshold']}")
    print(f"best_false_hit_rate: {best['false_hit_rate']}")
    print(f"best_recall: {best['recall']}")
    print(f"best_review_rate: {best['review_rate']}")
    print(f"report: {REPORT_MD}")
    print("=" * 72)


if __name__ == "__main__":
    main()
