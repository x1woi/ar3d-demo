#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Sweep real_eval_200 dual thresholds with a low-confidence rate constraint."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List


INPUT_CSV = Path("paper_repro_outputs/cache_similarity_eval_real_eval_200/real_eval_200_features.csv")
OUTPUT_DIR = Path("paper_repro_outputs/cache_similarity_eval_real_eval_200/threshold_sweep_conservative")
SWEEP_CSV = OUTPUT_DIR / "threshold_sweep_with_review_constraint.csv"
BEST_JSON = OUTPUT_DIR / "best_threshold_review_constrained.json"
REPORT_MD = OUTPUT_DIR / "threshold_review_constraint_report.md"

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
    "low_confidence_candidate_rate",
    "auto_hit_count",
    "low_confidence_candidate_count",
    "miss_count",
    "false_hit_count",
    "false_miss_count",
    "true_auto_hit_count",
    "low_confidence_true_candidate_count",
    "low_confidence_false_candidate_count",
]


def values(start: float, stop: float, step: float = 0.01) -> List[float]:
    n = int(round((stop - start) / step)) + 1
    return [round(start + i * step, 2) for i in range(n)]


def load_rows(path: Path) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            text_score = float(row.get("text_score") or 0.0)
            image_score = float(row.get("image_score") or text_score)
            rows.append(
                {
                    "score": 0.5 * text_score + 0.5 * image_score,
                    "should_hit": 1.0 if str(row.get("should_hit")).strip() in {"1", "true", "True"} else 0.0,
                }
            )
    return rows


def evaluate(rows: List[Dict[str, float]], weak: float, strong: float) -> Dict[str, Any]:
    total = len(rows)
    positives = sum(1 for row in rows if row["should_hit"] == 1.0)
    negatives = total - positives
    auto_hit_count = 0
    low_count = 0
    miss_count = 0
    false_hit_count = 0
    false_miss_count = 0
    true_auto_hit_count = 0
    true_miss_count = 0
    low_true = 0
    low_false = 0

    for row in rows:
        score = row["score"]
        should_hit = row["should_hit"] == 1.0
        if score >= strong:
            auto_hit_count += 1
            if should_hit:
                true_auto_hit_count += 1
            else:
                false_hit_count += 1
        elif score >= weak:
            low_count += 1
            if should_hit:
                low_true += 1
            else:
                low_false += 1
        else:
            miss_count += 1
            if should_hit:
                false_miss_count += 1
            else:
                true_miss_count += 1

    precision = true_auto_hit_count / auto_hit_count if auto_hit_count else 0.0
    recall = (true_auto_hit_count + low_true) / positives if positives else 0.0
    false_hit_rate = false_hit_count / negatives if negatives else 0.0
    false_miss_rate = false_miss_count / positives if positives else 0.0
    low_rate = low_count / total if total else 0.0
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
        "review_rate": round(low_rate, 4),
        "low_confidence_candidate_rate": round(low_rate, 4),
        "auto_hit_count": auto_hit_count,
        "low_confidence_candidate_count": low_count,
        "miss_count": miss_count,
        "false_hit_count": false_hit_count,
        "false_miss_count": false_miss_count,
        "true_auto_hit_count": true_auto_hit_count,
        "low_confidence_true_candidate_count": low_true,
        "low_confidence_false_candidate_count": low_false,
    }


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows([{field: row.get(field, "") for field in FIELDS} for row in rows])


def choose_best(rows: List[Dict[str, Any]]) -> tuple[Dict[str, Any], float, str]:
    strict = [row for row in rows if row["false_hit_rate"] == 0.0 and row["review_rate"] <= 0.30]
    if strict:
        pool = strict
        constraint = 0.30
        mode = "strict"
    else:
        relaxed = [row for row in rows if row["false_hit_rate"] == 0.0 and row["review_rate"] <= 0.35]
        pool = relaxed
        constraint = 0.35
        mode = "relaxed"
    if not pool:
        raise RuntimeError("No threshold pair satisfies false_hit_rate=0 with review_rate<=0.35.")
    best = sorted(
        pool,
        key=lambda row: (-row["recall"], row["review_rate"], -row["auto_hit_count"]),
    )[0]
    return best, constraint, mode


def make_report(rows: List[Dict[str, Any]], best: Dict[str, Any], constraint: float, mode: str) -> None:
    old = evaluate(load_rows(INPUT_CSV), 0.70, 0.78)
    top = sorted(
        [row for row in rows if row["false_hit_rate"] == 0.0 and row["review_rate"] <= constraint],
        key=lambda row: (-row["recall"], row["review_rate"], -row["auto_hit_count"]),
    )[:12]
    table = [
        "| weak | strong | recall | false_hit_rate | low_conf_rate | auto_hit | low_conf | miss |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in top:
        table.append(
            f"| {row['weak_threshold']} | {row['strong_threshold']} | {row['recall']} | "
            f"{row['false_hit_rate']} | {row['low_confidence_candidate_rate']} | "
            f"{row['auto_hit_count']} | {row['low_confidence_candidate_count']} | {row['miss_count']} |"
        )

    text = f"""# real_eval_200 带低置信候选约束的双阈值扫描报告

## 1. 实验目的

本轮重新扫描 real_eval_200 的 weak_threshold / strong_threshold，但增加 `low_confidence_candidate_rate <= 0.30` 的约束。目标是在自动误复用为 0 的前提下，避免中间候选区过大。

## 2. 决策定义

```text
score = 0.5 * text_score + 0.5 * image_score

score >= strong_threshold:
  auto_hit

weak_threshold <= score < strong_threshold:
  low_confidence_candidate

score < weak_threshold:
  miss
```

注意：本报告不再使用“人工确认”表述，中间区统一称为 low_confidence_candidate / 低置信候选区。

## 3. 扫描设置

- weak_threshold: 0.55 到 0.85
- strong_threshold: 0.65 到 0.95
- step: 0.01
- weak_threshold < strong_threshold

选择规则：

1. false_hit_rate = 0
2. low_confidence_candidate_rate <= 0.30
3. recall 最高
4. low_confidence_candidate_rate 更低
5. auto_hit_count 更高

如果没有满足 0.30 的组合，才放宽到 0.35。本轮实际使用约束：{constraint}（{mode}）。

## 4. 原参数对比

- old weak = 0.70
- old strong = 0.78
- old false_hit_rate = {old['false_hit_rate']}
- old recall = {old['recall']}
- old low_confidence_candidate_rate = {old['low_confidence_candidate_rate']}
- old auto_hit_count = {old['auto_hit_count']}
- old low_confidence_candidate_count = {old['low_confidence_candidate_count']}

## 5. 推荐阈值

- recommended_weak_threshold = {best['weak_threshold']}
- recommended_strong_threshold = {best['strong_threshold']}
- false_hit_rate = {best['false_hit_rate']}
- recall = {best['recall']}
- low_confidence_candidate_rate = {best['low_confidence_candidate_rate']}
- auto_hit_count = {best['auto_hit_count']}
- low_confidence_candidate_count = {best['low_confidence_candidate_count']}
- miss_count = {best['miss_count']}
- false_miss_count = {best['false_miss_count']}

## 6. Top 候选

{chr(10).join(table)}

## 7. 结论

本轮推荐参数只作为候选，不直接修改 plus.py。相比无约束扫描得到的 `weak=0.55,strong=0.74`，本轮增加低置信候选区规模约束，更适合工程运行时控制交互/候选负担。
"""
    REPORT_MD.write_text(text, encoding="utf-8-sig")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = load_rows(INPUT_CSV)
    results = [
        evaluate(rows, weak, strong)
        for weak in values(0.55, 0.85)
        for strong in values(0.65, 0.95)
        if weak < strong
    ]
    write_csv(SWEEP_CSV, results)
    best, constraint, mode = choose_best(results)
    payload = {
        "selection_rule": [
            "false_hit_rate = 0",
            "review_rate/low_confidence_candidate_rate <= 0.30, else <= 0.35",
            "maximize recall",
            "tie-break lower low_confidence_candidate_rate",
            "tie-break higher auto_hit_count",
        ],
        "constraint_used": constraint,
        "constraint_mode": mode,
        "best": best,
        "total_candidates": len(results),
        "zero_false_hit_candidates": len([r for r in results if r["false_hit_rate"] == 0.0]),
        "strict_candidates": len([r for r in results if r["false_hit_rate"] == 0.0 and r["review_rate"] <= 0.30]),
        "relaxed_candidates": len([r for r in results if r["false_hit_rate"] == 0.0 and r["review_rate"] <= 0.35]),
        "recommended_for_plus_py_change": False,
    }
    BEST_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    make_report(results, best, constraint, mode)
    print("=" * 72)
    print(f"total_candidates: {len(results)}")
    print(f"constraint_used: {constraint}")
    print(f"constraint_mode: {mode}")
    print(f"best_weak_threshold: {best['weak_threshold']}")
    print(f"best_strong_threshold: {best['strong_threshold']}")
    print(f"best_false_hit_rate: {best['false_hit_rate']}")
    print(f"best_recall: {best['recall']}")
    print(f"best_low_confidence_candidate_rate: {best['low_confidence_candidate_rate']}")
    print(f"best_auto_hit_count: {best['auto_hit_count']}")
    print(f"report: {REPORT_MD}")
    print("=" * 72)


if __name__ == "__main__":
    main()
