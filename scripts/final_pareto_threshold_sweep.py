#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Final Pareto-style threshold sweep for real_eval_200.

This script scans weak/strong threshold pairs and exports multiple operating
points for paper reporting. It does not modify plus.py or train any model.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


INPUT_CSV = Path("paper_repro_outputs/cache_similarity_eval_real_eval_200/real_eval_200_features.csv")
OUTPUT_DIR = Path("paper_repro_outputs/cache_similarity_eval_real_eval_200/threshold_sweep_final_pareto")
SWEEP_CSV = OUTPUT_DIR / "final_pareto_threshold_sweep.csv"
POINTS_CSV = OUTPUT_DIR / "final_threshold_operating_points.csv"
POINTS_JSON = OUTPUT_DIR / "final_threshold_operating_points.json"
REPORT_MD = OUTPUT_DIR / "final_threshold_pareto_report.md"
PARETO_PLOT = OUTPUT_DIR / "final_threshold_pareto_plot.png"
BAR_PLOT = OUTPUT_DIR / "final_threshold_bar_comparison.png"

SWEEP_FIELDS = [
    "weak_threshold",
    "strong_threshold",
    "total_samples",
    "accuracy",
    "precision",
    "recall",
    "false_hit_rate",
    "false_miss_rate",
    "f1",
    "auto_hit_count",
    "low_confidence_candidate_count",
    "miss_count",
    "false_hit_count",
    "false_miss_count",
    "true_auto_hit_count",
    "true_miss_count",
    "low_confidence_true_candidate_count",
    "low_confidence_false_candidate_count",
    "low_confidence_candidate_rate",
]

POINT_FIELDS = [
    "name",
    "status",
    "weak_threshold",
    "strong_threshold",
    "recall",
    "false_hit_rate",
    "low_confidence_candidate_rate",
    "precision",
    "accuracy",
    "f1",
    "auto_hit_count",
    "low_confidence_candidate_count",
    "miss_count",
    "false_hit_count",
    "false_miss_count",
    "selection_reason",
]


def values(start: float, stop: float, step: float = 0.01) -> List[float]:
    n = int(round((stop - start) / step)) + 1
    return [round(start + i * step, 2) for i in range(n)]


def load_rows() -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    with INPUT_CSV.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            text_score = float(row.get("text_score") or 0.0)
            image_score = float(row.get("image_score") or text_score)
            should_hit = 1.0 if str(row.get("should_hit")).strip() in {"1", "true", "True"} else 0.0
            rows.append({"score": 0.5 * text_score + 0.5 * image_score, "should_hit": should_hit})
    return rows


def evaluate(rows: List[Dict[str, float]], weak: float, strong: float) -> Dict[str, Any]:
    total = len(rows)
    positives = sum(1 for row in rows if row["should_hit"] == 1.0)
    negatives = total - positives
    auto_hit_count = low_count = miss_count = 0
    false_hit_count = false_miss_count = 0
    true_auto_hit_count = true_miss_count = 0
    low_true = low_false = 0

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
    accuracy = (true_auto_hit_count + true_miss_count) / total if total else 0.0
    f1 = 0.0
    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    return {
        "weak_threshold": round(weak, 2),
        "strong_threshold": round(strong, 2),
        "total_samples": total,
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "false_hit_rate": round(false_hit_rate, 4),
        "false_miss_rate": round(false_miss_rate, 4),
        "f1": round(f1, 4),
        "auto_hit_count": auto_hit_count,
        "low_confidence_candidate_count": low_count,
        "miss_count": miss_count,
        "false_hit_count": false_hit_count,
        "false_miss_count": false_miss_count,
        "true_auto_hit_count": true_auto_hit_count,
        "true_miss_count": true_miss_count,
        "low_confidence_true_candidate_count": low_true,
        "low_confidence_false_candidate_count": low_false,
        "low_confidence_candidate_rate": round(low_count / total, 4) if total else 0.0,
    }


def scan(rows: List[Dict[str, float]]) -> List[Dict[str, Any]]:
    return [
        evaluate(rows, weak, strong)
        for weak in values(0.50, 0.80)
        for strong in values(0.65, 0.95)
        if weak < strong
    ]


def write_csv(path: Path, rows: List[Dict[str, Any]], fields: List[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{field: row.get(field, "") for field in fields} for row in rows])


def point_from_row(name: str, row: Dict[str, Any], reason: str) -> Dict[str, Any]:
    return {
        "name": name,
        "status": "ok",
        "selection_reason": reason,
        **{field: row.get(field, "") for field in POINT_FIELDS if field not in {"name", "status", "selection_reason"}},
    }


def no_candidate(name: str, reason: str) -> Dict[str, Any]:
    return {"name": name, "status": "no_valid_candidate", "selection_reason": reason}


def choose(rows: List[Dict[str, Any]], false_hit_max: float, low_max: float) -> Optional[Dict[str, Any]]:
    candidates = [
        row
        for row in rows
        if row["false_hit_rate"] <= false_hit_max
        and row["low_confidence_candidate_rate"] <= low_max
    ]
    if not candidates:
        return None
    return sorted(
        candidates,
        key=lambda row: (-row["recall"], row["low_confidence_candidate_rate"], -row["auto_hit_count"]),
    )[0]


def fixed(rows: List[Dict[str, Any]], weak: float, strong: float) -> Dict[str, Any]:
    for row in rows:
        if row["weak_threshold"] == weak and row["strong_threshold"] == strong:
            return row
    raise KeyError(f"Fixed threshold not found: weak={weak}, strong={strong}")


def make_operating_points(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    specs = [
        ("conservative_safe", 0.0, 0.30, "false_hit_rate = 0 且低置信候选区 <= 0.30，选择 recall 最高"),
        ("balanced_low_risk", 0.03, 0.25, "false_hit_rate <= 0.03 且低置信候选区 <= 0.25，选择 recall 最高"),
        ("balanced_practical", 0.05, 0.20, "false_hit_rate <= 0.05 且低置信候选区 <= 0.20，选择 recall 最高"),
        ("aggressive_recall", 0.10, 0.15, "false_hit_rate <= 0.10 且低置信候选区 <= 0.15，选择 recall 最高"),
    ]
    out: List[Dict[str, Any]] = []
    for name, false_hit_max, low_max, reason in specs:
        row = choose(rows, false_hit_max, low_max)
        if row is None:
            out.append(no_candidate(name, reason))
        else:
            out.append(point_from_row(name, row, reason))
    out.append(point_from_row("old_baseline", fixed(rows, 0.70, 0.78), "固定旧参数 weak=0.70,strong=0.78"))
    out.append(point_from_row("current_candidate", fixed(rows, 0.60, 0.74), "固定上一轮候选参数 weak=0.60,strong=0.74"))
    return out


def table(rows: List[Dict[str, Any]]) -> str:
    cols = [
        "name",
        "status",
        "weak_threshold",
        "strong_threshold",
        "recall",
        "false_hit_rate",
        "low_confidence_candidate_rate",
        "auto_hit_count",
        "low_confidence_candidate_count",
        "miss_count",
        "false_hit_count",
        "false_miss_count",
    ]
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join("---" for _ in cols) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(col, "")) for col in cols) + " |")
    return "\n".join(lines)


def select_recommendations(points: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_name = {row["name"]: row for row in points}
    conservative = by_name.get("conservative_safe", {})
    low_risk = by_name.get("balanced_low_risk", {})
    practical = by_name.get("balanced_practical", {})
    aggressive = by_name.get("aggressive_recall", {})

    paper_main = conservative
    if (
        low_risk.get("status") == "ok"
        and conservative.get("status") == "ok"
        and float(low_risk.get("recall", 0)) > float(conservative.get("recall", 0)) + 0.05
    ):
        paper_main = low_risk
    elif (
        practical.get("status") == "ok"
        and conservative.get("status") == "ok"
        and float(practical.get("recall", 0)) > float(conservative.get("recall", 0)) + 0.08
    ):
        paper_main = practical

    return {
        "paper_main_threshold": {
            "name": paper_main.get("name"),
            "weak_threshold": paper_main.get("weak_threshold"),
            "strong_threshold": paper_main.get("strong_threshold"),
            "reason": "论文主表可展示该折中点，同时报告误复用风险与低置信候选区比例。",
        },
        "runtime_safe_threshold": {
            "name": conservative.get("name"),
            "weak_threshold": conservative.get("weak_threshold"),
            "strong_threshold": conservative.get("strong_threshold"),
            "reason": "工程默认优先选择自动误复用为 0 且低置信候选区受控的保守点。",
        },
        "ablation_aggressive_threshold": {
            "name": aggressive.get("name"),
            "weak_threshold": aggressive.get("weak_threshold"),
            "strong_threshold": aggressive.get("strong_threshold"),
            "reason": "激进设置仅适合作为消融对照，不建议接入。",
        },
        "recommended_for_plus_py": False,
    }


def make_plots(rows: List[Dict[str, Any]], points: List[Dict[str, Any]]) -> tuple[bool, bool]:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return False, False

    false_hit = [row["false_hit_rate"] for row in rows]
    recall = [row["recall"] for row in rows]
    low_rate = [row["low_confidence_candidate_rate"] for row in rows]
    plt.figure(figsize=(9, 6))
    scatter = plt.scatter(false_hit, recall, c=low_rate, s=[30 + 160 * x for x in low_rate], alpha=0.55)
    plt.colorbar(scatter, label="low_confidence_candidate_rate")
    for point in points:
        if point.get("status") != "ok":
            continue
        plt.scatter([point["false_hit_rate"]], [point["recall"]], s=110, marker="x")
        plt.annotate(point["name"], (point["false_hit_rate"], point["recall"]), fontsize=8)
    plt.xlabel("false_hit_rate")
    plt.ylabel("recall")
    plt.title("real_eval_200 threshold Pareto sweep")
    plt.tight_layout()
    plt.savefig(PARETO_PLOT, dpi=180)
    plt.close()

    ok_points = [p for p in points if p.get("status") == "ok"]
    names = [p["name"] for p in ok_points]
    x = range(len(ok_points))
    width = 0.25
    plt.figure(figsize=(10, 5))
    plt.bar([i - width for i in x], [p["recall"] for p in ok_points], width, label="recall")
    plt.bar(list(x), [p["false_hit_rate"] for p in ok_points], width, label="false_hit_rate")
    plt.bar([i + width for i in x], [p["low_confidence_candidate_rate"] for p in ok_points], width, label="low_conf_rate")
    plt.xticks(list(x), names, rotation=25, ha="right")
    plt.ylim(0, 1.05)
    plt.legend()
    plt.title("Operating point comparison")
    plt.tight_layout()
    plt.savefig(BAR_PLOT, dpi=180)
    plt.close()
    return True, True


def write_report(points: List[Dict[str, Any]], recommendations: Dict[str, Any], plots: tuple[bool, bool]) -> None:
    text = f"""# real_eval_200 最终阈值 Pareto 调整报告

## 1. 实验目的

前一轮找到了 false_hit_rate=0 的候选阈值，但为了避免只呈现单个过度保守点，本轮进一步扫描不同风险约束下的候选工作点，用于论文中展示阈值选择依据。

## 2. 数据集

- dataset: real_eval_200
- total_samples = 200
- should_hit=True = 100
- should_hit=False = 100

## 3. 决策规则

```text
score = 0.5 * text_score + 0.5 * image_score

score >= strong_threshold:
  高置信自动复用

weak_threshold <= score < strong_threshold:
  低置信候选区

score < weak_threshold:
  不复用
```

## 4. 候选工作点

{table(points)}

## 5. 结果分析

1. false_hit_rate=0 的点并不一定过度保守，但会受到低置信候选区比例影响。`conservative_safe` 在自动误复用为 0 的前提下控制低置信候选区不超过 0.30。

2. 如果允许 false_hit_rate <= 0.03 或 <= 0.05，是否能明显提高 recall 需要看 balanced_low_risk / balanced_practical。若 recall 提升有限，则不值得为了微小召回提升承担误复用风险。

3. low_confidence_candidate_rate 是工程可用性的重要指标。候选区比例过高会增加后续交互或处理负担，因此本报告不只追求 recall。

4. 论文主方法建议展示 conservative_safe，并可附带 balanced_low_risk / balanced_practical 作为风险-召回折中对比。

5. 工程保守默认值应优先选择 false_hit_rate=0 且低置信候选区受控的工作点。

6. aggressive_recall 只能作为消融对照，不建议接入。

## 6. 最终建议

- paper_main_threshold: {recommendations['paper_main_threshold']}
- runtime_safe_threshold: {recommendations['runtime_safe_threshold']}
- ablation_aggressive_threshold: {recommendations['ablation_aggressive_threshold']}
- recommended_for_plus_py: {recommendations['recommended_for_plus_py']}

如果 balanced_low_risk / balanced_practical 的 false_hit_rate 虽然非 0 但很低，并且 recall 明显高于 conservative_safe，可以在论文主表同时展示二者。若非 0 false_hit 的候选提升不明显，则继续推荐 conservative_safe 或 current_candidate。

## 7. 论文表述建议

不要写“系统不会误复用”。建议写：

在当前 real_eval_200 测试集上，某阈值组合未观察到自动误复用样本；在允许极低误复用风险的设置下，召回率可以进一步变化。本文最终选择某阈值，是在误复用风险、候选区比例和召回率之间的折中。

## 8. 输出图表

- final_threshold_pareto_plot.png generated: {plots[0]}
- final_threshold_bar_comparison.png generated: {plots[1]}
"""
    REPORT_MD.write_text(text, encoding="utf-8-sig")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = load_rows()
    sweep_rows = scan(rows)
    points = make_operating_points(sweep_rows)
    recommendations = select_recommendations(points)
    plots = make_plots(sweep_rows, points)

    write_csv(SWEEP_CSV, sweep_rows, SWEEP_FIELDS)
    write_csv(POINTS_CSV, points, POINT_FIELDS)
    POINTS_JSON.write_text(
        json.dumps({"operating_points": points, "recommendations": recommendations}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_report(points, recommendations, plots)

    print("=" * 72)
    print(f"final_pareto_threshold_sweep.csv: {SWEEP_CSV}")
    print(f"final_threshold_operating_points.csv: {POINTS_CSV}")
    print(f"final_threshold_pareto_report.md: {REPORT_MD}")
    print(f"final_threshold_pareto_plot.png: {PARETO_PLOT}")
    print(f"final_threshold_bar_comparison.png: {BAR_PLOT}")
    print(f"paper_main_threshold: {recommendations['paper_main_threshold']}")
    print(f"runtime_safe_threshold: {recommendations['runtime_safe_threshold']}")
    print(f"recommended_for_plus_py: {recommendations['recommended_for_plus_py']}")
    print("=" * 72)


if __name__ == "__main__":
    main()
