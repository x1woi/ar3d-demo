#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Ablation for learning-based fusion without metadata features.

This script only uses online-available score features from the existing
v3_real_70 learning dataset. It does not call Qwen, TripoSR, Stable Fast 3D,
or modify plus.py/runtime policy.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import warnings
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


DEFAULT_DATASET = Path(
    "paper_repro_outputs/cache_similarity_eval_v3_real_70/learning_fusion_classifier/"
    "learning_fusion_dataset.csv"
)
DEFAULT_ORIGINAL_JSON = Path(
    "paper_repro_outputs/cache_similarity_eval_v3_real_70/learning_fusion_classifier/"
    "learning_fusion_model_comparison.json"
)
DEFAULT_OUTPUT_DIR = Path(
    "paper_repro_outputs/cache_similarity_eval_v3_real_70/"
    "learning_fusion_classifier_ablation_no_metadata"
)
FEATURE_COLUMNS = [
    "text_score",
    "image_score",
    "fusion_score",
    "score_abs_diff",
    "score_max",
    "score_min",
]
RANDOM_STATE = 42
RULE_THRESHOLD = 0.6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train score-only fusion classifiers without sample_type/category metadata."
    )
    parser.add_argument("--dataset-csv", default=str(DEFAULT_DATASET))
    parser.add_argument("--original-json", default=str(DEFAULT_ORIGINAL_JSON))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--random-state", type=int, default=RANDOM_STATE)
    return parser.parse_args()


def to_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    text = str(value).strip()
    if not text:
        return default
    try:
        out = float(text)
    except ValueError:
        return default
    if math.isnan(out) or math.isinf(out):
        return default
    return out


def to_int_label(value: Any) -> int:
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "hit"}:
        return 1
    return 0


def read_dataset(path: Path) -> Tuple[List[Dict[str, Any]], np.ndarray, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"learning dataset not found: {path}")
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        missing = [c for c in FEATURE_COLUMNS + ["should_hit"] if c not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"dataset missing required columns: {missing}")
        for index, row in enumerate(reader):
            item = {
                "index": int(to_float(row.get("index"), index)),
                "image": row.get("image", ""),
                "should_hit": to_int_label(row.get("should_hit")),
            }
            for col in FEATURE_COLUMNS:
                item[col] = to_float(row.get(col))
            rows.append(item)
    x = np.array([[item[col] for col in FEATURE_COLUMNS] for item in rows], dtype=float)
    y = np.array([item["should_hit"] for item in rows], dtype=int)
    return rows, x, y


def write_ablation_dataset(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["index", "image", "should_hit", *FEATURE_COLUMNS]
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for item in rows:
            writer.writerow({k: item.get(k, "") for k in fields})


def make_models(random_state: int) -> Dict[str, Any]:
    return {
        "Logistic Regression": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        max_iter=1000,
                        class_weight="balanced",
                        solver="liblinear",
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=200,
            max_depth=4,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=random_state,
        ),
        "MLPClassifier": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "model",
                    MLPClassifier(
                        hidden_layer_sizes=(8,),
                        activation="relu",
                        alpha=0.01,
                        max_iter=2000,
                        random_state=random_state,
                    ),
                ),
            ]
        ),
    }


def calc_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    positives = int((y_true == 1).sum())
    negatives = int((y_true == 0).sum())
    return {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "false_hit_rate": round(fp / negatives, 4) if negatives else 0.0,
        "false_miss_rate": round(fn / positives, 4) if positives else 0.0,
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "auto_hit_count": int(y_pred.sum()),
        "false_hit_count": fp,
        "false_miss_count": fn,
    }


def evaluate(x: np.ndarray, y: np.ndarray, random_state: int) -> Tuple[List[Dict[str, Any]], Dict[str, np.ndarray]]:
    if min(Counter(y.tolist()).values()) < 5:
        raise ValueError("StratifiedKFold(n_splits=5) requires at least 5 samples per class.")
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

    predictions: Dict[str, np.ndarray] = {
        "rule-fusion baseline": (x[:, FEATURE_COLUMNS.index("fusion_score")] >= RULE_THRESHOLD).astype(int)
    }

    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    for name, model in make_models(random_state).items():
        pred = np.zeros(len(y), dtype=int)
        for train_idx, test_idx in splitter.split(x, y):
            model.fit(x[train_idx], y[train_idx])
            pred[test_idx] = model.predict(x[test_idx])
        predictions[name] = pred

    rows: List[Dict[str, Any]] = []
    for method, pred in predictions.items():
        row = {"method": method, **calc_metrics(y, pred), "recommended_for_integration": False}
        if method == "rule-fusion baseline":
            row["remarks"] = "当前规则融合 baseline，阈值=0.6。"
        else:
            row["remarks"] = "仅使用分数特征的学习式消融结果，暂不接入 plus.py。"
        rows.append(row)
    return rows, predictions


def write_comparison_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    fields = [
        "method",
        "accuracy",
        "precision",
        "recall",
        "false_hit_rate",
        "false_miss_rate",
        "f1",
        "auto_hit_count",
        "false_hit_count",
        "false_miss_count",
        "recommended_for_integration",
        "remarks",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})


def read_original_result(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8-sig") as f:
            return json.load(f)
    except Exception:
        return {}


def pick_best(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    return sorted(
        rows,
        key=lambda r: (
            1 if r["false_hit_rate"] == 0 else 0,
            r["recall"],
            r["f1"],
            -r["false_miss_rate"],
        ),
        reverse=True,
    )[0]


def markdown_table(rows: List[Dict[str, Any]]) -> str:
    cols = [
        "method",
        "accuracy",
        "precision",
        "recall",
        "false_hit_rate",
        "false_miss_rate",
        "f1",
        "auto_hit_count",
        "false_hit_count",
        "false_miss_count",
    ]
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(c, "")) for c in cols) + " |")
    return "\n".join(lines)


def write_report(
    path: Path,
    dataset_size: int,
    y: np.ndarray,
    rows: List[Dict[str, Any]],
    original: Dict[str, Any],
    best: Dict[str, Any],
) -> str:
    original_best = original.get("best_model", {}) if isinstance(original, dict) else {}
    original_best_name = original_best.get("method", "unknown")
    original_recall = original_best.get("recall", "unknown")
    original_false_hit = original_best.get("false_hit_rate", "unknown")

    rule = next(r for r in rows if r["method"] == "rule-fusion baseline")
    learning_rows = [r for r in rows if r["method"] != "rule-fusion baseline"]
    learning_best = max(learning_rows, key=lambda r: (r["recall"], -r["false_hit_rate"], r["f1"]))
    advantage_declined = (
        learning_best["recall"] < 1.0
        or learning_best["false_hit_rate"] > 0
        or learning_best["recall"] <= rule["recall"]
    )
    conclusion = (
        "上一轮高分可能受到元数据影响。当前仍推荐规则融合 + 双阈值策略作为主流程。"
        if advantage_declined
        else "学习式融合在去元数据后仍有潜力，但由于样本只有 70 条，仍需扩样本和独立测试集验证，暂不接入 plus.py。"
    )
    compare_text = (
        f"原学习式实验最佳模型为 {original_best_name}，recall={original_recall}，"
        f"false_hit_rate={original_false_hit}；本轮去元数据最佳模型为 {best['method']}，"
        f"recall={best['recall']}，false_hit_rate={best['false_hit_rate']}。"
    )
    label_counts = Counter(y.tolist())
    text = f"""# 去元数据学习式图文融合消融实验报告

## 1. 实验目的

上一轮学习式融合实验使用了 sample_type / category 等元数据，可能存在标签泄漏风险。本轮只使用线上可获得的相似度分数特征，验证学习式融合是否仍然有效。

## 2. 特征设置

使用：

- text_score
- image_score
- fusion_score
- score_abs_diff
- score_max
- score_min

不使用：

- sample_type
- category

样本总数：{dataset_size}

标签分布：should_hit=True {label_counts.get(1, 0)}，should_hit=False {label_counts.get(0, 0)}

交叉验证：StratifiedKFold(n_splits=5)，random_state=42

## 3. 对比方法

- rule-fusion baseline
- Logistic Regression
- RandomForest
- MLPClassifier

## 4. 实验结果

{markdown_table(rows)}

## 5. 结果分析

1. 去掉元数据后，最佳学习式模型为 `{learning_best['method']}`，recall={learning_best['recall']}；规则融合 baseline recall={rule['recall']}。
2. false_hit_rate：最佳学习式模型 false_hit_rate={learning_best['false_hit_rate']}，false_hit_count={learning_best['false_hit_count']}。如果出现 false_hit，则需要优先保守处理。
3. recall 变化：{compare_text}
4. RandomForest / MLP 稳定性：在 70 条小样本上，复杂模型容易受到分数分布偶然性影响，即使指标较高，也不能直接证明线上泛化。
5. 是否接入 plus.py：当前没有足够证据直接接入 plus.py，也不修改 runtime policy。

## 6. 当前结论

{conclusion}

本轮实验只作为 preliminary / exploratory，用于回应导师关于 MLP / 学习式融合的方向建议。后续若继续推进，需要扩样本、独立测试集和更严格的线上可获得特征约束。
"""
    path.write_text(text, encoding="utf-8-sig")
    return compare_text


def main() -> None:
    args = parse_args()
    dataset_path = Path(args.dataset_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows, x, y = read_dataset(dataset_path)
    write_ablation_dataset(output_dir / "learning_fusion_ablation_dataset.csv", rows)
    comparison, _ = evaluate(x, y, args.random_state)
    write_comparison_csv(output_dir / "learning_fusion_ablation_comparison.csv", comparison)
    best = pick_best(comparison)
    original = read_original_result(Path(args.original_json))
    compared = write_report(
        output_dir / "learning_fusion_ablation_report.md",
        len(rows),
        y,
        comparison,
        original,
        best,
    )
    payload = {
        "schema": "learning_fusion_classifier_ablation_no_metadata.v1",
        "dataset_csv": str(dataset_path),
        "dataset_size": len(rows),
        "feature_columns": FEATURE_COLUMNS,
        "excluded_features": ["sample_type", "category"],
        "cv": "StratifiedKFold(n_splits=5, shuffle=True, random_state=42)",
        "rule_threshold": RULE_THRESHOLD,
        "best_model": best,
        "compared_to_original_learning_result": compared,
        "recommended_for_integration": False,
        "models": comparison,
    }
    (output_dir / "learning_fusion_ablation_comparison.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("========================================================================")
    print(f"ablation_dataset_size: {len(rows)}")
    print(f"feature_columns: {FEATURE_COLUMNS}")
    print(f"best_model: {best['method']}")
    print(f"best_model_false_hit_rate: {best['false_hit_rate']}")
    print(f"best_model_recall: {best['recall']}")
    print(f"compared_to_original_learning_result: {compared}")
    print("recommended_for_integration: False")
    print(f"learning_fusion_ablation_report.md: {output_dir / 'learning_fusion_ablation_report.md'}")
    print("========================================================================")


if __name__ == "__main__":
    main()
