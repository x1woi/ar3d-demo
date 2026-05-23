#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Train small sklearn classifiers for v3_real_70 similarity fusion.

This script is intentionally offline-only: it reads existing v3_real_70
similarity scores and does not call Qwen, TripoSR, Stable Fast 3D, or plus.py.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import LeaveOneOut, StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


DEFAULT_SUMMARY_CSV = Path("paper_repro_outputs/cache_similarity_eval_v3_real_70/summary.csv")
DEFAULT_SUMMARY_JSON = Path("paper_repro_outputs/cache_similarity_eval_v3_real_70/summary.json")
DEFAULT_OUTPUT_DIR = Path(
    "paper_repro_outputs/cache_similarity_eval_v3_real_70/learning_fusion_classifier"
)
RANDOM_STATE = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train small fusion classifiers on v3_real_70 results.")
    parser.add_argument("--summary-csv", default=str(DEFAULT_SUMMARY_CSV))
    parser.add_argument("--summary-json", default=str(DEFAULT_SUMMARY_JSON))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--random-state", type=int, default=RANDOM_STATE)
    return parser.parse_args()


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"summary csv not found: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def to_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    text = str(value).strip()
    if text == "":
        return default
    try:
        out = float(text)
    except ValueError:
        return default
    if math.isnan(out) or math.isinf(out):
        return default
    return out


def to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    return text in {"true", "1", "yes", "y", "hit"}


def looks_garbled(value: str) -> bool:
    text = (value or "").strip()
    return bool(text) and text.count("?") >= max(2, len(text) // 2)


def build_dataset(rows: List[Dict[str, str]]) -> Tuple[List[Dict[str, Any]], List[str], List[str]]:
    dataset: List[Dict[str, Any]] = []
    categories = [r.get("category", "") for r in rows]
    usable_category = any(c and not looks_garbled(c) for c in categories)

    for idx, row in enumerate(rows):
        text_score = to_float(row.get("text_score"))
        image_score = to_float(row.get("image_score"), default=text_score)
        fusion_score = to_float(row.get("fusion_score"), default=float("nan"))
        if math.isnan(fusion_score):
            fusion_score = to_float(row.get("fused_score"), default=0.5 * text_score + 0.5 * image_score)
        score_abs_diff = abs(text_score - image_score)
        score_max = max(text_score, image_score)
        score_min = min(text_score, image_score)

        sample_type = row.get("sample_type", "") or "unknown"
        category = row.get("category", "") or "unknown"
        if looks_garbled(category) or not usable_category:
            category = "unknown"

        item: Dict[str, Any] = {
            "index": idx,
            "image": row.get("image", ""),
            "sample_type": sample_type,
            "category": category,
            "should_hit": int(to_bool(row.get("should_hit"))),
            "text_score": text_score,
            "image_score": image_score,
            "fusion_score": fusion_score,
            "score_abs_diff": score_abs_diff,
            "score_max": score_max,
            "score_min": score_min,
            "best_keyword": row.get("best_keyword", ""),
            "best_filename": row.get("best_filename", ""),
        }
        dataset.append(item)

    numeric_features = [
        "text_score",
        "image_score",
        "fusion_score",
        "score_abs_diff",
        "score_max",
        "score_min",
    ]
    categorical_features = ["sample_type"]
    if usable_category:
        categorical_features.append("category")
    return dataset, numeric_features, categorical_features


def write_dataset_csv(path: Path, dataset: List[Dict[str, Any]], feature_columns: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "index",
        "image",
        "sample_type",
        "category",
        "should_hit",
        *feature_columns,
        "best_keyword",
        "best_filename",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for item in dataset:
            writer.writerow({k: item.get(k, "") for k in fieldnames})


def metrics_from_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    positive_count = int((y_true == 1).sum())
    negative_count = int((y_true == 0).sum())
    return {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "false_hit_rate": round(fp / negative_count, 4) if negative_count else 0.0,
        "false_miss_rate": round(fn / positive_count, 4) if positive_count else 0.0,
        "auto_hit_count": int(y_pred.sum()),
        "false_hit_count": fp,
        "false_miss_count": fn,
    }


def group_metrics(dataset: List[Dict[str, Any]], y_pred: np.ndarray) -> Dict[str, Dict[str, Any]]:
    grouped: Dict[str, List[int]] = defaultdict(list)
    for i, item in enumerate(dataset):
        grouped[item["sample_type"]].append(i)

    out: Dict[str, Dict[str, Any]] = {}
    y_true = np.array([item["should_hit"] for item in dataset], dtype=int)
    for group in ["positive", "near_positive", "hard_negative", "negative"]:
        indices = grouped.get(group, [])
        if not indices:
            out[group] = {"count": 0, "hit_count": 0, "miss_count": 0, "false_hit_rate": 0.0, "false_miss_rate": 0.0}
            continue
        yt = y_true[indices]
        yp = y_pred[indices]
        m = metrics_from_predictions(yt, yp)
        out[group] = {
            "count": len(indices),
            "hit_count": int(yp.sum()),
            "miss_count": int(len(indices) - yp.sum()),
            "false_hit_rate": m["false_hit_rate"],
            "false_miss_rate": m["false_miss_rate"],
        }
    return out


def make_models(
    numeric_features: Sequence[str], categorical_features: Sequence[str], random_state: int
) -> Dict[str, Pipeline]:
    return {
        "Logistic Regression": Pipeline(
            steps=[
                ("vectorizer", DictVectorizer(sparse=True)),
                ("scaler", StandardScaler(with_mean=False)),
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
        "RandomForest": Pipeline(
            steps=[
                ("vectorizer", DictVectorizer(sparse=False)),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=200,
                        max_depth=4,
                        min_samples_leaf=2,
                        class_weight="balanced",
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        "MLPClassifier": Pipeline(
            steps=[
                ("vectorizer", DictVectorizer(sparse=True)),
                ("scaler", StandardScaler(with_mean=False)),
                (
                    "model",
                    MLPClassifier(
                        hidden_layer_sizes=(8,),
                        activation="relu",
                        alpha=0.01,
                        max_iter=1000,
                        random_state=random_state,
                    ),
                ),
            ]
        ),
    }


def select_cv(y: np.ndarray, random_state: int) -> Tuple[str, Iterable[Tuple[np.ndarray, np.ndarray]]]:
    counts = Counter(y.tolist())
    min_class_count = min(counts.values())
    if min_class_count >= 5:
        splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
        return "StratifiedKFold(n_splits=5)", splitter.split(np.zeros_like(y), y)
    if min_class_count >= 2:
        n_splits = min_class_count
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        return f"StratifiedKFold(n_splits={n_splits})", splitter.split(np.zeros_like(y), y)
    splitter = LeaveOneOut()
    return "LeaveOneOut", splitter.split(np.zeros_like(y), y)


def evaluate_models(
    dataset: List[Dict[str, Any]],
    numeric_features: Sequence[str],
    categorical_features: Sequence[str],
    rule_threshold: float,
    random_state: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, np.ndarray], str]:
    y = np.array([item["should_hit"] for item in dataset], dtype=int)
    x_records = [{**{f: item[f] for f in numeric_features}, **{f: item[f] for f in categorical_features}} for item in dataset]

    predictions: Dict[str, np.ndarray] = {}
    predictions["rule-fusion baseline"] = np.array(
        [1 if item["fusion_score"] >= rule_threshold else 0 for item in dataset], dtype=int
    )

    cv_name, cv_splits_iter = select_cv(y, random_state)
    cv_splits = list(cv_splits_iter)
    models = make_models(numeric_features, categorical_features, random_state)
    for name, model in models.items():
        pred = np.zeros(len(dataset), dtype=int)
        for train_idx, test_idx in cv_splits:
            x_train = [x_records[i] for i in train_idx]
            y_train = y[train_idx]
            x_test = [x_records[i] for i in test_idx]
            model.fit(x_train, y_train)
            pred[test_idx] = model.predict(x_test)
        predictions[name] = pred

    comparison: List[Dict[str, Any]] = []
    for name, pred in predictions.items():
        m = metrics_from_predictions(y, pred)
        groups = group_metrics(dataset, pred)
        row = {
            "method": name,
            **m,
            "positive_false_miss_rate": groups["positive"]["false_miss_rate"],
            "near_positive_false_miss_rate": groups["near_positive"]["false_miss_rate"],
            "hard_negative_false_hit_rate": groups["hard_negative"]["false_hit_rate"],
            "negative_false_hit_rate": groups["negative"]["false_hit_rate"],
            "recommended_for_integration": False,
            "remarks": "",
        }
        comparison.append(row)

    rule = next(r for r in comparison if r["method"] == "rule-fusion baseline")
    for row in comparison:
        if row["method"] == "rule-fusion baseline":
            row["remarks"] = f"当前工程 baseline，阈值={rule_threshold}，可解释且已完成双阈值验证。"
            continue
        if row["false_hit_rate"] > rule["false_hit_rate"]:
            row["remarks"] = "召回变化需结合误命中风险观察；当前样本量较小，不建议直接接入。"
        elif row["recall"] > rule["recall"]:
            row["remarks"] = "召回优于规则 baseline，但仅为小样本交叉验证结果，需要扩样本复核。"
        else:
            row["remarks"] = "未明显优于规则 baseline，暂不建议工程接入。"

    return comparison, predictions, cv_name


def fit_and_save_models(
    dataset: List[Dict[str, Any]],
    numeric_features: Sequence[str],
    categorical_features: Sequence[str],
    output_dir: Path,
    random_state: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    y = np.array([item["should_hit"] for item in dataset], dtype=int)
    x_records = [{**{f: item[f] for f in numeric_features}, **{f: item[f] for f in categorical_features}} for item in dataset]
    models = make_models(numeric_features, categorical_features, random_state)
    filenames = {
        "Logistic Regression": "logistic_regression.pkl",
        "RandomForest": "random_forest.pkl",
        "MLPClassifier": "mlp_classifier.pkl",
    }
    for name, model in models.items():
        model.fit(x_records, y)
        joblib.dump(model, output_dir / filenames[name])


def write_comparison_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
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
        "positive_false_miss_rate",
        "near_positive_false_miss_rate",
        "hard_negative_false_hit_rate",
        "negative_false_hit_rate",
        "recommended_for_integration",
        "remarks",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


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
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    lines = [header, sep]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(c, "")) for c in cols) + " |")
    return "\n".join(lines)


def choose_best_model(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    # Prefer zero false-hit models; then higher recall/f1. This is model selection for analysis,
    # not a recommendation to integrate into the runtime policy.
    return sorted(
        rows,
        key=lambda r: (
            0 if r["false_hit_rate"] == 0 else -1,
            r["recall"],
            r["f1"],
            -r["false_miss_rate"],
        ),
        reverse=True,
    )[0]


def write_report(
    path: Path,
    dataset: List[Dict[str, Any]],
    numeric_features: Sequence[str],
    categorical_features: Sequence[str],
    rows: List[Dict[str, Any]],
    best: Dict[str, Any],
    cv_name: str,
    rule_threshold: float,
) -> None:
    y_counts = Counter(item["should_hit"] for item in dataset)
    sample_type_counts = Counter(item["sample_type"] for item in dataset)
    learning_best = max([r for r in rows if r["method"] != "rule-fusion baseline"], key=lambda r: (r["recall"], -r["false_hit_rate"], r["f1"]))
    rule = next(r for r in rows if r["method"] == "rule-fusion baseline")
    learning_has_clear_advantage = (
        learning_best["false_hit_rate"] <= rule["false_hit_rate"]
        and learning_best["recall"] > rule["recall"]
        and learning_best["false_hit_count"] == 0
    )
    recommendation = (
        "学习式融合有一定潜力，但当前样本量只有 70 条，仍需要扩样本后再考虑接入 plus.py。"
        if learning_has_clear_advantage
        else "当前仍推荐使用规则融合 + 双阈值策略，因为它可解释、稳定、样本量要求低，并且已经完成工程验证。"
    )
    text = f"""# 学习式图文融合分类器初步实验报告

## 1. 实验目的

本实验对应导师会议中“后续可考虑将文本分数和图像分数输入 MLP 做分类器”的建议，用于初步判断学习式融合是否优于当前规则融合。当前样本只有 70 条，因此本实验只作为 preliminary / exploratory，不作为最终训练结论。

## 2. 数据来源

本实验只使用已有 v3_real_70 实验结果：`summary.csv` 与 `summary.json`。实验过程中没有重新调用 Qwen，没有调用 TripoSR，没有运行 Stable Fast 3D，也没有重新生成 3D 模型。

- 样本总数：{len(dataset)}
- should_hit=True：{y_counts.get(1, 0)}
- should_hit=False：{y_counts.get(0, 0)}
- sample_type 分布：{dict(sample_type_counts)}
- 交叉验证方式：{cv_name}

## 3. 特征设计

本轮使用的是已有相似度结果上的轻量特征，不涉及深度图像训练：

- 数值特征：{", ".join(numeric_features)}
- 类别特征：{", ".join(categorical_features)}
- 规则融合 baseline 阈值：{rule_threshold}

其中 `score_abs_diff`、`score_max`、`score_min` 由 text_score 与 image_score 派生，用于补充两个模态之间的差异信息。由于 category 在部分历史 CSV 中存在编码占位符，本脚本会自动跳过不可用的 category 字段。

## 4. 对比方法

- rule-fusion baseline
- Logistic Regression
- RandomForest
- MLPClassifier

## 5. 实验结果

{markdown_table(rows)}

完整结果已保存到 `learning_fusion_model_comparison.csv` 和 `learning_fusion_model_comparison.json`。

## 6. 结果分析

1. 是否提升 recall：本轮最佳学习式方法为 `{learning_best["method"]}`，recall={learning_best["recall"]}；规则融合 baseline recall={rule["recall"]}。
2. 是否引入 false_hit：最佳学习式方法 false_hit_rate={learning_best["false_hit_rate"]}，false_hit_count={learning_best["false_hit_count"]}。在缓存复用任务中，false_hit 比 false_miss 风险更高，因此不能只看 recall。
3. 元数据风险：本轮特征中包含 `sample_type` 和部分 `category` 信息，它们在离线实验中有助于观察学习式融合潜力，但也可能带来标签泄漏或分布记忆。因此学习式方法的高分不能直接等同于真实线上泛化能力。
4. Logistic Regression 稳定性：Logistic Regression 参数少、可解释性较好，适合作为后续扩样本后的第一类学习式 baseline。
5. RandomForest 过拟合风险：RandomForest 对小样本和类别元数据较敏感，容易记住 sample_type 分布，需要在更多 hard_negative / near_positive 上复核。
6. MLP 是否足够：当前只有 70 条样本，不足以支撑 MLP 作为稳定工程策略。MLPClassifier 可以作为探索，但不建议直接接入。

## 7. 当前推荐

{recommendation}

本轮训练得到的 pkl 文件只用于离线复核，不修改 runtime policy，也不接入 plus.py。

## 8. 后续建议

- 扩样本到 100 条以上；
- 保持 hard_negative / near_positive 数量均衡；
- 再评估 Logistic Regression / RandomForest / MLP；
- 对学习式方法增加独立测试集，而不只依赖交叉验证；
- 暂不直接接入工程主流程。
"""
    path.write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    summary_csv = Path(args.summary_csv)
    summary_json = Path(args.summary_json)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = read_csv_rows(summary_csv)
    summary = read_json(summary_json)
    dataset, numeric_features, categorical_features = build_dataset(rows)
    feature_columns = [*numeric_features, *categorical_features]
    rule_threshold = to_float(summary.get("recommended_threshold"), default=0.6)

    write_dataset_csv(output_dir / "learning_fusion_dataset.csv", dataset, feature_columns)
    comparison, predictions, cv_name = evaluate_models(
        dataset, numeric_features, categorical_features, rule_threshold, args.random_state
    )
    fit_and_save_models(
        dataset, numeric_features, categorical_features, output_dir / "trained_models", args.random_state
    )

    write_comparison_csv(output_dir / "learning_fusion_model_comparison.csv", comparison)
    best = choose_best_model(comparison)
    json_payload = {
        "schema": "learning_fusion_classifier.v1",
        "summary_csv": str(summary_csv),
        "dataset_size": len(dataset),
        "feature_columns": feature_columns,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "cv": cv_name,
        "rule_threshold": rule_threshold,
        "best_model": best,
        "recommended_for_integration": False,
        "recommendation_reason": "小样本 exploratory 结果，仅用于验证学习式融合潜力；暂不接入 plus.py。",
        "models": comparison,
    }
    (output_dir / "learning_fusion_model_comparison.json").write_text(
        json.dumps(json_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_report(
        output_dir / "learning_fusion_classifier_report.md",
        dataset,
        numeric_features,
        categorical_features,
        comparison,
        best,
        cv_name,
        rule_threshold,
    )

    print("========================================================================")
    print(f"dataset_size: {len(dataset)}")
    print(f"feature_columns: {feature_columns}")
    print(f"best_model: {best['method']}")
    print(f"best_model_false_hit_rate: {best['false_hit_rate']}")
    print(f"best_model_recall: {best['recall']}")
    print("recommended_for_integration: False")
    print(f"learning_fusion_classifier_report.md: {output_dir / 'learning_fusion_classifier_report.md'}")
    print("========================================================================")


if __name__ == "__main__":
    main()
