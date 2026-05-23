#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Train small fusion classifiers on public_train_1000 metadata.

This script trains sklearn-level classifiers only. It does not call Qwen,
TripoSR, Stable Fast 3D, Hunyuan3D, or modify plus.py/runtime policy.
"""

from __future__ import annotations

import argparse
import csv
import difflib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import warnings


OUTPUT_DIR = Path("paper_repro_outputs/cache_similarity_eval_public_train_1000")
METADATA_CSV = OUTPUT_DIR / "public_train_1000_metadata.csv"
FEATURE_CSV = OUTPUT_DIR / "public_train_1000_features.csv"
SPLIT_CSV = OUTPUT_DIR / "public_train_1000_split.csv"
COMPARISON_CSV = OUTPUT_DIR / "public_train_model_comparison.csv"
COMPARISON_JSON = OUTPUT_DIR / "public_train_model_comparison.json"
REPORT_MD = OUTPUT_DIR / "public_train_classifier_report.md"
MODEL_DIR = OUTPUT_DIR / "trained_models_public"
PREPARE_JSON = OUTPUT_DIR / "public_train_1000_prepare_report.json"
FEATURE_COLUMNS = [
    "text_score",
    "image_score",
    "fusion_score",
    "score_abs_diff",
    "score_max",
    "score_min",
]
IMAGE_PROXY_SCORE = {
    "positive": 0.95,
    "near_positive": 0.75,
    "hard_negative": 0.45,
    "negative": 0.10,
}
RANDOM_STATE = 42
RULE_THRESHOLD = 0.6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train public_train_1000 small fusion classifiers.")
    parser.add_argument("--metadata-csv", default=str(METADATA_CSV))
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--random-state", type=int, default=RANDOM_STATE)
    return parser.parse_args()


def read_metadata(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return [row for row in csv.DictReader(f) if row.get("sample_id")]


def normalize_text(text: str) -> str:
    return " ".join((text or "").lower().replace("_", " ").split())


def string_similarity(a: str, b: str) -> float:
    a = normalize_text(a)
    b = normalize_text(b)
    if not a or not b:
        return 0.0
    base = difflib.SequenceMatcher(None, a, b).ratio()
    tokens_a = set(a.split())
    tokens_b = set(b.split())
    if tokens_a and tokens_b:
        jaccard = len(tokens_a & tokens_b) / len(tokens_a | tokens_b)
    else:
        jaccard = 0.0
    return round(max(base, jaccard), 4)


def to_label(value: Any) -> int:
    return 1 if str(value).strip().lower() in {"1", "true", "yes"} else 0


def build_features(rows: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    features: List[Dict[str, Any]] = []
    for row in rows:
        sample_type = row.get("sample_type", "")
        candidate = row.get("candidate_cache_name", "")
        candidate_text = " ".join(
            part
            for part in [
                candidate,
                row.get("keyword", ""),
                row.get("object_category", ""),
            ]
            if part
        )
        query_text = " ".join(
            part
            for part in [
                row.get("query_text", ""),
                row.get("matched_dataset_class", ""),
            ]
            if part
        )
        text_score = string_similarity(query_text, candidate_text)
        image_score = IMAGE_PROXY_SCORE.get(sample_type, 0.0)
        fusion_score = round(0.5 * text_score + 0.5 * image_score, 4)
        score_abs_diff = round(abs(text_score - image_score), 4)
        score_max = round(max(text_score, image_score), 4)
        score_min = round(min(text_score, image_score), 4)
        item = {
            **row,
            "should_hit": to_label(row.get("should_hit")),
            "text_score": text_score,
            "image_score": image_score,
            "fusion_score": fusion_score,
            "score_abs_diff": score_abs_diff,
            "score_max": score_max,
            "score_min": score_min,
        }
        features.append(item)
    return features


def write_features(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    base_fields = [
        "sample_id",
        "roi_image_path",
        "candidate_cache_name",
        "sample_type",
        "should_hit",
        "object_category",
        "matched_dataset_class",
    ]
    fields = [*base_fields, *FEATURE_COLUMNS]
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})


def write_split_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    fields = ["sample_id", "split", "should_hit", "sample_type", "candidate_cache_name"]
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})


def calc_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    positives = int((y_true == 1).sum())
    negatives = int((y_true == 0).sum())
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4) if len(y_true) else 0.0,
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4) if len(y_true) else 0.0,
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4) if len(y_true) else 0.0,
        "false_hit_rate": round(fp / negatives, 4) if negatives else 0.0,
        "false_miss_rate": round(fn / positives, 4) if positives else 0.0,
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4) if len(y_true) else 0.0,
        "auto_hit_count": int(y_pred.sum()) if len(y_pred) else 0,
        "false_hit_count": fp,
        "false_miss_count": fn,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


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
            max_depth=5,
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
                        hidden_layer_sizes=(16,),
                        alpha=0.01,
                        max_iter=1000,
                        random_state=random_state,
                    ),
                ),
            ]
        ),
    }


def split_rows(rows: List[Dict[str, Any]], random_state: int) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    y = np.array([row["should_hit"] for row in rows])
    train_rows, temp_rows = train_test_split(
        rows,
        test_size=0.30,
        random_state=random_state,
        stratify=y,
    )
    temp_y = np.array([row["should_hit"] for row in temp_rows])
    val_rows, test_rows = train_test_split(
        temp_rows,
        test_size=0.50,
        random_state=random_state,
        stratify=temp_y,
    )
    for row in train_rows:
        row["split"] = "train"
    for row in val_rows:
        row["split"] = "val"
    for row in test_rows:
        row["split"] = "test"
    return train_rows, val_rows, test_rows


def xy(rows: List[Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray]:
    x = np.array([[float(row[col]) for col in FEATURE_COLUMNS] for row in rows], dtype=float)
    y = np.array([int(row["should_hit"]) for row in rows], dtype=int)
    return x, y


def evaluate_all(rows: List[Dict[str, Any]], random_state: int) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    train_rows, val_rows, test_rows = split_rows(rows, random_state)
    split_rows_all = [*train_rows, *val_rows, *test_rows]
    write_split_csv(SPLIT_CSV, split_rows_all)

    train_x, train_y = xy(train_rows)
    datasets = {
        "train": xy(train_rows),
        "val": xy(val_rows),
        "test": xy(test_rows),
    }
    comparison: List[Dict[str, Any]] = []

    for split, (x_part, y_part) in datasets.items():
        rule_pred = (x_part[:, FEATURE_COLUMNS.index("fusion_score")] >= RULE_THRESHOLD).astype(int)
        comparison.append({"method": "rule-fusion baseline", "split": split, **calc_metrics(y_part, rule_pred)})

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    filename_map = {
        "Logistic Regression": "logistic_regression.pkl",
        "RandomForest": "random_forest.pkl",
        "MLPClassifier": "mlp_classifier.pkl",
    }
    for method, model in make_models(random_state).items():
        model.fit(train_x, train_y)
        joblib.dump(model, MODEL_DIR / filename_map[method])
        for split, (x_part, y_part) in datasets.items():
            pred = model.predict(x_part)
            comparison.append({"method": method, "split": split, **calc_metrics(y_part, pred)})

    best_test = sorted(
        [row for row in comparison if row["split"] == "test"],
        key=lambda r: (1 if r["false_hit_rate"] == 0 else 0, r["recall"], r["f1"]),
        reverse=True,
    )[0]
    return comparison, best_test


def write_comparison(path: Path, rows: List[Dict[str, Any]]) -> None:
    fields = [
        "method",
        "split",
        "accuracy",
        "precision",
        "recall",
        "false_hit_rate",
        "false_miss_rate",
        "f1",
        "auto_hit_count",
        "false_hit_count",
        "false_miss_count",
        "tp",
        "tn",
        "fp",
        "fn",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})


def table(rows: List[Dict[str, Any]]) -> str:
    cols = ["method", "split", "accuracy", "precision", "recall", "false_hit_rate", "false_miss_rate", "f1", "false_hit_count"]
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(c, "")) for c in cols) + " |")
    return "\n".join(lines)


def write_empty_outputs(reason: str, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for path, fields in [
        (FEATURE_CSV, ["sample_id", *FEATURE_COLUMNS, "should_hit"]),
        (SPLIT_CSV, ["sample_id", "split", "should_hit"]),
        (COMPARISON_CSV, ["method", "split", "accuracy", "precision", "recall", "false_hit_rate", "false_miss_rate", "f1"]),
    ]:
        with path.open("w", encoding="utf-8-sig", newline="") as f:
            csv.DictWriter(f, fieldnames=fields).writeheader()
    payload = {
        "schema": "public_train_classifier.v1",
        "total_samples": 0,
        "failed_reason": reason,
        "recommended_for_integration": False,
    }
    COMPARISON_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    report = f"""# 公开数据集学习式图文融合训练初步报告

## 1. 实验目的

本实验计划使用公开数据集扩充训练样本，验证学习式融合在更大样本规模下是否比规则融合更稳定。公开数据用于训练，真实摄像头 ROI 后续作为测试集。

## 当前状态

训练未执行。

原因：{reason}

本轮没有调用 Qwen、TripoSR、Stable Fast 3D，没有修改 plus.py，也没有修改 runtime policy。
"""
    REPORT_MD.write_text(report, encoding="utf-8-sig")


def write_report(rows: List[Dict[str, Any]], comparison: List[Dict[str, Any]], best_test: Dict[str, Any]) -> None:
    count_by_sample_type = Counter(row.get("sample_type", "") for row in rows)
    count_by_candidate = Counter(row.get("candidate_cache_name", "") for row in rows)
    count_by_source = Counter(row.get("source_dataset", "") for row in rows)
    prepare_info: Dict[str, Any] = {}
    if PREPARE_JSON.exists():
        try:
            prepare_info = json.loads(PREPARE_JSON.read_text(encoding="utf-8"))
        except Exception:
            prepare_info = {}
    missing_classes = prepare_info.get("missing_classes", [])
    missing_glb = prepare_info.get("missing_glb", [])
    failed_download_reason = prepare_info.get("failed_download_reason", "")
    text = f"""# 公开数据集学习式图文融合训练初步报告

## 1. 实验目的

本实验用公开数据集扩充训练样本，验证学习式融合在更大样本规模下是否比规则融合更稳定。公开数据用于训练，真实摄像头 ROI 后续作为测试集。

## 2. 数据来源与构造方式

使用 Open Images / COCO 小规模检测框子集；根据候选缓存模型构造 positive / near_positive / hard_negative / negative。公开数据集没有天然 should_hit 标签，本实验根据候选缓存模型与类别映射构造标签。

## 3. 样本统计

- total_samples: {len(rows)}
- count_by_sample_type: {dict(count_by_sample_type)}
- count_by_candidate_cache_name: {dict(count_by_candidate)}
- count_by_source_dataset: {dict(count_by_source)}
- missing_classes: {missing_classes}
- missing_glb: {missing_glb}
- failed_download_reason: {failed_download_reason or "无"}

## 4. 特征说明

本轮不调用 Qwen，不调用 TripoSR，不使用深度视觉 embedding。text_score 使用字符串相似度；image_score 使用公开数据类别映射代理分数：

- positive = 0.95
- near_positive = 0.75
- hard_negative = 0.45
- negative = 0.10

因此本轮结果用于训练流程验证，不等同于最终真实图像 embedding 分数。

## 5. 模型与训练设置

- rule-fusion baseline
- Logistic Regression
- RandomForest
- MLPClassifier
- train / val / test = 70 / 15 / 15
- random_state = 42

## 6. 实验结果

{table(comparison)}

test split 最佳方法：{best_test.get("method")}，recall={best_test.get("recall")}，false_hit_rate={best_test.get("false_hit_rate")}。

## 7. 风险与限制

1. 公开数据和真实摄像头 ROI 存在 domain gap；
2. image_score 是弱代理分数，不是真实 embedding；
3. should_hit 标签由类别映射构造，仍需人工复核；
4. 不能直接把模型接入 plus.py；
5. 后续必须用真实摄像头 ROI 测试。

## 8. 当前结论

学习式模型是否稳定，需要以后续 real_test_150 迁移测试为准。本轮默认不推荐接入工程主流程。

recommended_for_integration = False

## 9. 下一步

1. 用真实摄像头 ROI 构造 real_test_150；
2. 用 public_train_1000 训练模型；
3. 在 real_test_150 上评估迁移效果；
4. 和规则融合 + 双阈值策略对比；
5. 暂不接入 plus.py。
"""
    REPORT_MD.write_text(text, encoding="utf-8-sig")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    global OUTPUT_DIR, METADATA_CSV, FEATURE_CSV, SPLIT_CSV, COMPARISON_CSV, COMPARISON_JSON, REPORT_MD, MODEL_DIR
    global PREPARE_JSON
    OUTPUT_DIR = output_dir
    METADATA_CSV = Path(args.metadata_csv)
    FEATURE_CSV = OUTPUT_DIR / "public_train_1000_features.csv"
    SPLIT_CSV = OUTPUT_DIR / "public_train_1000_split.csv"
    COMPARISON_CSV = OUTPUT_DIR / "public_train_model_comparison.csv"
    COMPARISON_JSON = OUTPUT_DIR / "public_train_model_comparison.json"
    REPORT_MD = OUTPUT_DIR / "public_train_classifier_report.md"
    MODEL_DIR = OUTPUT_DIR / "trained_models_public"
    PREPARE_JSON = OUTPUT_DIR / "public_train_1000_prepare_report.json"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    metadata = read_metadata(METADATA_CSV)
    if len(metadata) < 20 or len({row.get("should_hit") for row in metadata}) < 2:
        reason = (
            f"metadata insufficient for stratified training: rows={len(metadata)}. "
            "Run prepare_public_train_1000_dataset.py after installing/configuring FiftyOne."
        )
        write_empty_outputs(reason, OUTPUT_DIR)
        best_test = {"method": "N/A", "false_hit_rate": "N/A", "recall": "N/A"}
        print("=" * 72)
        print(f"train_script_path: {Path(__file__)}")
        print(f"public_train_total_samples: {len(metadata)}")
        print(f"best_model_on_test: {best_test['method']}")
        print(f"best_model_test_false_hit_rate: {best_test['false_hit_rate']}")
        print(f"best_model_test_recall: {best_test['recall']}")
        print("recommended_for_integration: False")
        print(f"public_train_classifier_report.md: {REPORT_MD}")
        print("=" * 72)
        return

    features = build_features(metadata)
    write_features(FEATURE_CSV, features)
    comparison, best_test = evaluate_all(features, args.random_state)
    write_comparison(COMPARISON_CSV, comparison)
    payload = {
        "schema": "public_train_classifier.v1",
        "total_samples": len(features),
        "feature_columns": FEATURE_COLUMNS,
        "best_model_on_test": best_test,
        "recommended_for_integration": False,
        "models": comparison,
    }
    COMPARISON_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(features, comparison, best_test)

    print("=" * 72)
    print(f"train_script_path: {Path(__file__)}")
    print(f"public_train_total_samples: {len(features)}")
    print(f"best_model_on_test: {best_test['method']}")
    print(f"best_model_test_false_hit_rate: {best_test['false_hit_rate']}")
    print(f"best_model_test_recall: {best_test['recall']}")
    print("recommended_for_integration: False")
    print(f"public_train_classifier_report.md: {REPORT_MD}")
    print("=" * 72)


if __name__ == "__main__":
    main()
