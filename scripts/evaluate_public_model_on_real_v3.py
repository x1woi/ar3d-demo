#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Evaluate public-trained fusion classifiers on real v3_real_70 scores.

This script loads existing public_train_1000 sklearn models and evaluates them
on existing v3_real_70 feature scores. It does not retrain models, call Qwen,
call TripoSR, run Stable Fast 3D, or modify plus.py.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


DEFAULT_MODEL_DIR = Path(
    "paper_repro_outputs/cache_similarity_eval_public_train_1000/trained_models_public"
)
DEFAULT_FEATURE_CSV = Path(
    "paper_repro_outputs/cache_similarity_eval_v3_real_70/learning_fusion_classifier/"
    "learning_fusion_dataset.csv"
)
DEFAULT_SUMMARY_CSV = Path("paper_repro_outputs/cache_similarity_eval_v3_real_70/summary.csv")
DEFAULT_PUBLIC_PREPARE_JSON = Path(
    "paper_repro_outputs/cache_similarity_eval_public_train_1000/public_train_1000_prepare_report.json"
)
DEFAULT_OUTPUT_DIR = Path(
    "paper_repro_outputs/cache_similarity_eval_public_train_1000/real_v3_transfer_eval"
)
FEATURE_COLUMNS = [
    "text_score",
    "image_score",
    "fusion_score",
    "score_abs_diff",
    "score_max",
    "score_min",
]
RULE_THRESHOLD = 0.6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate public-trained classifiers on real v3_real_70.")
    parser.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR))
    parser.add_argument("--real-feature-csv", default=str(DEFAULT_FEATURE_CSV))
    parser.add_argument("--summary-csv", default=str(DEFAULT_SUMMARY_CSV))
    parser.add_argument("--public-prepare-json", default=str(DEFAULT_PUBLIC_PREPARE_JSON))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
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


def to_label(value: Any) -> int:
    return 1 if str(value).strip().lower() in {"1", "true", "yes", "hit"} else 0


def read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def load_real_features(feature_csv: Path, summary_csv: Path) -> Tuple[List[Dict[str, Any]], str]:
    rows = read_csv(feature_csv)
    source = str(feature_csv)
    if not rows:
        rows = read_csv(summary_csv)
        source = str(summary_csv)
    if not rows:
        raise FileNotFoundError("No real v3 feature csv or summary csv found.")

    dataset: List[Dict[str, Any]] = []
    for idx, row in enumerate(rows):
        text_score = to_float(row.get("text_score"))
        image_score = to_float(row.get("image_score"), default=text_score)
        fusion_score = to_float(row.get("fusion_score"), default=float("nan"))
        if math.isnan(fusion_score):
            fusion_score = to_float(row.get("fused_score"), default=0.5 * text_score + 0.5 * image_score)
        item = {
            "index": idx,
            "image": row.get("image", ""),
            "should_hit": to_label(row.get("should_hit")),
            "text_score": text_score,
            "image_score": image_score,
            "fusion_score": fusion_score,
            "score_abs_diff": abs(text_score - image_score),
            "score_max": max(text_score, image_score),
            "score_min": min(text_score, image_score),
        }
        dataset.append(item)
    return dataset, source


def xy(dataset: List[Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray]:
    x = np.array([[float(item[col]) for col in FEATURE_COLUMNS] for item in dataset], dtype=float)
    y = np.array([int(item["should_hit"]) for item in dataset], dtype=int)
    return x, y


def calc_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    positives = int((y_true == 1).sum())
    negatives = int((y_true == 0).sum())
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "false_hit_rate": round(fp / negatives, 4) if negatives else 0.0,
        "false_miss_rate": round(fn / positives, 4) if positives else 0.0,
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "auto_hit_count": int(y_pred.sum()),
        "false_hit_count": fp,
        "false_miss_count": fn,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def load_models(model_dir: Path) -> Dict[str, Any]:
    paths = {
        "public-trained Logistic Regression": model_dir / "logistic_regression.pkl",
        "public-trained RandomForest": model_dir / "random_forest.pkl",
        "public-trained MLPClassifier": model_dir / "mlp_classifier.pkl",
    }
    models = {}
    missing = []
    for name, path in paths.items():
        if path.exists():
            models[name] = joblib.load(path)
        else:
            missing.append(str(path))
    if missing:
        raise FileNotFoundError(f"Missing trained model files: {missing}")
    return models


def evaluate(dataset: List[Dict[str, Any]], model_dir: Path) -> List[Dict[str, Any]]:
    x, y = xy(dataset)
    rows: List[Dict[str, Any]] = []

    rule_pred = (x[:, FEATURE_COLUMNS.index("fusion_score")] >= RULE_THRESHOLD).astype(int)
    rows.append({"method": "rule-fusion baseline", **calc_metrics(y, rule_pred)})

    for name, model in load_models(model_dir).items():
        pred = model.predict(x)
        rows.append({"method": name, **calc_metrics(y, pred)})
    return rows


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
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
        "tp",
        "tn",
        "fp",
        "fn",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def pick_best(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    return sorted(
        rows,
        key=lambda r: (1 if r["false_hit_rate"] == 0 else 0, r["recall"], r["f1"]),
        reverse=True,
    )[0]


def table(rows: List[Dict[str, Any]]) -> str:
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
        lines.append("| " + " | ".join(str(row.get(col, "")) for col in cols) + " |")
    return "\n".join(lines)


def write_report(
    path: Path,
    rows: List[Dict[str, Any]],
    dataset: List[Dict[str, Any]],
    source_csv: str,
    public_info: Dict[str, Any],
    best: Dict[str, Any],
) -> bool:
    rule = next(row for row in rows if row["method"] == "rule-fusion baseline")
    logistic = next((row for row in rows if "Logistic" in row["method"]), {})
    public_counts = public_info.get("count_by_sample_type", {})
    real_counts = Counter(item["should_hit"] for item in dataset)
    public_model_has_false_hit = any(
        row["method"] != "rule-fusion baseline" and row["false_hit_count"] > 0 for row in rows
    )
    public_model_improves_safely = any(
        row["method"] != "rule-fusion baseline"
        and row["false_hit_rate"] == 0
        and row["recall"] > rule["recall"]
        for row in rows
    )
    recommended = False
    if public_model_has_false_hit:
        conclusion = "公开数据训练模型迁移到真实 v3_real_70 后引入 false_hit 风险，暂不适合接入工程主流程，当前仍推荐规则融合 + 双阈值策略。"
    elif public_model_improves_safely:
        conclusion = "公开数据训练模型在真实 v3_real_70 上保持 false_hit_rate=0 且 recall 有提升，说明学习式融合具有进一步验证价值；但仍需 real_test_150 独立测试，不直接接入 plus.py。"
    else:
        conclusion = "公开数据训练模型未明显优于规则融合，当前仍推荐规则融合 + 双阈值策略。"

    text = f"""# Public Train → Real v3 迁移评估报告

## 1. 实验目的

public_train_755 在公开数据构造样本上表现很好，但其 image_score 是类别代理分数。本实验把 public 训练模型迁移到真实 v3_real_70 分数上测试，验证是否存在 domain gap。

## 2. 训练来源

- source_dataset = {public_info.get("source_dataset_used", "open-images-v7")}
- total_samples = {public_info.get("total_samples", 755)}
- positive = {public_counts.get("positive", "unknown")}
- near_positive = {public_counts.get("near_positive", "unknown")}
- hard_negative = {public_counts.get("hard_negative", "unknown")}
- negative = {public_counts.get("negative", "unknown")}
- image_score 为类别映射代理分数，不是真实图像 embedding 分数。

## 3. 测试来源

- 使用已有 v3_real_70 结果：`{source_csv}`
- real_test_size = {len(dataset)}
- should_hit=True = {real_counts.get(1, 0)}
- should_hit=False = {real_counts.get(0, 0)}
- 未重新调用 Qwen / TripoSR，未重新生成模型。

## 4. 特征设置

只使用：

- text_score
- image_score
- fusion_score
- score_abs_diff
- score_max
- score_min

不使用：

- sample_type
- category
- source_dataset
- object_category

## 5. 实验结果

{table(rows)}

## 6. 结果分析

1. public-trained Logistic Regression 在 real_v3 上 false_hit_rate = {logistic.get("false_hit_rate", "N/A")}，recall = {logistic.get("recall", "N/A")}。
2. rule-fusion baseline false_hit_rate = {rule["false_hit_rate"]}，recall = {rule["recall"]}。
3. 如果 public-trained 模型 recall 下降或 false_hit 上升，说明公开数据类别代理分数与真实摄像头 ROI 分数之间存在 domain gap。
4. public_train 阶段的高分受到类别代理 image_score 影响，不能直接等价为真实图像相似度效果。
5. 工程接入仍应以真实 ROI 测试结果为准。

## 7. 当前结论

{conclusion}

recommended_for_integration = {recommended}
"""
    path.write_text(text, encoding="utf-8-sig")
    return recommended


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset, source_csv = load_real_features(Path(args.real_feature_csv), Path(args.summary_csv))
    rows = evaluate(dataset, Path(args.model_dir))
    best = pick_best(rows)
    csv_path = output_dir / "public_to_real_v3_eval.csv"
    json_path = output_dir / "public_to_real_v3_eval.json"
    report_path = output_dir / "public_to_real_v3_eval_report.md"
    write_csv(csv_path, rows)
    public_info = read_json(Path(args.public_prepare_json))
    recommended = write_report(report_path, rows, dataset, source_csv, public_info, best)
    rule = next(row for row in rows if row["method"] == "rule-fusion baseline")
    payload = {
        "schema": "public_to_real_v3_transfer_eval.v1",
        "model_dir": str(args.model_dir),
        "real_feature_source": source_csv,
        "real_test_size": len(dataset),
        "feature_columns": FEATURE_COLUMNS,
        "best_model_on_real_v3": best,
        "rule_fusion": rule,
        "recommended_for_integration": recommended,
        "results": rows,
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=" * 72)
    print(f"real_test_size: {len(dataset)}")
    print(f"best_model_on_real_v3: {best['method']}")
    print(f"best_model_real_false_hit_rate: {best['false_hit_rate']}")
    print(f"best_model_real_recall: {best['recall']}")
    print(f"rule_fusion_false_hit_rate: {rule['false_hit_rate']}")
    print(f"rule_fusion_recall: {rule['recall']}")
    print(f"recommended_for_integration: {recommended}")
    print(f"public_to_real_v3_eval_report.md: {report_path}")
    print("=" * 72)


if __name__ == "__main__":
    main()
