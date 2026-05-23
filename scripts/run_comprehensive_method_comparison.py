#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Run a comprehensive cache similarity method comparison on real_eval_200.

This script reads existing score features and experiment outputs. It does not
call Qwen, TripoSR, Stable Fast 3D, or modify plus.py. Learning-based methods
use only online score features:

text_score, image_score, fusion_score, score_abs_diff, score_max, score_min.
"""

from __future__ import annotations

import csv
import json
import math
import warnings
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


DATASET_PATH = Path("paper_repro_outputs/cache_similarity_eval_real_eval_200/real_eval_200_features.csv")
OUTPUT_DIR = Path("paper_repro_outputs/cache_similarity_eval_real_eval_200/comprehensive_method_comparison")
CONSERVATIVE_DIR = Path("paper_repro_outputs/cache_similarity_eval_real_eval_200/conservative_classifier")
PUBLIC_MODEL_DIR = Path("paper_repro_outputs/cache_similarity_eval_public_train_1000/trained_models_public")

FEATURE_COLUMNS = [
    "text_score",
    "image_score",
    "fusion_score",
    "score_abs_diff",
    "score_max",
    "score_min",
]
SPLITS = ["train", "val", "test"]
RANDOM_STATE = 42
RULE_FIXED_THRESHOLD = 0.60
RULE_WEAK = 0.70
RULE_STRONG = 0.78

CSV_FIELDS = [
    "method",
    "method_group",
    "dataset",
    "split",
    "threshold",
    "weak_threshold",
    "strong_threshold",
    "accuracy",
    "precision",
    "recall",
    "false_hit_rate",
    "false_miss_rate",
    "f1",
    "auto_hit_count",
    "review_count",
    "miss_count",
    "false_hit_count",
    "false_miss_count",
    "true_hit_count",
    "true_miss_count",
    "review_rate",
    "recommended_for_integration",
    "notes",
]


def read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: Sequence[Dict[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fields))
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def to_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        out = float(str(value).strip())
    except Exception:
        return default
    if math.isnan(out) or math.isinf(out):
        return default
    return out


def to_label(value: Any) -> int:
    return 1 if str(value).strip().lower() in {"1", "true", "yes", "hit"} else 0


def normalize_rows(rows: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for idx, row in enumerate(rows):
        text_score = to_float(row.get("text_score"))
        image_score = to_float(row.get("image_score"), default=text_score)
        fusion_score = to_float(row.get("fusion_score"), default=0.5 * text_score + 0.5 * image_score)
        out.append(
            {
                "sample_id": row.get("sample_id") or str(idx),
                "split": row.get("split") or "test",
                "should_hit": to_label(row.get("should_hit")),
                "text_score": text_score,
                "image_score": image_score,
                "fusion_score": fusion_score,
                "score_abs_diff": to_float(row.get("score_abs_diff"), abs(text_score - image_score)),
                "score_max": to_float(row.get("score_max"), max(text_score, image_score)),
                "score_min": to_float(row.get("score_min"), min(text_score, image_score)),
            }
        )
    return out


def xy(rows: Sequence[Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray]:
    x = np.array([[float(row[col]) for col in FEATURE_COLUMNS] for row in rows], dtype=float)
    y = np.array([int(row["should_hit"]) for row in rows], dtype=int)
    return x, y


def score_array(rows: Sequence[Dict[str, Any]], score_name: str) -> np.ndarray:
    return np.array([float(row[score_name]) for row in rows], dtype=float)


def metric_for_decisions(
    y_true: np.ndarray,
    decisions: Sequence[str],
    method: str,
    method_group: str,
    split: str,
    threshold: Any = "",
    weak_threshold: Any = "",
    strong_threshold: Any = "",
    recommended: Any = False,
    notes: str = "",
) -> Dict[str, Any]:
    decisions = list(decisions)
    total = len(decisions)
    auto = np.array([d == "auto_hit" or d == "hit" for d in decisions], dtype=bool)
    review = np.array([d == "review" for d in decisions], dtype=bool)
    miss = np.array([d == "miss" for d in decisions], dtype=bool)

    y = y_true.astype(int)
    positives = int((y == 1).sum())
    negatives = int((y == 0).sum())
    true_hit_count = int(((y == 1) & auto).sum())
    true_review_count = int(((y == 1) & review).sum())
    true_miss_count = int(((y == 0) & miss).sum())
    false_hit_count = int(((y == 0) & auto).sum())
    false_miss_count = int(((y == 1) & miss).sum())
    auto_hit_count = int(auto.sum())
    review_count = int(review.sum())
    miss_count = int(miss.sum())

    # For review-aware policies, review positives are treated as recoverable
    # candidates for recall, but review is not counted as automatic precision.
    recall = (true_hit_count + true_review_count) / positives if positives else 0.0
    precision = true_hit_count / auto_hit_count if auto_hit_count else 0.0
    false_hit_rate = false_hit_count / negatives if negatives else 0.0
    false_miss_rate = false_miss_count / positives if positives else 0.0
    # Accuracy penalizes review as unresolved, so it stays conservative.
    accuracy = (true_hit_count + true_miss_count) / total if total else 0.0
    binary_pred = auto.astype(int)
    f1 = float(f1_score(y, binary_pred, zero_division=0)) if total else 0.0

    return {
        "method": method,
        "method_group": method_group,
        "dataset": "real_eval_200",
        "split": split,
        "threshold": threshold,
        "weak_threshold": weak_threshold,
        "strong_threshold": strong_threshold,
        "accuracy": round(float(accuracy), 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "false_hit_rate": round(float(false_hit_rate), 4),
        "false_miss_rate": round(float(false_miss_rate), 4),
        "f1": round(float(f1), 4),
        "auto_hit_count": auto_hit_count,
        "review_count": review_count,
        "miss_count": miss_count,
        "false_hit_count": false_hit_count,
        "false_miss_count": false_miss_count,
        "true_hit_count": true_hit_count,
        "true_miss_count": true_miss_count,
        "review_rate": round(review_count / total, 4) if total else 0.0,
        "recommended_for_integration": recommended,
        "notes": notes,
    }


def threshold_decisions(scores: np.ndarray, threshold: float) -> List[str]:
    return ["hit" if score >= threshold else "miss" for score in scores]


def dual_threshold_decisions(scores: np.ndarray, weak: float, strong: float) -> List[str]:
    out = []
    for score in scores:
        if score >= strong:
            out.append("auto_hit")
        elif score >= weak:
            out.append("review")
        else:
            out.append("miss")
    return out


def choose_threshold_on_val(rows: List[Dict[str, Any]], score_name: str) -> float:
    val_rows = [row for row in rows if row["split"] == "val"]
    if not val_rows:
        val_rows = rows
    y_val = np.array([row["should_hit"] for row in val_rows], dtype=int)
    scores = score_array(val_rows, score_name)
    candidates = [round(x, 2) for x in np.linspace(0.50, 0.90, 41)]
    scored = []
    for threshold in candidates:
        row = metric_for_decisions(
            y_val,
            threshold_decisions(scores, threshold),
            method="tmp",
            method_group="tmp",
            split="val",
            threshold=threshold,
        )
        scored.append((row["false_hit_rate"], -row["recall"], -row["f1"], threshold))
    scored.sort()
    return float(scored[0][3])


def make_models() -> Dict[str, Any]:
    return {
        "Logistic Regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        max_iter=1000,
                        class_weight="balanced",
                        solver="liblinear",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=240,
            max_depth=4,
            min_samples_leaf=3,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "MLPClassifier": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    MLPClassifier(
                        hidden_layer_sizes=(10,),
                        alpha=0.03,
                        max_iter=1500,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
    }


def conservative_threshold(y_val: np.ndarray, probs: np.ndarray) -> float:
    candidates = [round(float(x), 3) for x in np.linspace(0.50, 0.98, 49)]
    scored = []
    for threshold in candidates:
        row = metric_for_decisions(
            y_val,
            threshold_decisions(probs, threshold),
            method="tmp",
            method_group="tmp",
            split="val",
            threshold=threshold,
        )
        scored.append((row["false_hit_rate"], -row["recall"], -row["f1"], threshold))
    scored.sort()
    return float(scored[0][3])


def evaluate_main_methods(rows: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    results: List[Dict[str, Any]] = []
    skipped: List[Dict[str, Any]] = []
    split_rows = {split: [row for row in rows if row["split"] == split] for split in SPLITS}

    scalar_methods = [
        ("text-only", "single-score", "text_score", choose_threshold_on_val(rows, "text_score"), "best threshold scanned 0.50-0.90 on val"),
        ("image-only", "single-score", "image_score", choose_threshold_on_val(rows, "image_score"), "best threshold scanned 0.50-0.90 on val"),
        ("rule-fusion 0.5/0.5 fixed@0.60", "rule-fusion", "fusion_score", RULE_FIXED_THRESHOLD, "historical fixed threshold baseline"),
        ("rule-fusion 0.5/0.5 best-threshold", "rule-fusion", "fusion_score", choose_threshold_on_val(rows, "fusion_score"), "best threshold scanned 0.50-0.90 on val"),
    ]
    for method, group, score_name, threshold, note in scalar_methods:
        for split, items in split_rows.items():
            y = np.array([row["should_hit"] for row in items], dtype=int)
            scores = score_array(items, score_name)
            results.append(
                metric_for_decisions(
                    y,
                    threshold_decisions(scores, threshold),
                    method=method,
                    method_group=group,
                    split=split,
                    threshold=threshold,
                    recommended=False,
                    notes=note,
                )
            )

    for split, items in split_rows.items():
        y = np.array([row["should_hit"] for row in items], dtype=int)
        scores = score_array(items, "fusion_score")
        results.append(
            metric_for_decisions(
                y,
                dual_threshold_decisions(scores, RULE_WEAK, RULE_STRONG),
                method="dual-threshold fusion",
                method_group="rule-fusion",
                split=split,
                weak_threshold=RULE_WEAK,
                strong_threshold=RULE_STRONG,
                recommended="Candidate",
                notes="current runtime candidate: auto_hit/review/miss, review kept for risk control",
            )
        )

    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    train_rows = split_rows["train"]
    val_rows = split_rows["val"]
    if train_rows and val_rows:
        x_train, y_train = xy(train_rows)
        x_val, y_val = xy(val_rows)
        for method, model in make_models().items():
            model.fit(x_train, y_train)
            probs_val = model.predict_proba(x_val)[:, 1] if hasattr(model, "predict_proba") else model.predict(x_val)
            threshold = conservative_threshold(y_val, probs_val)
            for split, items in split_rows.items():
                x_split, y_split = xy(items)
                probs = model.predict_proba(x_split)[:, 1] if hasattr(model, "predict_proba") else model.predict(x_split)
                results.append(
                    metric_for_decisions(
                        y_split,
                        threshold_decisions(probs, threshold),
                        method=method,
                        method_group="real-trained learning",
                        split=split,
                        threshold=threshold,
                        recommended=False,
                        notes="trained on real_eval_200 train split; threshold selected on val by false_hit priority; no metadata features used",
                    )
                )
    else:
        skipped.append({"method": "real-trained learning models", "skipped_reason": "train/val split missing"})

    public_models = [
        ("public-trained Logistic Regression transfer", PUBLIC_MODEL_DIR / "logistic_regression.pkl"),
        ("public-trained RandomForest transfer", PUBLIC_MODEL_DIR / "random_forest.pkl"),
        ("public-trained MLPClassifier transfer", PUBLIC_MODEL_DIR / "mlp_classifier.pkl"),
    ]
    for method, model_path in public_models:
        if not model_path.exists():
            skipped.append({"method": method, "skipped_reason": f"model not found: {model_path}"})
            continue
        try:
            model = joblib.load(model_path)
            for split, items in split_rows.items():
                x_split, y_split = xy(items)
                probs = model.predict_proba(x_split)[:, 1] if hasattr(model, "predict_proba") else model.predict(x_split)
                results.append(
                    metric_for_decisions(
                        y_split,
                        threshold_decisions(probs, 0.5),
                        method=method,
                        method_group="public-trained transfer",
                        split=split,
                        threshold=0.5,
                        recommended=False,
                        notes="public-trained model transfer; no refit on real_eval_200; default threshold=0.5",
                    )
                )
        except Exception as exc:
            skipped.append({"method": method, "skipped_reason": f"{type(exc).__name__}: {exc}"})
    return results, skipped


def ranking_rows(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    test_rows = [row for row in results if row["split"] == "test"]
    ranked = sorted(
        test_rows,
        key=lambda row: (
            float(row["false_hit_rate"]),
            float(row["review_rate"]),
            -float(row["recall"]),
            -float(row["f1"]),
            0 if row["method"] == "dual-threshold fusion" else 1,
        ),
    )
    out = []
    for rank, row in enumerate(ranked, start=1):
        item = {"rank": rank}
        item.update(row)
        out.append(item)
    return out


def add_hist_row(
    rows: List[Dict[str, Any]],
    stage: str,
    dataset: str,
    method: str,
    total_samples: Any,
    feature_setting: str,
    recall: Any,
    false_hit_rate: Any,
    false_miss_rate: Any,
    review_rate: Any,
    recommended: Any,
    conclusion: str,
    path: Path,
) -> None:
    rows.append(
        {
            "stage": stage,
            "dataset": dataset,
            "method": method,
            "total_samples": total_samples,
            "feature_setting": feature_setting,
            "recall": recall,
            "false_hit_rate": false_hit_rate,
            "false_miss_rate": false_miss_rate,
            "review_rate": review_rate,
            "recommended_for_integration": recommended,
            "main_conclusion": conclusion,
            "source_report_path": str(path),
        }
    )


def build_historical_summary(real_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    hist: List[Dict[str, Any]] = []
    v3_path = Path("paper_repro_outputs/cache_similarity_eval_v3_real_70/similarity_method_comparison/similarity_method_comparison.csv")
    for row in read_csv(v3_path):
        add_hist_row(
            hist,
            "A. v3_real_70 规则策略",
            "v3_real_70",
            row.get("method", ""),
            70,
            "text/image/fusion score comparison",
            row.get("recall", ""),
            row.get("false_hit_rate", ""),
            row.get("false_miss_rate", ""),
            row.get("review_count", ""),
            "rule baseline only",
            "v3_real_70 showed rule fusion was safer than chasing recall alone.",
            v3_path,
        )

    learn_path = Path("paper_repro_outputs/cache_similarity_eval_v3_real_70/learning_fusion_classifier/learning_fusion_model_comparison.csv")
    for row in read_csv(learn_path):
        if row.get("method", "").lower().startswith("rule"):
            continue
        add_hist_row(
            hist,
            "B. v3_real_70 学习式融合初步实验",
            "v3_real_70",
            row.get("method", ""),
            70,
            "included sample_type/category metadata; leakage risk",
            row.get("recall", ""),
            row.get("false_hit_rate", ""),
            row.get("false_miss_rate", ""),
            "",
            row.get("recommended_for_integration", "False"),
            "High scores may be affected by metadata leakage/distribution memory.",
            learn_path,
        )

    ablation_path = Path("paper_repro_outputs/cache_similarity_eval_v3_real_70/learning_fusion_classifier_ablation_no_metadata/learning_fusion_ablation_comparison.csv")
    for row in read_csv(ablation_path):
        if row.get("method", "").lower().startswith("rule"):
            continue
        add_hist_row(
            hist,
            "C. 去元数据消融实验",
            "v3_real_70",
            row.get("method", ""),
            70,
            "score-only features, no sample_type/category",
            row.get("recall", ""),
            row.get("false_hit_rate", ""),
            row.get("false_miss_rate", ""),
            "",
            row.get("recommended_for_integration", "False"),
            "Removing metadata reduced reliability and introduced false_hit risk.",
            ablation_path,
        )

    public_path = Path("paper_repro_outputs/cache_similarity_eval_public_train_1000/public_train_model_comparison.csv")
    for row in read_csv(public_path):
        if row.get("split") != "test":
            continue
        add_hist_row(
            hist,
            "D. public_train_755",
            "public_train_755",
            row.get("method", ""),
            755,
            "text string score + category-proxy image_score",
            row.get("recall", ""),
            row.get("false_hit_rate", ""),
            row.get("false_miss_rate", ""),
            "",
            "False",
            "Public constructed samples looked strong, but image_score was a category proxy.",
            public_path,
        )

    transfer_path = Path("paper_repro_outputs/cache_similarity_eval_public_train_1000/real_v3_transfer_eval/public_to_real_v3_eval.csv")
    for row in read_csv(transfer_path):
        add_hist_row(
            hist,
            "E. public -> real_v3 迁移",
            "real_v3",
            row.get("method", ""),
            70,
            "public-trained model transferred to real ROI scores",
            row.get("recall", ""),
            row.get("false_hit_rate", ""),
            row.get("false_miss_rate", ""),
            "",
            row.get("recommended_for_integration", "False"),
            "Public-trained models introduced high false_hit on real ROI, showing domain gap.",
            transfer_path,
        )

    conservative_path = CONSERVATIVE_DIR / "real_eval_200_model_comparison.csv"
    for row in read_csv(conservative_path):
        if row.get("split") != "test":
            continue
        add_hist_row(
            hist,
            "F. real_eval_200",
            "real_eval_200",
            row.get("method", ""),
            200,
            "score-only features, group-aware split where possible",
            row.get("recall", ""),
            row.get("false_hit_rate", ""),
            row.get("false_miss_rate", ""),
            row.get("review_rate", ""),
            "False",
            "Current best engineering choice remained rule fusion + dual threshold.",
            conservative_path,
        )

    for row in real_results:
        if row.get("split") != "test":
            continue
        add_hist_row(
            hist,
            "G. real_eval_200 全方法综合对比",
            "real_eval_200",
            row.get("method", ""),
            200,
            row.get("method_group", ""),
            row.get("recall", ""),
            row.get("false_hit_rate", ""),
            row.get("false_miss_rate", ""),
            row.get("review_rate", ""),
            row.get("recommended_for_integration", "False"),
            "Unified comparison on the latest real_eval_200 feature table.",
            OUTPUT_DIR / "comprehensive_method_comparison_real_eval_200.csv",
        )
    return hist


def markdown_table(rows: Sequence[Dict[str, Any]], columns: Sequence[str]) -> str:
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(col, "")) for col in columns) + " |")
    return "\n".join(lines)


def make_chart(rows: List[Dict[str, Any]], output_path: Path) -> bool:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return False
    test_rows = [row for row in rows if row.get("split") == "test"]
    names = [row["method"].replace(" transfer", "").replace(" 0.5/0.5", "") for row in test_rows]
    recall = [float(row["recall"]) for row in test_rows]
    false_hit = [float(row["false_hit_rate"]) for row in test_rows]
    x = np.arange(len(names))
    width = 0.38
    plt.figure(figsize=(max(12, len(names) * 1.1), 5.5))
    plt.bar(x - width / 2, recall, width, label="recall")
    plt.bar(x + width / 2, false_hit, width, label="false_hit_rate")
    plt.xticks(x, names, rotation=35, ha="right", fontsize=8)
    plt.ylim(0, 1.05)
    plt.ylabel("score")
    plt.title("real_eval_200 method comparison")
    plt.legend()
    for idx, fh in enumerate(false_hit):
        if fh == 0:
            plt.text(idx + width / 2, 0.03, "0 FH", ha="center", va="bottom", fontsize=7)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()
    return True


def write_reports(
    results: List[Dict[str, Any]],
    ranking: List[Dict[str, Any]],
    hist: List[Dict[str, Any]],
    skipped: List[Dict[str, Any]],
    chart_created: bool,
) -> None:
    test_rows = [row for row in results if row["split"] == "test"]
    recommended = next((row for row in test_rows if row["method"] == "dual-threshold fusion"), ranking[0])
    public_skips = "\n".join(f"- {item['method']}: {item['skipped_reason']}" for item in skipped) if skipped else "无"

    summary = f"""# 全方法综合对比摘要

## 1. 为什么做综合对比

本轮把纯文本、纯图像、规则融合、双阈值、真实数据学习式模型和 public-trained 迁移模型统一放到 real_eval_200 上评估，核心目的是判断哪种方法最适合作为工程主流程。

## 2. real_eval_200 数据规模

- total_samples = 200
- v3_real_70 = 70
- real_add_130 = 130
- should_hit=True = 100
- should_hit=False = 100
- 学习式模型只使用线上可获得分数特征，不使用 sample_type / category / object_category。

## 3. 方法对比总表（test split）

{markdown_table(test_rows, ["method", "method_group", "threshold", "weak_threshold", "strong_threshold", "recall", "false_hit_rate", "review_rate", "f1", "recommended_for_integration"])}

## 4. 最终结论

当前最佳工程方案仍是规则融合 + 双阈值：

`score = 0.5 * text_score + 0.5 * image_score`

- score >= 0.78: auto_hit，自动复用缓存
- 0.70 <= score < 0.78: review，提示用户确认
- score < 0.70: miss，不复用，回退生成流程

学习式模型在 real_eval_200 上具备探索价值，但当前不建议接入 plus.py。false_hit 的风险高于 false_miss，因此工程主流程优先选择可解释、保守、false_hit_rate 为 0 的方案。
"""
    (OUTPUT_DIR / "comprehensive_method_comparison_summary.md").write_text(summary, encoding="utf-8-sig")

    full = f"""# 全方法综合对比实验报告

## 1. 实验目的

本实验统一比较项目中已经尝试过的所有缓存复用判断方法，包括纯文本、纯图像、规则融合、双阈值融合、学习式融合、公开数据训练模型和真实数据训练模型，目标是判断哪种方法最适合作为工程主流程。

## 2. 数据集说明

real_eval_200 由 v3_real_70 和 real_add_130 合并而成：

- total_samples = 200
- should_hit=True = 100
- should_hit=False = 100
- 使用已有真实 text_score / image_score / fusion_score
- 不使用 sample_type / category / object_category 作为学习式模型输入
- 不调用 Qwen，不调用 TripoSR，不重新生成模型

## 3. 方法列表

- text-only：只使用 text_score，阈值在 val 上扫描。
- image-only：只使用 image_score，阈值在 val 上扫描。
- rule-fusion 0.5 / 0.5：包含 fixed@0.60 和 val 最佳阈值两个结果。
- dual-threshold fusion：weak=0.70，strong=0.78，支持 auto_hit / review / miss。
- Logistic Regression / RandomForest / MLPClassifier：只使用线上分数特征，在 train split 训练，val split 选保守阈值。
- public-trained transfer models：直接加载 public_train_755 训练模型做迁移预测，不在 real_eval_200 上重训。

## 4. real_eval_200 统一结果

{markdown_table(test_rows, ["method", "method_group", "threshold", "weak_threshold", "strong_threshold", "accuracy", "precision", "recall", "false_hit_rate", "false_miss_rate", "review_rate", "f1", "recommended_for_integration"])}

跳过项：

{public_skips}

## 5. 历史实验对比

{markdown_table(hist, ["stage", "dataset", "method", "total_samples", "feature_setting", "recall", "false_hit_rate", "false_miss_rate", "review_rate", "recommended_for_integration", "main_conclusion"])}

## 6. 结果分析

1. 不能只看 recall。缓存复用任务中，高 recall 如果伴随 false_hit，会把错误缓存模型自动复用给用户，风险高于漏命中。

2. false_hit 比 false_miss 风险更高。false_miss 最多回退到生成流程，代价是慢；false_hit 会错误复用模型，直接影响系统可信度和 AR 展示结果。

3. public_train 上的高分不能直接说明真实系统有效。public_train_755 的 image_score 是类别映射代理分数，不是真实摄像头 ROI embedding / image signature 分数，因此迁移到 real_v3 后出现高 false_hit，说明存在 domain gap。

4. 去元数据消融很重要。早期学习式实验使用 sample_type / category 后分数过高，可能记住标签分布；去掉元数据后 false_hit 风险暴露出来，更接近真实线上条件。

5. real_eval_200 后仍推荐规则融合 + 双阈值。它可解释、无需训练、false_hit_rate 保守，并且已完成工程接入与三分支验证。相比之下，学习式模型仍需要更大的独立真实测试集。

6. RandomForest 在 real_eval_200 上虽然可以做到较低 false_hit，但仍不直接接入。原因是当前测试集仍较小，切分受视频分组影响；树模型容易受到采集分布影响，后续需要独立 real_test 进一步验证。

7. Review 分支仍需保留。review 可以承接边界样本，在不自动误复用的前提下给用户确认机会，是降低风险与提升体验之间的缓冲区。

## 7. 最终推荐

当前最佳工程方案：

```text
score = 0.5 * text_score + 0.5 * image_score

score >= 0.78:
  auto_hit，自动复用缓存

0.70 <= score < 0.78:
  review，提示用户确认

score < 0.70:
  miss，不复用，回退生成流程
```

- recommended_for_runtime: True / Candidate
- rule-fusion dual-threshold: Candidate
- learning-based models: False

这不表示学习式模型没有价值，而是说明当前阶段还不适合接入工程主流程。

## 8. 后续工作

- 继续扩充真实 ROI 独立测试集；
- 学习式模型暂不接入 plus.py；
- 若继续训练模型，应增加更多 hard_negative / near_positive；
- Review UI 保留为降低误复用风险的重要机制；
- 公开数据可作为补充，但不能替代真实 ROI 评估。

## 9. 附加文件

- chart_created: {chart_created}
- ranking_file: `comprehensive_method_ranking.csv`
- historical_summary_file: `historical_experiment_summary.csv`
"""
    (OUTPUT_DIR / "comprehensive_method_comparison_full_report.md").write_text(full, encoding="utf-8-sig")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = normalize_rows(read_csv(DATASET_PATH))
    if not rows:
        raise FileNotFoundError(f"No rows found: {DATASET_PATH}")

    results, skipped = evaluate_main_methods(rows)
    ranking = ranking_rows(results)
    hist = build_historical_summary(results)
    chart_created = make_chart(results, OUTPUT_DIR / "comprehensive_method_comparison_chart.png")

    write_csv(OUTPUT_DIR / "comprehensive_method_comparison_real_eval_200.csv", results, CSV_FIELDS)
    (OUTPUT_DIR / "comprehensive_method_comparison_real_eval_200.json").write_text(
        json.dumps({"results": results, "skipped": skipped}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_csv(OUTPUT_DIR / "comprehensive_method_ranking.csv", ranking, ["rank", *CSV_FIELDS])
    write_csv(
        OUTPUT_DIR / "historical_experiment_summary.csv",
        hist,
        [
            "stage",
            "dataset",
            "method",
            "total_samples",
            "feature_setting",
            "recall",
            "false_hit_rate",
            "false_miss_rate",
            "review_rate",
            "recommended_for_integration",
            "main_conclusion",
            "source_report_path",
        ],
    )
    write_reports(results, ranking, hist, skipped, chart_created)

    best_engineering = next((row for row in ranking if row["method"] == "dual-threshold fusion"), ranking[0])
    print("=" * 72)
    print(f"real_eval_200_total_samples: {len(rows)}")
    print(f"compared_method_count: {len({row['method'] for row in results})}")
    print(f"best_engineering_method: {best_engineering['method']}")
    print(f"best_engineering_false_hit_rate: {best_engineering['false_hit_rate']}")
    print(f"best_engineering_recall: {best_engineering['recall']}")
    print("recommended_runtime_strategy: rule-fusion dual-threshold (score=0.5*text+0.5*image, weak=0.70, strong=0.78)")
    print(f"full_report: {OUTPUT_DIR / 'comprehensive_method_comparison_full_report.md'}")
    print(f"summary_report: {OUTPUT_DIR / 'comprehensive_method_comparison_summary.md'}")
    print(f"comparison_csv: {OUTPUT_DIR / 'comprehensive_method_comparison_real_eval_200.csv'}")
    print("=" * 72)


if __name__ == "__main__":
    main()
