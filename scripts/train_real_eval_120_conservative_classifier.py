#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Train conservative learning-based fusion classifiers on real_eval_120.

real_eval_120 = v3_real_70 existing score features + real_challenge_50 score
features. This script is offline-only and does not touch plus.py or any 3D
generation pipeline.
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

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


DEFAULT_V3_DATASET = Path(
    "paper_repro_outputs/cache_similarity_eval_v3_real_70/learning_fusion_classifier/"
    "learning_fusion_dataset.csv"
)
DEFAULT_CHALLENGE_FEATURES = Path(
    "paper_repro_outputs/cache_similarity_eval_real_challenge_50/real_challenge_50_features.csv"
)
DEFAULT_OUTPUT_DIR = Path(
    "paper_repro_outputs/cache_similarity_eval_real_eval_120/conservative_classifier"
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
RULE_STRONG = 0.78
RULE_WEAK = 0.70


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train conservative classifiers on real_eval_120.")
    parser.add_argument("--v3-dataset", default=str(DEFAULT_V3_DATASET))
    parser.add_argument("--challenge-features", default=str(DEFAULT_CHALLENGE_FEATURES))
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


def to_label(value: Any) -> int:
    return 1 if str(value).strip().lower() in {"1", "true", "yes", "hit"} else 0


def read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def normalize_rows(rows: List[Dict[str, str]], source: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for idx, row in enumerate(rows):
        text_score = to_float(row.get("text_score"))
        image_score = to_float(row.get("image_score"), default=text_score)
        fusion_score = to_float(row.get("fusion_score"), default=float("nan"))
        if math.isnan(fusion_score):
            fusion_score = to_float(row.get("fused_score"), default=0.5 * text_score + 0.5 * image_score)
        out.append(
            {
                "sample_id": row.get("sample_id") or row.get("index") or f"{source}_{idx:04d}",
                "image": row.get("image") or row.get("roi_image_path", ""),
                "should_hit": to_label(row.get("should_hit")),
                "text_score": text_score,
                "image_score": image_score,
                "fusion_score": fusion_score,
                "score_abs_diff": abs(text_score - image_score),
                "score_max": max(text_score, image_score),
                "score_min": min(text_score, image_score),
                "source": source,
            }
        )
    return out


def xy(rows: List[Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray]:
    x = np.array([[float(row[col]) for col in FEATURE_COLUMNS] for row in rows], dtype=float)
    y = np.array([int(row["should_hit"]) for row in rows], dtype=int)
    return x, y


def make_models(random_state: int) -> Dict[str, Any]:
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
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    MLPClassifier(
                        hidden_layer_sizes=(8,),
                        alpha=0.02,
                        max_iter=1500,
                        random_state=random_state,
                    ),
                ),
            ]
        ),
    }


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
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
    }


def dual_threshold_metrics(scores: np.ndarray, y: np.ndarray, weak: float, strong: float) -> Dict[str, Any]:
    auto = scores >= strong
    review = (scores >= weak) & (scores < strong)
    miss = scores < weak
    auto_pred = auto.astype(int)
    m = metrics(y, auto_pred)
    positives = int((y == 1).sum())
    review_true = int(((y == 1) & review).sum())
    recall_if_review_accepted = round((int(((y == 1) & auto).sum()) + review_true) / positives, 4) if positives else 0.0
    m.update(
        {
            "review_count": int(review.sum()),
            "review_rate": round(float(review.mean()), 4) if len(review) else 0.0,
            "miss_count": int(miss.sum()),
            "recall_if_review_accepted": recall_if_review_accepted,
        }
    )
    return m


def choose_threshold(probs: np.ndarray, y: np.ndarray) -> float:
    candidates = [round(x, 2) for x in np.linspace(0.5, 0.95, 10)]
    scored = []
    for threshold in candidates:
        pred = (probs >= threshold).astype(int)
        m = metrics(y, pred)
        scored.append((m["false_hit_rate"], -m["recall"], -m["f1"], threshold))
    scored.sort()
    return scored[0][3]


def evaluate(rows: List[Dict[str, Any]], output_dir: Path, random_state: int) -> List[Dict[str, Any]]:
    x, y = xy(rows)
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    results: List[Dict[str, Any]] = []

    rule_scores = x[:, FEATURE_COLUMNS.index("fusion_score")]
    rule_m = dual_threshold_metrics(rule_scores, y, RULE_WEAK, RULE_STRONG)
    results.append(
        {
            "method": "rule-fusion dual-threshold",
            "threshold": f"weak={RULE_WEAK}, strong={RULE_STRONG}",
            **rule_m,
        }
    )

    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    output_dir.joinpath("trained_models").mkdir(parents=True, exist_ok=True)
    model_files = {
        "Logistic Regression": "logistic_regression.pkl",
        "RandomForest": "random_forest.pkl",
        "MLPClassifier": "mlp_classifier.pkl",
    }

    for name, model in make_models(random_state).items():
        probs = np.zeros(len(rows), dtype=float)
        for train_idx, test_idx in splitter.split(x, y):
            model.fit(x[train_idx], y[train_idx])
            if hasattr(model, "predict_proba"):
                probs[test_idx] = model.predict_proba(x[test_idx])[:, 1]
            else:
                probs[test_idx] = model.predict(x[test_idx])
        threshold = choose_threshold(probs, y)
        pred = (probs >= threshold).astype(int)
        m = metrics(y, pred)
        # Use a narrow review band below the conservative threshold for reporting.
        review_weak = max(0.0, threshold - 0.08)
        dt = dual_threshold_metrics(probs, y, review_weak, threshold)
        results.append(
            {
                "method": name,
                "threshold": threshold,
                **m,
                "review_rate": dt["review_rate"],
                "review_count": dt["review_count"],
                "recall_if_review_accepted": dt["recall_if_review_accepted"],
            }
        )
        # Fit final model on all real_eval_120 data for offline artifact only.
        model.fit(x, y)
        joblib.dump(model, output_dir / "trained_models" / model_files[name])
    return results


def write_dataset(path: Path, rows: List[Dict[str, Any]]) -> None:
    fields = ["sample_id", "image", "source", "should_hit", *FEATURE_COLUMNS]
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def write_results(path: Path, rows: List[Dict[str, Any]]) -> None:
    fields = [
        "method",
        "threshold",
        "accuracy",
        "precision",
        "recall",
        "false_hit_rate",
        "review_rate",
        "false_miss_rate",
        "f1",
        "auto_hit_count",
        "false_hit_count",
        "false_miss_count",
        "review_count",
        "recall_if_review_accepted",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def report(output_dir: Path, rows: List[Dict[str, Any]], results: List[Dict[str, Any]], blocked_reason: str = "") -> None:
    path = output_dir / "real_eval_120_conservative_classifier_report.md"
    if blocked_reason:
        text = f"""# real_eval_120 保守型学习式融合模型报告

## 当前状态

训练未执行。

原因：{blocked_reason}

需要先补充 real_challenge_50 特征文件：`paper_repro_outputs/cache_similarity_eval_real_challenge_50/real_challenge_50_features.csv`。

本脚本没有修改 plus.py，没有调用 Qwen / TripoSR / Stable Fast 3D，也没有接入工程主流程。
"""
        path.write_text(text, encoding="utf-8-sig")
        return

    rule = next(r for r in results if r["method"] == "rule-fusion dual-threshold")
    best = sorted(
        results,
        key=lambda r: (1 if r["false_hit_rate"] == 0 else 0, r["recall"], r["f1"]),
        reverse=True,
    )[0]
    conclusion = (
        "学习式模型在 real_eval_120 上保持 false_hit_rate 接近 0，具备进一步验证价值，但仍不建议直接接入 plus.py。"
        if best["method"] != rule["method"] and best["false_hit_rate"] <= 0.03
        else "当前仍推荐规则融合 + 双阈值策略；学习式模型需要更多真实挑战样本和独立测试集验证，暂不接入 plus.py。"
    )
    table = "\n".join(
        [
            "| method | recall | false_hit_rate | review_rate | false_miss_rate | f1 |",
            "| --- | --- | --- | --- | --- | --- |",
            *[
                f"| {r['method']} | {r['recall']} | {r['false_hit_rate']} | {r.get('review_rate', '')} | {r['false_miss_rate']} | {r['f1']} |"
                for r in results
            ],
        ]
    )
    text = f"""# real_eval_120 保守型学习式融合模型报告

## 1. 实验目的

本实验将 v3_real_70 与 real_challenge_50 合并为 real_eval_120，训练保守型学习式融合模型。训练目标不是单纯提高 recall，而是在尽量降低 false_hit_rate 的前提下观察学习式方法是否优于规则融合 + 双阈值策略。

## 2. 数据组成

- total_samples = {len(rows)}
- source 分布 = {dict(Counter(row['source'] for row in rows))}
- should_hit 分布 = {dict(Counter(row['should_hit'] for row in rows))}

## 3. 特征设置

只使用线上可获得分数特征：

- text_score
- image_score
- fusion_score
- score_abs_diff
- score_max
- score_min

不使用 sample_type / category / object_category 等元数据。

## 4. 实验结果

{table}

## 5. 必须回答的问题

1. 学习式模型在 real_eval_120 上是否能保持 false_hit_rate 接近 0：见上表 false_hit_rate。
2. 是否比规则融合 + 双阈值更好：需要同时满足低 false_hit_rate 和更高 recall，不能只看 recall。
3. 是否仍然不建议接入 plus.py：是。当前模型只作为离线探索，不接入工程主流程。

## 6. 当前结论

{conclusion}
"""
    path.write_text(text, encoding="utf-8-sig")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    v3_rows = normalize_rows(read_csv(Path(args.v3_dataset)), "v3_real_70")
    challenge_rows = normalize_rows(read_csv(Path(args.challenge_features)), "real_challenge_50")
    json_path = output_dir / "real_eval_120_conservative_classifier_results.json"

    if len(challenge_rows) < 50:
        reason = f"real_challenge_50 features insufficient: found {len(challenge_rows)}, expected 50."
        report(output_dir, [], [], reason)
        json_path.write_text(
            json.dumps(
                {
                    "status": "blocked",
                    "reason": reason,
                    "v3_count": len(v3_rows),
                    "challenge_count": len(challenge_rows),
                    "recommended_for_integration": False,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print("=" * 72)
        print("training_executed: False")
        print(f"blocked_reason: {reason}")
        print(f"report: {output_dir / 'real_eval_120_conservative_classifier_report.md'}")
        print("=" * 72)
        return

    rows = v3_rows + challenge_rows
    write_dataset(output_dir / "real_eval_120_dataset.csv", rows)
    results = evaluate(rows, output_dir, args.random_state)
    write_results(output_dir / "real_eval_120_conservative_classifier_results.csv", results)
    report(output_dir, rows, results)
    best = sorted(
        results,
        key=lambda r: (1 if r["false_hit_rate"] == 0 else 0, r["recall"], r["f1"]),
        reverse=True,
    )[0]
    json_path.write_text(
        json.dumps(
            {
                "status": "completed",
                "total_samples": len(rows),
                "best_model": best,
                "recommended_for_integration": False,
                "results": results,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print("=" * 72)
    print("training_executed: True")
    print(f"total_samples: {len(rows)}")
    print(f"best_model: {best['method']}")
    print(f"best_model_false_hit_rate: {best['false_hit_rate']}")
    print(f"best_model_recall: {best['recall']}")
    print("recommended_for_integration: False")
    print(f"report: {output_dir / 'real_eval_120_conservative_classifier_report.md'}")
    print("=" * 72)


if __name__ == "__main__":
    main()
