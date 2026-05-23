#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Train conservative learning-based fusion classifiers on real_eval_200.

real_eval_200 = existing v3_real_70 score features + real_add_130 score
features. This script is offline-only: it does not modify plus.py and does not
call any 3D generation pipeline.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import warnings
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


DEFAULT_V3_DATASET = Path(
    "paper_repro_outputs/cache_similarity_eval_v3_real_70/learning_fusion_classifier/"
    "learning_fusion_dataset.csv"
)
DEFAULT_REAL_ADD_FEATURES = Path("paper_repro_outputs/cache_similarity_eval_real_add_130/real_add_130_features.csv")
DEFAULT_OUTPUT_DIR = Path("paper_repro_outputs/cache_similarity_eval_real_eval_200/conservative_classifier")

FEATURE_COLUMNS = [
    "text_score",
    "image_score",
    "fusion_score",
    "score_abs_diff",
    "score_max",
    "score_min",
]
RANDOM_STATE = 42
RULE_WEAK = 0.70
RULE_STRONG = 0.78


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train conservative classifiers on real_eval_200.")
    parser.add_argument("--v3-dataset", default=str(DEFAULT_V3_DATASET))
    parser.add_argument("--real-add-features", default=str(DEFAULT_REAL_ADD_FEATURES))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--random-state", type=int, default=RANDOM_STATE)
    return parser.parse_args()


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
        source_video = row.get("source_video") or ""
        sample_id = row.get("sample_id") or row.get("index") or f"{source}_{idx:04d}"
        group_id = source_video.strip() if source_video.strip() else f"{source}_{sample_id}"
        out.append(
            {
                "sample_id": sample_id,
                "image": row.get("image") or row.get("roi_image_path", ""),
                "source": source,
                "source_video": source_video,
                "group_id": group_id,
                "should_hit": to_label(row.get("should_hit")),
                "text_score": text_score,
                "image_score": image_score,
                "fusion_score": fusion_score,
                "score_abs_diff": abs(text_score - image_score),
                "score_max": max(text_score, image_score),
                "score_min": min(text_score, image_score),
            }
        )
    return out


def xy(rows: Sequence[Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray]:
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
            n_estimators=240,
            max_depth=4,
            min_samples_leaf=3,
            class_weight="balanced",
            random_state=random_state,
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
                        random_state=random_state,
                    ),
                ),
            ]
        ),
    }


def metric_dict(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
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
    m = metric_dict(y, auto.astype(int))
    positives = int((y == 1).sum())
    review_true = int(((y == 1) & review).sum())
    auto_true = int(((y == 1) & auto).sum())
    m.update(
        {
            "review_count": int(review.sum()),
            "review_rate": round(float(review.mean()), 4) if len(review) else 0.0,
            "miss_count": int(miss.sum()),
            "recall_if_review_accepted": round((auto_true + review_true) / positives, 4) if positives else 0.0,
        }
    )
    return m


def choose_conservative_threshold(y_true: np.ndarray, probs: np.ndarray) -> float:
    thresholds = [round(float(x), 3) for x in np.linspace(0.50, 0.98, 49)]
    candidates = []
    for threshold in thresholds:
        pred = (probs >= threshold).astype(int)
        m = metric_dict(y_true, pred)
        candidates.append((m["false_hit_rate"], -m["recall"], -m["f1"], threshold))
    candidates.sort()
    return candidates[0][3]


def split_rows(rows: List[Dict[str, Any]], random_state: int) -> Tuple[List[int], List[int], List[int], str]:
    y = np.array([int(row["should_hit"]) for row in rows], dtype=int)
    groups = np.array([str(row.get("group_id") or row["sample_id"]) for row in rows])
    unique_groups = set(groups)

    if len(unique_groups) >= 12:
        try:
            gss = GroupShuffleSplit(n_splits=1, train_size=0.70, random_state=random_state)
            train_idx, hold_idx = next(gss.split(np.zeros(len(rows)), y, groups))
            hold_y = y[hold_idx]
            hold_groups = groups[hold_idx]
            gss2 = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=random_state + 1)
            val_rel, test_rel = next(gss2.split(np.zeros(len(hold_idx)), hold_y, hold_groups))
            return (
                list(train_idx),
                list(hold_idx[val_rel]),
                list(hold_idx[test_rel]),
                "group-aware split by source_video/group_id; sizes may differ slightly from 140/30/30",
            )
        except Exception:
            pass

    indices = np.arange(len(rows))
    train_idx, hold_idx = train_test_split(
        indices,
        train_size=140,
        stratify=y,
        random_state=random_state,
    )
    hold_y = y[hold_idx]
    val_idx, test_idx = train_test_split(
        hold_idx,
        train_size=30,
        test_size=30,
        stratify=hold_y,
        random_state=random_state,
    )
    return (
        list(train_idx),
        list(val_idx),
        list(test_idx),
        "stratified sample split; source_video grouping was unavailable or insufficient, so leakage risk is reported",
    )


def rows_by_idx(rows: List[Dict[str, Any]], idx: List[int]) -> List[Dict[str, Any]]:
    return [rows[i] for i in idx]


def evaluate(rows: List[Dict[str, Any]], output_dir: Path, random_state: int) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    train_idx, val_idx, test_idx, split_note = split_rows(rows, random_state)
    split_map = {i: "train" for i in train_idx}
    split_map.update({i: "val" for i in val_idx})
    split_map.update({i: "test" for i in test_idx})
    for idx, row in enumerate(rows):
        row["split"] = split_map.get(idx, "unused")

    train_rows = rows_by_idx(rows, train_idx)
    val_rows = rows_by_idx(rows, val_idx)
    test_rows = rows_by_idx(rows, test_idx)
    x_train, y_train = xy(train_rows)
    x_val, y_val = xy(val_rows)
    x_test, y_test = xy(test_rows)

    output_dir.joinpath("trained_models").mkdir(parents=True, exist_ok=True)
    results: List[Dict[str, Any]] = []

    for split_name, split_rows_items in [("train", train_rows), ("val", val_rows), ("test", test_rows)]:
        x_split, y_split = xy(split_rows_items)
        scores = x_split[:, FEATURE_COLUMNS.index("fusion_score")]
        m = dual_threshold_metrics(scores, y_split, RULE_WEAK, RULE_STRONG)
        results.append(
            {
                "method": "rule-fusion dual-threshold",
                "split": split_name,
                "threshold": f"weak={RULE_WEAK}, strong={RULE_STRONG}",
                **m,
            }
        )

    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    model_files = {
        "Logistic Regression": "logistic_regression.pkl",
        "RandomForest": "random_forest.pkl",
        "MLPClassifier": "mlp_classifier.pkl",
    }
    for name, model in make_models(random_state).items():
        model.fit(x_train, y_train)
        if hasattr(model, "predict_proba"):
            val_probs = model.predict_proba(x_val)[:, 1]
        else:
            val_probs = model.predict(x_val)
        threshold = choose_conservative_threshold(y_val, val_probs)
        for split_name, x_split, y_split in [
            ("train", x_train, y_train),
            ("val", x_val, y_val),
            ("test", x_test, y_test),
        ]:
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(x_split)[:, 1]
            else:
                probs = model.predict(x_split)
            pred = (probs >= threshold).astype(int)
            m = metric_dict(y_split, pred)
            review_weak = max(0.0, threshold - 0.08)
            dt = dual_threshold_metrics(probs, y_split, review_weak, threshold)
            results.append(
                {
                    "method": name,
                    "split": split_name,
                    "threshold": threshold,
                    **m,
                    "review_count": dt["review_count"],
                    "review_rate": dt["review_rate"],
                    "recall_if_review_accepted": dt["recall_if_review_accepted"],
                }
            )
        joblib.dump(model, output_dir / "trained_models" / model_files[name])

    split_stats = {
        "split_note": split_note,
        "train_size": len(train_rows),
        "val_size": len(val_rows),
        "test_size": len(test_rows),
        "train_label_counts": dict(Counter(row["should_hit"] for row in train_rows)),
        "val_label_counts": dict(Counter(row["should_hit"] for row in val_rows)),
        "test_label_counts": dict(Counter(row["should_hit"] for row in test_rows)),
    }
    return results, split_stats


def write_dataset(path: Path, rows: List[Dict[str, Any]]) -> None:
    fields = ["sample_id", "image", "source", "source_video", "group_id", "split", "should_hit", *FEATURE_COLUMNS]
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def write_results(path: Path, results: List[Dict[str, Any]]) -> None:
    fields = [
        "method",
        "split",
        "threshold",
        "accuracy",
        "precision",
        "recall",
        "false_hit_rate",
        "false_miss_rate",
        "f1",
        "auto_hit_count",
        "false_hit_count",
        "false_miss_count",
        "review_count",
        "review_rate",
        "recall_if_review_accepted",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in results:
            writer.writerow({field: row.get(field, "") for field in fields})


def choose_best(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    test_results = [row for row in results if row.get("split") == "test"]
    return sorted(
        test_results,
        key=lambda row: (
            row["false_hit_rate"],
            -row["recall"],
            -row["f1"],
            0 if row["method"] == "rule-fusion dual-threshold" else 1,
        ),
    )[0]


def write_blocked_report(output_dir: Path, reason: str, v3_count: int, real_add_count: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    text = f"""# real_eval_200 保守型学习式融合模型报告

## 当前状态

训练未执行。

blocked_reason: {reason}

- v3_real_70 rows: {v3_count}
- real_add_130 rows: {real_add_count}

只有 `real_add_130_features.csv` 达到 130 条真实特征后，才会合并形成 real_eval_200 并训练模型。
"""
    (output_dir / "real_eval_200_conservative_classifier_report.md").write_text(text, encoding="utf-8-sig")
    (output_dir / "real_eval_200_conservative_classifier_results.json").write_text(
        json.dumps(
            {
                "status": "blocked",
                "reason": reason,
                "v3_count": v3_count,
                "real_add_count": real_add_count,
                "training_executed": False,
                "recommended_for_integration": False,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def write_report(output_dir: Path, rows: List[Dict[str, Any]], results: List[Dict[str, Any]], split_stats: Dict[str, Any]) -> Dict[str, Any]:
    best = choose_best(results)
    rule_test = next(row for row in results if row["method"] == "rule-fusion dual-threshold" and row["split"] == "test")
    recommended_for_integration = False
    keep_review = True
    if best["method"] != "rule-fusion dual-threshold" and best["false_hit_rate"] <= 0.03 and best["recall"] > rule_test["recall"]:
        best_note = "学习式模型在测试集上有进一步验证价值，但仍需独立真实测试集确认，暂不接入 plus.py。"
    else:
        best_note = "当前最佳工程方案仍是规则融合 + 双阈值策略；学习式模型暂不接入 plus.py。"

    test_results = [row for row in results if row["split"] == "test"]
    table = "\n".join(
        [
            "| method | threshold | recall | false_hit_rate | review_rate | false_miss_rate | f1 |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
            *[
                f"| {row['method']} | {row.get('threshold','')} | {row['recall']} | {row['false_hit_rate']} | "
                f"{row.get('review_rate','')} | {row['false_miss_rate']} | {row['f1']} |"
                for row in test_results
            ],
        ]
    )
    source_counts = dict(Counter(row["source"] for row in rows))
    label_counts = dict(Counter(row["should_hit"] for row in rows))
    text = f"""# real_eval_200 保守型学习式融合模型报告

## 1. 实验目的

本实验将已有 v3_real_70 与新增 real_add_130 真实 ROI 特征合并为 real_eval_200，训练保守型学习式图文融合分类器。目标不是单纯提高 recall，而是在 false_hit_rate 接近 0 的前提下，观察学习式模型是否比规则融合 + 双阈值更适合工程接入。

## 2. 数据组成

- total_samples: {len(rows)}
- source_counts: {source_counts}
- should_hit_counts: {label_counts}
- split: train={split_stats['train_size']}, val={split_stats['val_size']}, test={split_stats['test_size']}
- split_note: {split_stats['split_note']}

## 3. 特征设置

只使用线上可获得的分数特征：

- text_score
- image_score
- fusion_score
- score_abs_diff
- score_max
- score_min

不使用 sample_type / category / object_category 作为模型输入，避免标签泄漏或分布记忆。

## 4. 测试集结果

{table}

## 5. 必须回答的问题

1. real_eval_200 上学习式模型是否能保持 false_hit_rate 接近 0？
   - 见测试集 `false_hit_rate`。模型选择以 false_hit_rate 为第一优先级。

2. 是否比规则融合 + 双阈值更好？
   - 需要同时满足更低或接近 0 的 false_hit_rate，以及更高 recall。不能只看 recall。

3. 是否建议接入 plus.py？
   - recommended_for_integration = {recommended_for_integration}
   - 当前仍不建议直接接入 plus.py。

4. 是否仍需要保留 review 分支？
   - keep_review = {keep_review}
   - review 分支仍建议保留，用于处理边界样本并降低误复用风险。

5. 当前最佳工程方案是什么？
   - {best_note}
   - 当前候选工程策略仍是：`score = 0.5 * text_score + 0.5 * image_score`，`score >= 0.78` 自动复用，`0.70 <= score < 0.78` 进入 review，`score < 0.70` 不复用。

## 6. 当前结论

本报告用于判断学习式融合是否有工程接入价值。若学习式模型在测试集上出现 false_hit，则不应接入主流程；若保持 false_hit_rate 接近 0 且 recall 明显提升，也仍需更多真实独立测试集验证后再考虑接入。
"""
    report_path = output_dir / "real_eval_200_conservative_classifier_report.md"
    report_path.write_text(text, encoding="utf-8-sig")

    payload = {
        "status": "completed",
        "total_samples": len(rows),
        "split_stats": split_stats,
        "best_model": best,
        "rule_test": rule_test,
        "recommended_for_integration": recommended_for_integration,
        "keep_review": keep_review,
        "results": results,
    }
    (output_dir / "real_eval_200_conservative_classifier_results.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return payload


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    v3_rows = normalize_rows(read_csv(Path(args.v3_dataset)), "v3_real_70")
    real_add_rows = normalize_rows(read_csv(Path(args.real_add_features)), "real_add_130")

    if len(v3_rows) < 70 or len(real_add_rows) < 130:
        reason = f"features incomplete: v3_rows={len(v3_rows)} expected>=70, real_add_rows={len(real_add_rows)} expected=130"
        write_blocked_report(output_dir, reason, len(v3_rows), len(real_add_rows))
        print("=" * 72)
        print("training_executed: False")
        print(f"blocked_reason: {reason}")
        print(f"report: {output_dir / 'real_eval_200_conservative_classifier_report.md'}")
        print("=" * 72)
        return

    rows = v3_rows + real_add_rows
    if len(rows) != 200:
        reason = f"real_eval_200 size mismatch: got {len(rows)}, expected 200"
        write_blocked_report(output_dir, reason, len(v3_rows), len(real_add_rows))
        print("=" * 72)
        print("training_executed: False")
        print(f"blocked_reason: {reason}")
        print(f"report: {output_dir / 'real_eval_200_conservative_classifier_report.md'}")
        print("=" * 72)
        return

    results, split_stats = evaluate(rows, output_dir, int(args.random_state))
    write_dataset(Path("paper_repro_outputs/cache_similarity_eval_real_eval_200/real_eval_200_features.csv"), rows)
    write_results(output_dir / "real_eval_200_model_comparison.csv", results)
    payload = write_report(output_dir, rows, results, split_stats)
    best = payload["best_model"]

    print("=" * 72)
    print("training_executed: True")
    print(f"real_eval_200_total_samples: {len(rows)}")
    print(f"best_model: {best['method']}")
    print(f"best_model_false_hit_rate: {best['false_hit_rate']}")
    print(f"best_model_recall: {best['recall']}")
    print("recommended_for_integration: False")
    print(f"report: {output_dir / 'real_eval_200_conservative_classifier_report.md'}")
    print("=" * 72)


if __name__ == "__main__":
    main()
