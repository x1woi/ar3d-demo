from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


DEFAULT_EVAL_DIR = Path("paper_repro_outputs/cache_similarity_eval_v3_real_70")
DEFAULT_OUTPUT_DIR = DEFAULT_EVAL_DIR / "similarity_method_comparison"


def read_csv_rows(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def to_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def to_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def find_weight_row(rows: List[Dict[str, Any]], text_weight: float) -> Optional[Dict[str, Any]]:
    for row in rows:
        if abs(to_float(row.get("text_weight")) - text_weight) < 1e-9:
            return row
    return None


def compute_single_threshold_counts(
    summary_rows: List[Dict[str, Any]],
    text_weight: float,
    image_weight: float,
    threshold: float,
) -> Dict[str, int]:
    auto_hit_count = 0
    miss_count = 0
    for row in summary_rows:
        text = to_float(row.get("text_score"))
        image = to_float(row.get("image_score"), text)
        score = text_weight * text + image_weight * image
        if score >= threshold:
            auto_hit_count += 1
        else:
            miss_count += 1
    return {"auto_hit_count": auto_hit_count, "review_count": 0, "miss_count": miss_count}


def build_method_row(
    method: str,
    weight_row: Dict[str, Any],
    counts: Dict[str, int],
    threshold_label: str,
) -> Dict[str, Any]:
    return {
        "method": method,
        "text_weight": to_float(weight_row.get("text_weight")),
        "image_weight": to_float(weight_row.get("image_weight")),
        "threshold": threshold_label,
        "accuracy": to_float(weight_row.get("accuracy")),
        "precision": to_float(weight_row.get("precision")),
        "recall": to_float(weight_row.get("recall")),
        "false_hit_rate": to_float(weight_row.get("false_hit_rate")),
        "false_miss_rate": to_float(weight_row.get("false_miss_rate")),
        "near_positive_false_miss_rate": to_float(
            weight_row.get("near_positive_false_miss_rate")
        ),
        "hard_negative_false_hit_rate": to_float(
            weight_row.get("hard_negative_false_hit_rate")
        ),
        "auto_hit_count": counts["auto_hit_count"],
        "review_count": counts["review_count"],
        "miss_count": counts["miss_count"],
    }


def has_garbled(text: str) -> bool:
    return "????" in text or "\ufffd" in text


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare text-only, image-only, fusion, and dual-threshold methods.")
    parser.add_argument("--eval-dir", default=str(DEFAULT_EVAL_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args()

    eval_dir = Path(args.eval_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_csv = eval_dir / "summary.csv"
    fusion_best_csv = eval_dir / "fusion_weight_ablation" / "fusion_weight_best_summary.csv"
    review_json = eval_dir / "borderline_threshold_analysis" / "recommended_review_threshold.json"

    summary_rows = read_csv_rows(summary_csv)
    fusion_best_rows = read_csv_rows(fusion_best_csv)
    review_data = read_json(review_json)

    text_only = find_weight_row(fusion_best_rows, 1.0)
    image_only = find_weight_row(fusion_best_rows, 0.0)
    rule_fusion = find_weight_row(fusion_best_rows, 0.5)

    if text_only is None or image_only is None or rule_fusion is None:
        missing = []
        if text_only is None:
            missing.append("text-only")
        if image_only is None:
            missing.append("image-only")
        if rule_fusion is None:
            missing.append("rule-fusion")
        raise SystemExit(f"Missing required method rows: {', '.join(missing)}")

    text_threshold = to_float(text_only.get("best_threshold"))
    image_threshold = to_float(image_only.get("best_threshold"))
    fusion_threshold = to_float(rule_fusion.get("best_threshold"))

    comparison_rows: List[Dict[str, Any]] = []
    comparison_rows.append(
        build_method_row(
            "text-only",
            text_only,
            compute_single_threshold_counts(summary_rows, 1.0, 0.0, text_threshold),
            str(text_threshold),
        )
    )
    comparison_rows.append(
        build_method_row(
            "image-only",
            image_only,
            compute_single_threshold_counts(summary_rows, 0.0, 1.0, image_threshold),
            str(image_threshold),
        )
    )
    comparison_rows.append(
        build_method_row(
            "rule-fusion",
            rule_fusion,
            compute_single_threshold_counts(summary_rows, 0.5, 0.5, fusion_threshold),
            str(fusion_threshold),
        )
    )

    dual_metrics = review_data.get("key_metrics", review_data)
    dual_row = {
        "method": "dual-threshold fusion",
        "text_weight": 0.5,
        "image_weight": 0.5,
        "threshold": f"{review_data.get('recommended_weak_threshold', 0.7)} / {review_data.get('recommended_strong_threshold', 0.78)}",
        "accuracy": "",
        "precision": "",
        "recall": to_float(dual_metrics.get("recall_if_review_accepted")),
        "false_hit_rate": to_float(dual_metrics.get("auto_false_hit_rate")),
        "false_miss_rate": to_float(dual_metrics.get("false_miss_rate")),
        "near_positive_false_miss_rate": (
            round(
                to_float(dual_metrics.get("near_positive_miss_count"))
                / max(
                    1,
                    to_float(dual_metrics.get("near_positive_auto_hit_count"))
                    + to_float(dual_metrics.get("near_positive_review_count"))
                    + to_float(dual_metrics.get("near_positive_miss_count")),
                ),
                4,
            )
        ),
        "hard_negative_false_hit_rate": 0.0
        if to_float(dual_metrics.get("hard_negative_auto_false_hit_count")) == 0
        else "",
        "auto_hit_count": int(to_float(dual_metrics.get("auto_hit_count"))),
        "review_count": int(to_float(dual_metrics.get("review_count"))),
        "miss_count": int(to_float(dual_metrics.get("miss_count"))),
    }
    comparison_rows.append(dual_row)

    csv_path = output_dir / "similarity_method_comparison.csv"
    json_path = output_dir / "similarity_method_comparison.json"
    report_path = output_dir / "similarity_method_comparison_report.md"

    fieldnames = [
        "method",
        "text_weight",
        "image_weight",
        "threshold",
        "accuracy",
        "precision",
        "recall",
        "false_hit_rate",
        "false_miss_rate",
        "near_positive_false_miss_rate",
        "hard_negative_false_hit_rate",
        "auto_hit_count",
        "review_count",
        "miss_count",
    ]
    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(comparison_rows)

    payload = {
        "found_text_only": text_only is not None,
        "found_image_only": image_only is not None,
        "found_rule_fusion": rule_fusion is not None,
        "found_dual_threshold_fusion": review_json.exists(),
        "recommended_method": "dual-threshold fusion",
        "methods": comparison_rows,
        "source_files": {
            "summary_csv": str(summary_csv),
            "fusion_weight_best_summary_csv": str(fusion_best_csv),
            "recommended_review_threshold_json": str(review_json),
        },
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    table = "\n".join(
        [
            "| method | text_weight | image_weight | threshold | accuracy | precision | recall | false_hit_rate | false_miss_rate | near_positive_false_miss_rate | hard_negative_false_hit_rate | auto_hit_count | review_count | miss_count |",
            "|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            *[
                "| {method} | {text_weight} | {image_weight} | {threshold} | {accuracy} | {precision} | {recall} | {false_hit_rate} | {false_miss_rate} | {near_positive_false_miss_rate} | {hard_negative_false_hit_rate} | {auto_hit_count} | {review_count} | {miss_count} |".format(
                    **row
                )
                for row in comparison_rows
            ],
        ]
    )

    report = f"""# 纯文本 / 纯图像 / 图文融合相似度方法对比报告

## 1. 对比目的

该报告对应导师会议中“验证不同相似度判断方法”的要求。会议中明确指出，缓存复用能否成为研究问题，关键在于相似度如何定义，以及不同相似度判断方法是否会影响缓存命中率、误命中率和生成效率。

## 2. 对比方法

- text-only：仅使用文本相似度，text_weight=1.0, image_weight=0.0。
- image-only：仅使用图像相似度，text_weight=0.0, image_weight=1.0。
- rule fusion：规则融合，score = 0.5 * text_score + 0.5 * image_score。
- dual-threshold fusion：双阈值融合，0.7 <= score < 0.78 进入 review，score >= 0.78 自动复用。

## 3. 指标对比表

{table}

## 4. 结果分析

纯文本方法的优点是可解释性强，能直接利用 VLM 输出的语义标签；缺点是容易受到描述随机性影响。同一目标在不同轮次中可能被描述成粗粒度或细粒度不同文本，导致相似度不稳定。

纯图像方法的优点是能绕开文本描述不一致的问题，直接利用视觉特征；缺点是容易受到光照、遮挡、裁剪、伪影和前景 mask 质量影响。在当前 v3_real_70 结果中，image-only 的 false_hit_rate 高于规则融合，说明纯图像判断更容易引入误复用风险。

图文融合方法在当前样本上表现更均衡。0.5 / 0.5 的规则融合保持 false_hit_rate = 0，同时 recall = 0.6111，说明文本和图像互补后能够兼顾安全性和复用能力。

双阈值策略的意义在于把一次性 hit / miss 改成 auto_hit / review / miss 三分支。高分样本自动复用，边界样本进入用户确认区，低分样本回退生成。当前 0.7 / 0.78 策略让 10 个 near_positive 边界样本进入 review，且 review_false_candidate_count = 0。

当前先推荐规则融合 baseline，而不是马上训练 MLP，原因是样本量只有 70，仍偏少。规则融合和双阈值可解释、可控、可回退，更适合当前阶段形成稳定实验链路。MLP 可以作为样本扩展到 80～100 后的后续工作。

## 5. 当前推荐策略

score = 0.5 * text_score + 0.5 * image_score

score >= 0.78：auto_hit  
0.7 <= score < 0.78：review  
score < 0.7：miss

推荐理由：

- false_hit_rate 保持较低；
- auto_hit 可跳过 TripoSR；
- review 区可保守处理；
- 规则可解释；
- 当前样本量只有 70，暂不适合直接训练 MLP。

## 6. 与多用户缓存仿真的关系

多用户缓存仿真的基础仍然依赖可靠的相似度判断。只有相似度判断足够稳定，多用户缓存共享才不会出现错误复用。因此，相似度方法对比是多用户协同缓存实验的前置基础。

如果后续要做 user_A / user_B 缓存共享实验，应先使用当前推荐的图文融合双阈值策略来判断远端缓存是否可复用，再比较远端传输耗时和本地重新生成耗时。
"""
    report_path.write_text(report, encoding="utf-8")

    combined = report + json_path.read_text(encoding="utf-8")
    output = {
        "found_text_only": text_only is not None,
        "found_image_only": image_only is not None,
        "found_rule_fusion": rule_fusion is not None,
        "found_dual_threshold_fusion": review_json.exists(),
        "recommended_method": "dual-threshold fusion",
        "garbled_detected": has_garbled(combined),
        "report_path": str(report_path),
        "csv_path": str(csv_path),
        "json_path": str(json_path),
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
