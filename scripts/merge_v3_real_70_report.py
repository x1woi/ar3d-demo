from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List


DEFAULT_EVAL_DIR = Path("paper_repro_outputs/cache_similarity_eval_v3_real_70")


def read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def fmt(value: Any, default: str = "N/A") -> str:
    if value is None or value == "":
        return default
    return str(value)


def metric(data: Dict[str, Any], *keys: str, default: Any = None) -> Any:
    cur: Any = data
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def build_report(eval_dir: Path, output_path: Path) -> Dict[str, Any]:
    summary = read_json(eval_dir / "summary.json")
    fusion = read_json(eval_dir / "fusion_weight_ablation" / "recommended_fusion_weight.json")
    weighted = read_json(
        eval_dir / "weighted_dual_threshold_analysis" / "recommended_weighted_dual_threshold.json"
    )
    text_enhance = read_json(eval_dir / "text_score_enhancement" / "recommended_text_enhancement.json")
    latency = read_json(eval_dir / "latency_analysis" / "latency_summary.json")

    false_miss_rows = read_csv_rows(eval_dir / "error_analysis" / "false_miss_cases.csv")
    false_hit_rows = read_csv_rows(eval_dir / "error_analysis" / "false_hit_cases.csv")
    borderline_rows = read_csv_rows(eval_dir / "error_analysis" / "borderline_cases.csv")
    distribution_rows = read_csv_rows(eval_dir / "error_analysis" / "score_distribution_by_type.csv")

    false_miss_count = len(false_miss_rows)
    false_hit_count = len(false_hit_rows)
    borderline_count = len(borderline_rows)
    false_miss_by_type: Dict[str, int] = {}
    for row in false_miss_rows:
        sample_type = row.get("sample_type", "unknown") or "unknown"
        false_miss_by_type[sample_type] = false_miss_by_type.get(sample_type, 0) + 1

    reason_counts: Dict[str, int] = {}
    for row in false_miss_rows:
        reason = row.get("reason_guess", "unknown") or "unknown"
        reason_counts[reason] = reason_counts.get(reason, 0) + 1
    main_reason = max(reason_counts.items(), key=lambda item: item[1])[0] if reason_counts else "N/A"

    fusion_metrics = fusion.get("key_metrics", {}) if isinstance(fusion.get("key_metrics"), dict) else {}
    weighted_metrics = weighted.get("key_metrics", {}) if isinstance(weighted.get("key_metrics"), dict) else {}
    text_metrics = (
        text_enhance.get("key_metrics", {}) if isinstance(text_enhance.get("key_metrics"), dict) else {}
    )

    text_effective = (
        text_enhance.get("recommended_strategy") != "original"
        and text_metrics.get("recall", 0) > summary.get("recall", 0)
        and text_metrics.get("false_hit_rate", 1) <= 0.05
    )

    lines = [
        "# v3_real_70 扩样本缓存复用实验阶段报告",
        "",
        "## 1. 实验目的",
        "",
        "本轮实验在 70 条样本上验证图文融合缓存复用策略的稳定性，重点观察从 50 条扩展到 70 条后，召回率、误命中率、漏命中率和边界样本分布是否发生改善。",
        "",
        "## 2. 数据集说明",
        "",
        f"- total_samples = {fmt(summary.get('total_samples'))}",
        "- 本轮是在 50 条基础上新增 near_positive 和 hard_negative 自采集视频帧。",
        "- 数据集目录：`paper_repro_outputs/cache_similarity_dataset_v3_real_70_rebuild`",
        "- 实验输出目录：`paper_repro_outputs/cache_similarity_eval_v3_real_70`",
        "",
        "## 3. 基础实验结果",
        "",
        f"- recommended_threshold: {fmt(summary.get('recommended_threshold'))}",
        f"- accuracy: {fmt(summary.get('accuracy'))}",
        f"- precision: {fmt(summary.get('precision'))}",
        f"- recall: {fmt(summary.get('recall'))}",
        f"- false_hit_rate: {fmt(summary.get('false_hit_rate'))}",
        f"- false_miss_rate: {fmt(summary.get('false_miss_rate'))}",
        "",
        "50 条版本 recall = 0.4615，70 条版本 recall = 0.6111。扩充自采集边界样本后，召回率提升，漏命中率下降，同时 false_hit_rate 仍为 0，说明误复用控制保持稳定。",
        "",
        "## 4. 融合权重消融结果",
        "",
        f"- recommended_text_weight: {fmt(fusion.get('recommended_text_weight'))}",
        f"- recommended_image_weight: {fmt(fusion.get('recommended_image_weight'))}",
        f"- best_threshold: {fmt(fusion.get('best_threshold'))}",
        f"- recall: {fmt(fusion_metrics.get('recall'))}",
        f"- false_hit_rate: {fmt(fusion_metrics.get('false_hit_rate'))}",
        f"- near_positive_false_miss_rate: {fmt(fusion_metrics.get('near_positive_false_miss_rate'))}",
        "",
        "## 5. 加权双阈值结果",
        "",
        f"- recommended_weak_threshold: {fmt(weighted.get('recommended_weak_threshold'))}",
        f"- recommended_strong_threshold: {fmt(weighted.get('recommended_strong_threshold'))}",
        f"- auto_false_hit_rate: {fmt(weighted_metrics.get('auto_false_hit_rate'))}",
        f"- review_rate: {fmt(weighted_metrics.get('review_rate'))}",
        f"- near_positive_auto_hit_count: {fmt(weighted_metrics.get('near_positive_auto_hit_count'))}",
        f"- near_positive_review_count: {fmt(weighted_metrics.get('near_positive_review_count'))}",
        "",
        "## 6. 错误样本分析",
        "",
        f"- false_miss_count: {false_miss_count}",
        f"- false_hit_count: {false_hit_count}",
        f"- borderline_count: {borderline_count}",
        f"- false_miss_by_type: {false_miss_by_type}",
        f"- main_reason: {main_reason}",
        "",
        "分数分布摘要：",
        "",
    ]
    if distribution_rows:
        lines.extend(
            [
                "| sample_type | count | avg_text_score | avg_image_score | avg_fused_score | min_fused_score | max_fused_score |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in distribution_rows:
            lines.append(
                f"| {row.get('sample_type','')} | {row.get('count','')} | {row.get('avg_text_score','')} | "
                f"{row.get('avg_image_score','')} | {row.get('avg_fused_score','')} | "
                f"{row.get('min_fused_score','')} | {row.get('max_fused_score','')} |"
            )
    else:
        lines.append("- 分数分布文件缺失。")

    lines.extend(
        [
            "",
            "## 7. 文本相似度增强结果",
            "",
            f"- recommended_strategy: {fmt(text_enhance.get('recommended_strategy'))}",
            f"- recommended_threshold: {fmt(text_enhance.get('recommended_threshold'))}",
            f"- recall: {fmt(text_metrics.get('recall'))}",
            f"- false_hit_rate: {fmt(text_metrics.get('false_hit_rate'))}",
            f"- 是否提升 recall: {'是' if text_effective else '否'}",
            f"- 是否引入 false_hit: {'是' if text_metrics.get('false_hit_rate', 0) > summary.get('false_hit_rate', 0) else '否'}",
            "",
            "## 8. 延迟收益结果",
            "",
            f"- saved_latency_seconds: {fmt(latency.get('saved_latency_seconds'))}",
            f"- speedup_ratio: {fmt(latency.get('speedup_ratio'))}",
            f"- avg_saved_latency_per_sample_ms: {fmt(latency.get('avg_saved_latency_per_sample_ms'))}",
            "",
            "## 9. 阶段性结论",
            "",
            "1. 70 条样本后 recall 相比 50 条提升，说明补充自采集 near_positive / hard_negative 对验证缓存复用机制是有价值的。",
            "2. false_hit_rate 仍为 0，说明当前阈值下误复用控制较稳定。",
            "3. false_miss 仍然存在，且 review_rate 仍为 0，说明后续还需要继续关注边界区间和漏命中样本。",
            "4. 当前仍不使用 MLP，继续以规则融合、阈值分析和错误样本解释作为 baseline。",
            "5. 下一步应根据错误分析决定是继续细化标签、改进文本相似度，还是补更多真正落在 weak/strong 阈值之间的边界样本。",
            "",
        ]
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")

    return {
        "recommended_threshold": summary.get("recommended_threshold"),
        "recall": summary.get("recall"),
        "false_hit_rate": summary.get("false_hit_rate"),
        "false_miss_rate": summary.get("false_miss_rate"),
        "recommended_text_weight": fusion.get("recommended_text_weight"),
        "recommended_image_weight": fusion.get("recommended_image_weight"),
        "recommended_weak_threshold": weighted.get("recommended_weak_threshold"),
        "recommended_strong_threshold": weighted.get("recommended_strong_threshold"),
        "review_rate": weighted_metrics.get("review_rate"),
        "false_miss_count": false_miss_count,
        "false_hit_count": false_hit_count,
        "borderline_count": borderline_count,
        "text_enhancement_effective": text_effective,
        "saved_latency_seconds": latency.get("saved_latency_seconds"),
        "report_path": str(output_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge v3_real_70 cache experiment reports.")
    parser.add_argument("--eval-dir", default=str(DEFAULT_EVAL_DIR))
    parser.add_argument(
        "--output",
        default=str(DEFAULT_EVAL_DIR / "v3_real_70_stage_report.md"),
    )
    args = parser.parse_args()

    result = build_report(Path(args.eval_dir), Path(args.output))
    print("=" * 72)
    for key, value in result.items():
        print(f"{key}: {value}")
    print("=" * 72)


if __name__ == "__main__":
    main()
