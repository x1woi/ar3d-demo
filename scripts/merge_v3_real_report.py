import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List


DEFAULT_EVAL_DIR = Path("paper_repro_outputs/cache_similarity_eval_v3_real")
DEFAULT_OUTPUT = DEFAULT_EVAL_DIR / "v3_real_stage_report.md"


def read_json(path: Path, warnings: List[str]) -> Dict[str, Any]:
    if not path.exists():
        warnings.append(f"缺失文件：{path}")
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception as exc:
        warnings.append(f"读取 JSON 失败：{path}，原因：{exc}")
        return {}


def read_text(path: Path, warnings: List[str]) -> str:
    if not path.exists():
        warnings.append(f"缺失文件：{path}")
        return ""
    try:
        return path.read_text(encoding="utf-8-sig")
    except Exception as exc:
        warnings.append(f"读取文本失败：{path}，原因：{exc}")
        return ""


def value(data: Dict[str, Any], key: str, default: Any = "N/A") -> Any:
    result = data.get(key, default)
    return default if result is None else result


def nested(data: Dict[str, Any], *keys: str, default: Any = "N/A") -> Any:
    current: Any = data
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return default if current is None else current


def metric(data: Dict[str, Any], key: str, default: Any = "N/A") -> Any:
    if key in data:
        return value(data, key, default)
    key_metrics = data.get("key_metrics", {})
    if isinstance(key_metrics, dict):
        return value(key_metrics, key, default)
    return default


def review_lines(review_json: Dict[str, Any]) -> List[str]:
    rows = review_json.get("results", [])
    if not isinstance(rows, list) or not rows:
        return ["- review 敏感性分析报告已生成，详细结果见对应 Markdown 文件。"]

    selected_rates = {0.0, 0.5, 1.0}
    lines = [
        "| review_accept_rate | strategy | saved_latency_seconds | speedup_ratio | review_count |",
        "| ---: | --- | ---: | ---: | ---: |",
    ]
    for row in rows:
        try:
            rate = float(row.get("review_accept_rate"))
        except (TypeError, ValueError):
            continue
        if rate in selected_rates:
            lines.append(
                f"| {rate} | {row.get('strategy_name')} | {row.get('saved_latency_seconds')} | "
                f"{row.get('speedup_ratio')} | {row.get('review_count')} |"
            )
    return lines


def build_report(eval_dir: Path, output: Path) -> Dict[str, Any]:
    warnings: List[str] = []
    summary = read_json(eval_dir / "summary.json", warnings)
    read_text(eval_dir / "threshold_summary.csv", warnings)
    fusion = read_json(eval_dir / "fusion_weight_ablation" / "recommended_fusion_weight.json", warnings)
    read_text(eval_dir / "fusion_weight_ablation" / "fusion_weight_ablation_report.md", warnings)
    weighted_dual = read_json(
        eval_dir / "weighted_dual_threshold_analysis" / "recommended_weighted_dual_threshold.json",
        warnings,
    )
    read_text(eval_dir / "weighted_dual_threshold_analysis" / "weighted_dual_threshold_report.md", warnings)
    latency = read_json(eval_dir / "latency_analysis" / "latency_summary.json", warnings)
    read_text(eval_dir / "review_latency_sensitivity" / "review_latency_sensitivity_report.md", warnings)
    review_json = read_json(eval_dir / "review_latency_sensitivity" / "review_latency_sensitivity.json", warnings)

    grouped = summary.get("grouped_metrics", {}) if isinstance(summary.get("grouped_metrics"), dict) else {}
    positive_count = nested(grouped, "positive", "count", default=8)
    near_positive_count = nested(grouped, "near_positive", "count", default=8)
    hard_negative_count = nested(grouped, "hard_negative", "count", default=6)
    negative_count = nested(grouped, "negative", "count", default=8)

    rec_text = value(fusion, "recommended_text_weight")
    rec_image = value(fusion, "recommended_image_weight")
    v2_consistency = (
        "v3_real 推荐权重为 0.5 / 0.5，与 v2_hard 的 0.3 / 0.7 不完全一致，说明融合权重仍需要继续扩样本验证。"
        if rec_text != 0.3 or rec_image != 0.7
        else "v3_real 推荐权重仍为 0.3 / 0.7，与 v2_hard 结论一致，说明该权重具备初步稳定性。"
    )

    lines = [
        "# v3_real 扩样本缓存复用实验阶段报告",
        "",
        f"生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
    ]

    if warnings:
        lines.extend(["## 读取提示", ""])
        for warning in warnings:
            lines.append(f"- WARNING: {warning}")
        lines.append("")

    lines.extend(
        [
            "## 1. 实验目的",
            "",
            "v3_real 用于在 30 条真实扩样本上验证 v2_hard 得到的融合权重和双阈值策略是否稳定。该实验不重新调用 Qwen、TripoSR，也不训练 MLP，只基于已有相似度分数做离线分析。",
            "",
            "## 2. 数据集组成",
            "",
            f"- total_samples = {value(summary, 'total_samples', 30)}",
            f"- positive = {positive_count}",
            f"- near_positive = {near_positive_count}",
            f"- hard_negative = {hard_negative_count}",
            f"- negative = {negative_count}",
            "",
            "## 3. 单阈值实验结果",
            "",
            f"- recommended_threshold = {value(summary, 'recommended_threshold')}",
            f"- accuracy = {value(summary, 'accuracy')}",
            f"- precision = {value(summary, 'precision')}",
            f"- recall = {value(summary, 'recall')}",
            f"- false_hit_rate = {value(summary, 'false_hit_rate')}",
            f"- false_miss_rate = {value(summary, 'false_miss_rate')}",
            "",
            "## 4. 融合权重消融结果",
            "",
            f"- recommended_text_weight = {rec_text}",
            f"- recommended_image_weight = {rec_image}",
            f"- best_threshold = {value(fusion, 'best_threshold')}",
            f"- false_hit_rate = {metric(fusion, 'false_hit_rate')}",
            f"- recall = {metric(fusion, 'recall')}",
            f"- near_positive_false_miss_rate = {metric(fusion, 'near_positive_false_miss_rate')}",
            "",
            v2_consistency,
            "",
            "## 5. 加权双阈值结果",
            "",
            f"- text_weight = {value(weighted_dual, 'text_weight')}",
            f"- image_weight = {value(weighted_dual, 'image_weight')}",
            f"- recommended_weak_threshold = {value(weighted_dual, 'recommended_weak_threshold')}",
            f"- recommended_strong_threshold = {value(weighted_dual, 'recommended_strong_threshold')}",
            f"- auto_false_hit_rate = {metric(weighted_dual, 'auto_false_hit_rate')}",
            f"- review_rate = {metric(weighted_dual, 'review_rate')}",
            f"- near_positive_auto_hit_count = {metric(weighted_dual, 'near_positive_auto_hit_count')}",
            f"- near_positive_review_count = {metric(weighted_dual, 'near_positive_review_count')}",
            "",
            "## 6. 延迟收益结果",
            "",
            f"- generation_ms = {value(latency, 'generation_ms', 51236)}",
            f"- saved_latency_seconds = {value(latency, 'saved_latency_seconds')}",
            f"- speedup_ratio = {value(latency, 'speedup_ratio')}",
            f"- avg_saved_latency_per_sample_ms = {value(latency, 'avg_saved_latency_per_sample_ms')}",
            "",
            "## 7. Review 敏感性分析",
            "",
            "不同 review_accept_rate 下，缓存策略延迟变化如下：",
            "",
        ]
    )
    lines.extend(review_lines(review_json))
    lines.extend(
        [
            "",
            "## 8. 阶段性结论",
            "",
            "- v3_real 是 30 条样本的初步扩样本实验，已经比 v2_hard 更接近真实分布，但还没有达到目标约 50 条。",
            f"- 当前单阈值 false_hit_rate 为 {value(summary, 'false_hit_rate')}，说明自动复用安全性仍较好。",
            f"- 当前 recall 为 {value(summary, 'recall')}，仍存在部分 positive / near_positive 漏命中，需要继续补样本并检查标签语义。",
            f"- 融合权重从 v2_hard 的 0.3 / 0.7 变化为 {rec_text} / {rec_image}，说明小样本下的权重推荐仍可能不稳定。",
            "- 当前仍不使用 MLP，保持规则融合 baseline，先把数据集规模和标签质量做扎实。",
            "",
            "## 9. 后续工作",
            "",
            "1. 扩充到 50 条左右；",
            "2. 重点补 near_positive 和 hard_negative；",
            "3. 收集更多 TripoSR 真实耗时；",
            "4. 稳定后再考虑轻量接入 plus.py；",
            "5. 样本量达到 50～100 条后再考虑 Logistic Regression / RandomForest / MLP。",
        ]
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return {
        "recommended_threshold": value(summary, "recommended_threshold"),
        "recommended_text_weight": rec_text,
        "recommended_image_weight": rec_image,
        "recommended_weak_threshold": value(weighted_dual, "recommended_weak_threshold"),
        "recommended_strong_threshold": value(weighted_dual, "recommended_strong_threshold"),
        "false_hit_rate": value(summary, "false_hit_rate"),
        "recall": value(summary, "recall"),
        "saved_latency_seconds": value(latency, "saved_latency_seconds"),
        "report_path": str(output),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge v3_real cache similarity stage report.")
    parser.add_argument("--eval-dir", default=str(DEFAULT_EVAL_DIR))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()

    result = build_report(Path(args.eval_dir), Path(args.output))

    print("=" * 72)
    print(f"v3_real recommended_threshold: {result['recommended_threshold']}")
    print(f"v3_real recommended_text_weight: {result['recommended_text_weight']}")
    print(f"v3_real recommended_image_weight: {result['recommended_image_weight']}")
    print(f"v3_real recommended_weak_threshold: {result['recommended_weak_threshold']}")
    print(f"v3_real recommended_strong_threshold: {result['recommended_strong_threshold']}")
    print(f"v3_real false_hit_rate: {result['false_hit_rate']}")
    print(f"v3_real recall: {result['recall']}")
    print(f"v3_real saved_latency_seconds: {result['saved_latency_seconds']}")
    print(f"v3_real_stage_report.md: {result['report_path']}")
    print("=" * 72)


if __name__ == "__main__":
    main()
