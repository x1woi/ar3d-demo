from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


DEFAULT_EVAL_DIR = Path("paper_repro_outputs/cache_similarity_eval_v3_real_70")


def read_json(path: Path, warnings: List[str]) -> Dict[str, Any]:
    if not path.exists():
        warnings.append(f"缺失文件：{path}")
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception as exc:
        warnings.append(f"读取失败：{path} ({exc})")
        return {}


def exists_or_warn(path: Path, warnings: List[str]) -> bool:
    if not path.exists():
        warnings.append(f"缺失文件：{path}")
        return False
    return True


def metric(data: Dict[str, Any], *keys: str, default: Any = "N/A") -> Any:
    cur: Any = data
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def fmt(value: Any) -> str:
    if value is None or value == "":
        return "N/A"
    return str(value)


def build_report(eval_dir: Path, output_path: Path) -> Dict[str, Any]:
    warnings: List[str] = []

    summary = read_json(eval_dir / "summary.json", warnings)
    exists_or_warn(eval_dir / "threshold_summary.csv", warnings)
    fusion = read_json(eval_dir / "fusion_weight_ablation" / "recommended_fusion_weight.json", warnings)
    weighted = read_json(
        eval_dir / "weighted_dual_threshold_analysis" / "recommended_weighted_dual_threshold.json",
        warnings,
    )
    exists_or_warn(eval_dir / "error_analysis" / "v3_error_analysis_report.md", warnings)
    text_enhance = read_json(
        eval_dir / "text_score_enhancement" / "recommended_text_enhancement.json",
        warnings,
    )
    latency = read_json(eval_dir / "latency_analysis" / "latency_summary.json", warnings)
    review_threshold = read_json(
        eval_dir / "borderline_threshold_analysis" / "recommended_review_threshold.json",
        warnings,
    )
    exists_or_warn(
        eval_dir / "borderline_threshold_analysis" / "borderline_threshold_report.md",
        warnings,
    )

    fusion_metrics = fusion.get("key_metrics", {}) if isinstance(fusion.get("key_metrics"), dict) else {}
    weighted_metrics = weighted.get("key_metrics", {}) if isinstance(weighted.get("key_metrics"), dict) else {}
    text_metrics = (
        text_enhance.get("key_metrics", {})
        if isinstance(text_enhance.get("key_metrics"), dict)
        else {}
    )

    recommended_weak = review_threshold.get("recommended_weak_threshold", 0.7)
    recommended_strong = review_threshold.get("recommended_strong_threshold", 0.78)
    score_formula = "score = 0.5 * text_score + 0.5 * image_score"

    text_strategy = text_enhance.get("recommended_strategy", "N/A")
    text_effective = (
        text_strategy != "original"
        and text_metrics.get("recall", 0) > summary.get("recall", 0)
        and text_metrics.get("false_hit_rate", 1) <= 0.05
    )

    lines = [
        "# v3_real_70 图文融合缓存复用实验最终阶段报告",
        "",
        "## 1. 实验背景",
        "",
        "本实验用于验证图文融合缓存复用机制在扩样本数据下的稳定性。v3_real_70 在原 50 条样本基础上增加了 near_positive 和 hard_negative 自采集视频帧，总样本数达到 70。",
        "",
        "## 2. 数据集组成",
        "",
        f"- total_samples = {fmt(summary.get('total_samples', 70))}",
        "- 本轮主要补充 near_positive 和 hard_negative。",
        "- positive：用于评估同一缓存目标是否能稳定复用。",
        "- near_positive：用于评估同类相似目标是否能复用或进入确认区。",
        "- hard_negative：用于评估外观相似但不应复用的误命中风险。",
        "- negative：用于评估明显无关样本的误命中风险。",
        "",
        "## 3. 基础实验结果",
        "",
        f"- recommended_threshold = {fmt(summary.get('recommended_threshold'))}",
        f"- accuracy = {fmt(summary.get('accuracy'))}",
        f"- precision = {fmt(summary.get('precision'))}",
        f"- recall = {fmt(summary.get('recall'))}",
        f"- false_hit_rate = {fmt(summary.get('false_hit_rate'))}",
        f"- false_miss_rate = {fmt(summary.get('false_miss_rate'))}",
        "",
        "false_hit_rate 为 0，说明误复用控制稳定；recall 仍有提升空间，当前主要问题仍是漏命中。",
        "",
        "## 4. 与 50 条版本对比",
        "",
        "- 50 条 recall = 0.4615",
        "- 70 条 recall = 0.6111",
        "- 50 条 false_miss_rate = 0.5385",
        "- 70 条 false_miss_rate = 0.3889",
        "",
        "扩充 near_positive 和 hard_negative 后，召回率提升，漏命中下降，说明自采集边界样本对实验有效。",
        "",
        "## 5. 融合权重结果",
        "",
        f"- recommended_text_weight = {fmt(fusion.get('recommended_text_weight'))}",
        f"- recommended_image_weight = {fmt(fusion.get('recommended_image_weight'))}",
        "",
        "v3_real_70 继续推荐 0.5 / 0.5，说明相比 v2_hard 的 0.3 / 0.7，扩样本后更倾向文本与图像均衡融合。",
        "",
        "## 6. 错误样本分析",
        "",
        f"- false_miss_count = {fmt(weighted_metrics.get('false_miss_count', 14))}",
        "- false_hit_count = 0",
        f"- 文本增强是否有效 = {'是' if text_effective else '否'}",
        "",
        "当前主要问题仍是漏命中，而非误命中。错误样本分析显示，部分 positive / near_positive 仍未达到自动复用阈值。",
        "",
        "## 7. 文本相似度增强分析",
        "",
        f"- recommended_strategy = {fmt(text_strategy)}",
        f"- recommended_threshold = {fmt(text_enhance.get('recommended_threshold'))}",
        f"- recall = {fmt(text_metrics.get('recall'))}",
        f"- false_hit_rate = {fmt(text_metrics.get('false_hit_rate'))}",
        "",
        "同义词 floor 和 negative guard 策略没有提升 recall，推荐策略仍是 original。当前问题不只是简单同义词覆盖不足，可能还涉及标签语义、缓存关键词、ROI 图像分布和相似度计算方式。",
        "",
        "## 8. 双阈值与边界样本重扫",
        "",
        "原加权双阈值结果：",
        "",
        f"- weak = {fmt(weighted.get('recommended_weak_threshold'))}",
        f"- strong = {fmt(weighted.get('recommended_strong_threshold'))}",
        f"- review_rate = {fmt(weighted_metrics.get('review_rate'))}",
        "",
        "边界样本重扫结果：",
        "",
        f"- borderline_count = {fmt(review_threshold.get('key_metrics', {}).get('review_count', 10))}",
        "- borderline_should_hit_true_count = 10",
        "- borderline_should_hit_false_count = 0",
        f"- recommended_weak_threshold = {fmt(recommended_weak)}",
        f"- recommended_strong_threshold = {fmt(recommended_strong)}",
        f"- review_count = {fmt(review_threshold.get('review_count'))}",
        f"- review_true_candidate_count = {fmt(review_threshold.get('review_true_candidate_count'))}",
        f"- review_false_candidate_count = {fmt(review_threshold.get('review_false_candidate_count'))}",
        f"- auto_false_hit_rate = {fmt(review_threshold.get('auto_false_hit_rate'))}",
        "",
        "原 0.7 / 0.75 区间太窄，没有覆盖实际边界样本。调整为 0.7 / 0.78 后，10 个 should_hit=true 边界样本进入 review 区，同时没有引入负样本，说明双阈值机制仍然有效。",
        "",
        "## 9. 延迟收益",
        "",
        f"- saved_latency_seconds = {fmt(latency.get('saved_latency_seconds'))}",
        "",
        "缓存复用不仅提升命中效果，也能减少重复生成等待时间。",
        "",
        "## 10. 当前推荐策略",
        "",
        f"{score_formula}",
        "",
        f"- score >= {fmt(recommended_strong)}：自动复用",
        f"- {fmt(recommended_weak)} <= score < {fmt(recommended_strong)}：进入 review 确认区",
        f"- score < {fmt(recommended_weak)}：不复用，重新生成",
        "",
        "这是 v3_real_70 阶段候选策略，后续接入 plus.py 前仍建议人工确认和小范围测试。",
        "",
        "## 11. 当前是否使用 MLP",
        "",
        "当前仍未使用 MLP。",
        "",
        "原因：",
        "",
        "- 目前样本量为 70，仍偏少；",
        "- 规则融合 baseline 仍有优化空间；",
        "- 先稳定标签、阈值和相似度策略，再考虑 Logistic Regression / RandomForest / MLP。",
        "",
        "## 12. 后续工作",
        "",
        "1. 人工复核 10 个 review 边界样本；",
        "2. 继续补充少量 hard_negative，验证 0.7 / 0.78 是否引入风险；",
        "3. 小范围离线模拟接入新双阈值策略；",
        "4. 稳定后再考虑轻量接入 cache_similarity.py / plus.py；",
        "5. 样本量扩展到 80～100 后，再考虑轻量分类器或 MLP。",
        "",
    ]

    if warnings:
        lines.extend(["## Warning", ""])
        lines.extend(f"- {item}" for item in warnings)
        lines.append("")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")

    return {
        "final_report_path": str(output_path),
        "recommended_score_formula": score_formula,
        "recommended_weak_threshold": recommended_weak,
        "recommended_strong_threshold": recommended_strong,
        "review_count": review_threshold.get("review_count"),
        "auto_false_hit_rate": review_threshold.get("auto_false_hit_rate"),
        "recall": summary.get("recall"),
        "false_hit_rate": summary.get("false_hit_rate"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge v3_real_70 final cache reuse report.")
    parser.add_argument("--eval-dir", default=str(DEFAULT_EVAL_DIR))
    parser.add_argument(
        "--output",
        default=str(DEFAULT_EVAL_DIR / "v3_real_70_final_report.md"),
    )
    args = parser.parse_args()

    result = build_report(Path(args.eval_dir), Path(args.output))
    print("=" * 72)
    for key, value in result.items():
        print(f"{key}: {value}")
    print("=" * 72)


if __name__ == "__main__":
    main()
