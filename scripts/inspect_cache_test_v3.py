from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Dict, List


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SAMPLE_TYPES = ["positive", "near_positive", "hard_negative", "negative"]
TARGET_COUNTS = {
    "positive": 12,
    "near_positive": 14,
    "hard_negative": 14,
    "negative": 10,
}


def count_images(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for item in path.iterdir() if item.is_file() and item.suffix.lower() in IMAGE_EXTS)


def collect_counts(root: Path) -> Dict[str, int]:
    counts = {f"{name}_count": count_images(root / name) for name in SAMPLE_TYPES}
    counts["total_count"] = sum(counts.values())
    return counts


def collect_deficits(counts: Dict[str, int]) -> Dict[str, int]:
    deficits = {
        f"{name}_deficit": max(0, TARGET_COUNTS[name] - counts[f"{name}_count"])
        for name in SAMPLE_TYPES
    }
    deficits["total_deficit"] = max(0, sum(TARGET_COUNTS.values()) - counts["total_count"])
    return deficits


def build_warnings(counts: Dict[str, int]) -> List[str]:
    warnings: List[str] = []
    if counts["total_count"] < 20:
        warnings.append("total_count < 20，当前样本量不足，只适合流程测试。")
    if counts["near_positive_count"] < 12:
        warnings.append("near_positive 数量不足，建议优先补充到 12 张左右，用于验证相似样本复用能力。")
    if counts["hard_negative_count"] < 12:
        warnings.append("hard_negative 数量不足，建议优先补充到 12 张左右，用于验证复杂负样本误命中风险。")
    for sample_type in SAMPLE_TYPES:
        if counts[f"{sample_type}_count"] == 0:
            warnings.append(f"{sample_type} 目录为空，对应分组指标无法评估。")
    return warnings


def priority_categories(deficits: Dict[str, int]) -> List[str]:
    priority = [
        name
        for name in ("hard_negative", "near_positive", "positive", "negative")
        if deficits[f"{name}_deficit"] > 0
    ]
    return priority


def next_step(counts: Dict[str, int], deficits: Dict[str, int]) -> str:
    all_non_empty = all(counts[f"{name}_count"] > 0 for name in SAMPLE_TYPES)
    if counts["total_count"] == 0:
        return "先从 contact sheet 中挑选候选图，使用 sort_cache_candidates_helper.py 分拣到 cache_test_v3_real。"
    priorities = priority_categories(deficits)
    priority_text = "、".join(priorities) if priorities else "无"
    if counts["total_count"] < 20:
        return f"继续补充样本，当前只适合流程测试；优先补：{priority_text}。"
    if counts["total_count"] >= 50 and all_non_empty:
        return "样本结构较完整，可以作为较完整的 v3_real 实验；下一步运行 run_cache_v3_experiment.py --rebuild-labels。"
    if counts["total_count"] >= 30:
        return f"可以做初步 v3_real 实验；正式汇报前建议继续补足到约 50 条，优先补：{priority_text}。"
    return f"可以继续补充样本；达到 30 条后可做初步 v3_real 实验。优先补：{priority_text}。"


def write_report(
    root: Path,
    output: Path,
    counts: Dict[str, int],
    deficits: Dict[str, int],
    warnings: List[str],
    suggestion: str,
) -> None:
    lines = [
        "# cache_test_v3_real 样本状态检查报告",
        "",
        f"- 检查时间：{time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"- root：{root}",
        "",
        "## 四类目录路径",
        "",
    ]
    for sample_type in SAMPLE_TYPES:
        lines.append(f"- {sample_type}: {root / sample_type}")

    lines.extend(
        [
            "",
            "## 图片数量",
            "",
            f"- positive_count: {counts['positive_count']}",
            f"- near_positive_count: {counts['near_positive_count']}",
            f"- hard_negative_count: {counts['hard_negative_count']}",
            f"- negative_count: {counts['negative_count']}",
            f"- total_count: {counts['total_count']}",
            "",
            "## 建议目标",
            "",
            f"- positive: {TARGET_COUNTS['positive']}",
            f"- near_positive: {TARGET_COUNTS['near_positive']}",
            f"- hard_negative: {TARGET_COUNTS['hard_negative']}",
            f"- negative: {TARGET_COUNTS['negative']}",
            f"- total: {sum(TARGET_COUNTS.values())}",
            "",
            "## 距离目标还差",
            "",
            f"- positive_deficit: {deficits['positive_deficit']}",
            f"- near_positive_deficit: {deficits['near_positive_deficit']}",
            f"- hard_negative_deficit: {deficits['hard_negative_deficit']}",
            f"- negative_deficit: {deficits['negative_deficit']}",
            f"- total_deficit: {deficits['total_deficit']}",
            "",
            "## 优先补充类别",
            "",
            f"- {'、'.join(priority_categories(deficits)) if priority_categories(deficits) else '无，已达到当前目标'}",
            "",
            "## Warning 列表",
            "",
        ]
    )
    if warnings:
        lines.extend(f"- {warning}" for warning in warnings)
    else:
        lines.append("- 无")

    lines.extend(
        [
            "",
            "## 下一步建议",
            "",
            suggestion,
            "",
            "常用命令：",
            "",
            "```powershell",
            ".\\.venv\\Scripts\\python.exe sort_cache_candidates_helper.py `",
            "  --source-dir paper_repro_outputs\\cache_candidate_images\\maybe_other `",
            "  --target-root cache_test_v3_real `",
            "  --target-type hard_negative `",
            "  --indices 1,2,3",
            "",
            ".\\.venv\\Scripts\\python.exe run_cache_v3_experiment.py --rebuild-labels",
            "```",
        ]
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def inspect(root: Path, output: Path) -> Dict[str, object]:
    counts = collect_counts(root)
    deficits = collect_deficits(counts)
    warnings = build_warnings(counts)
    suggestion = next_step(counts, deficits)
    write_report(root, output, counts, deficits, warnings, suggestion)
    return {
        "counts": counts,
        "deficits": deficits,
        "warnings": warnings,
        "suggestion": suggestion,
        "output": output,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect cache_test_v3_real sorting status.")
    parser.add_argument("--root", default="cache_test_v3_real")
    parser.add_argument("--output", default="cache_test_v3_real/inspect_report.md")
    args = parser.parse_args()

    result = inspect(Path(args.root), Path(args.output))
    counts: Dict[str, int] = result["counts"]  # type: ignore[assignment]
    deficits: Dict[str, int] = result["deficits"]  # type: ignore[assignment]

    print("=" * 72)
    print(f"root: {args.root}")
    print(f"positive_count: {counts['positive_count']}")
    print(f"near_positive_count: {counts['near_positive_count']}")
    print(f"hard_negative_count: {counts['hard_negative_count']}")
    print(f"negative_count: {counts['negative_count']}")
    print(f"total_count: {counts['total_count']}")
    print(f"positive_target: {TARGET_COUNTS['positive']}, deficit: {deficits['positive_deficit']}")
    print(f"near_positive_target: {TARGET_COUNTS['near_positive']}, deficit: {deficits['near_positive_deficit']}")
    print(f"hard_negative_target: {TARGET_COUNTS['hard_negative']}, deficit: {deficits['hard_negative_deficit']}")
    print(f"negative_target: {TARGET_COUNTS['negative']}, deficit: {deficits['negative_deficit']}")
    print(f"total_target: {sum(TARGET_COUNTS.values())}, deficit: {deficits['total_deficit']}")
    print(f"priority: {', '.join(priority_categories(deficits)) if priority_categories(deficits) else 'none'}")
    print(f"report: {result['output']}")
    print(f"next step: {result['suggestion']}")
    print("=" * 72)


if __name__ == "__main__":
    main()
