from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Dict, List


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SAMPLE_TYPES = ["positive", "near_positive", "hard_negative", "negative"]


def count_images(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(
        1
        for item in path.iterdir()
        if item.is_file() and item.suffix.lower() in IMAGE_EXTS
    )


def collect_counts(root: Path) -> Dict[str, int]:
    counts = {f"{name}_count": count_images(root / name) for name in SAMPLE_TYPES}
    counts["total_count"] = sum(counts.values())
    return counts


def build_warnings(counts: Dict[str, int]) -> List[str]:
    warnings: List[str] = []
    if counts["total_count"] == 0:
        warnings.append("当前还没有测试图片，请先从 cache_candidate_images 中分拣图片。")
    if counts["positive_count"] == 0:
        warnings.append("positive 样本为空，无法评估应命中样本的召回率。")
    if counts["negative_count"] == 0:
        warnings.append("negative 样本为空，无法评估明显负样本的误命中率。")
    if counts["hard_negative_count"] == 0:
        warnings.append("hard_negative 样本为空，无法评估复杂相似负样本的误命中率。")
    return warnings


def next_step(counts: Dict[str, int]) -> str:
    all_non_empty = all(counts[f"{name}_count"] > 0 for name in SAMPLE_TYPES)
    if counts["total_count"] == 0:
        return ".\\.venv\\Scripts\\python.exe collect_cache_candidate_images.py"
    if counts["total_count"] >= 20 and all_non_empty:
        return (
            "样本结构较完整，可以作为 v2_hard 初步正式实验。下一步运行："
            " .\\.venv\\Scripts\\python.exe run_cache_v2_experiment.py --rebuild-labels"
        )
    if counts["total_count"] >= 10 and counts["positive_count"] > 0 and counts["negative_count"] > 0:
        return "样本数量已满足初步实验，可以运行 run_cache_v2_experiment.py --rebuild-labels。"
    return "继续从 cache_candidate_images 中分拣图片，优先补足 positive、hard_negative、negative。"


def write_report(root: Path, output: Path, counts: Dict[str, int], warnings: List[str], suggestion: str) -> None:
    lines = [
        "# cache_test_v2 分拣状态检查报告",
        "",
        f"- 检查时间: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"- root: {root}",
        "",
        "## 四类目录路径",
        "",
    ]
    for name in SAMPLE_TYPES:
        lines.append(f"- {name}: {root / name}")

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
            "## Warning 列表",
            "",
        ]
    )
    if warnings:
        for warning in warnings:
            lines.append(f"- {warning}")
    else:
        lines.append("无")

    lines.extend(
        [
            "",
            "## 下一步建议命令",
            "",
            "```powershell",
        ]
    )
    if counts["total_count"] == 0:
        lines.append(".\\.venv\\Scripts\\python.exe collect_cache_candidate_images.py")
    else:
        lines.append(".\\.venv\\Scripts\\python.exe run_cache_v2_experiment.py --rebuild-labels")
        lines.append(".\\.venv\\Scripts\\python.exe check_cache_labels.py")
    lines.extend(["```", "", f"说明：{suggestion}", ""])

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")


def inspect(root: Path, output: Path) -> Dict[str, object]:
    counts = collect_counts(root)
    warnings = build_warnings(counts)
    suggestion = next_step(counts)
    write_report(root, output, counts, warnings, suggestion)
    return {
        "counts": counts,
        "warnings": warnings,
        "suggestion": suggestion,
        "output": output,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect cache_test_v2 sorting status.")
    parser.add_argument("--root", type=str, default="cache_test_v2")
    parser.add_argument("--output", type=str, default="cache_test_v2/inspect_report.md")
    args = parser.parse_args()

    result = inspect(Path(args.root), Path(args.output))
    counts: Dict[str, int] = result["counts"]  # type: ignore[assignment]

    print("=" * 72)
    print(f"root: {args.root}")
    print(f"positive_count: {counts['positive_count']}")
    print(f"near_positive_count: {counts['near_positive_count']}")
    print(f"hard_negative_count: {counts['hard_negative_count']}")
    print(f"negative_count: {counts['negative_count']}")
    print(f"total_count: {counts['total_count']}")
    print(f"report: {result['output']}")
    print(f"next step: {result['suggestion']}")
    print("=" * 72)


if __name__ == "__main__":
    main()
