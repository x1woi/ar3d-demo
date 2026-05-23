from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEFAULT_THRESHOLDS = "0.5,0.6,0.7,0.75,0.8,0.82,0.85,0.9"
SAMPLE_TYPES = ["positive", "near_positive", "hard_negative", "negative"]


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def list_images(path: Path) -> List[Path]:
    if not path.exists():
        return []
    return sorted(
        item for item in path.iterdir()
        if item.is_file() and item.suffix.lower() in IMAGE_EXTS
    )


def check_source_dirs(root_dir: Path) -> Tuple[bool, Dict[str, int]]:
    counts: Dict[str, int] = {}
    missing = []

    for sample_type in SAMPLE_TYPES:
        target = root_dir / sample_type
        if not target.exists():
            missing.append(str(target))
            counts[sample_type] = 0
            continue
        count = len(list_images(target))
        counts[sample_type] = count
        if count == 0:
            print(
                f"[WARNING] {sample_type} 目录为空，实验可能无法有效评估对应类别。"
            )

    if missing:
        print("[ERROR] cache_test_v2 四类目录尚未准备完整：")
        for item in missing:
            print(f"  - {item}")
        print("请先运行：")
        print(".\\.venv\\Scripts\\python.exe prepare_cache_v2_dirs.py")
        return False, counts

    return True, counts


def run_command(cmd: List[str]) -> None:
    print("[RUN]", " ".join(cmd))
    completed = subprocess.run(cmd, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"Command failed with code {completed.returncode}: {' '.join(cmd)}")


def build_dataset_if_needed(
    python_exe: str,
    source_root: Path,
    dataset_dir: Path,
    rebuild_labels: bool,
) -> None:
    labels_path = dataset_dir / "labels.json"
    if labels_path.exists() and not rebuild_labels:
        print(f"[INFO] 已存在 labels.json，默认不覆盖：{labels_path}")
        return

    run_command(
        [
            python_exe,
            "cache_dataset_auto_make.py",
            "--positive-dir",
            str(source_root / "positive"),
            "--near-positive-dir",
            str(source_root / "near_positive"),
            "--hard-negative-dir",
            str(source_root / "hard_negative"),
            "--negative-dir",
            str(source_root / "negative"),
            "--output-dir",
            str(dataset_dir),
            "--auto-fill-labels",
        ]
    )


def inspect_labels(labels_path: Path) -> Dict[str, Any]:
    if not labels_path.exists():
        raise FileNotFoundError(f"labels.json 不存在：{labels_path}")

    data = read_json(labels_path)
    samples = data.get("samples", [])
    counts = {sample_type: 0 for sample_type in SAMPLE_TYPES}
    empty_category = 0
    empty_query_text = 0

    for sample in samples:
        sample_type = str(sample.get("sample_type") or "default")
        if sample_type in counts:
            counts[sample_type] += 1
        if not str(sample.get("category") or "").strip():
            empty_category += 1
        if not str(sample.get("query_text") or "").strip():
            empty_query_text += 1

    total_empty_label = sum(
        1
        for sample in samples
        if not str(sample.get("category") or "").strip()
        or not str(sample.get("query_text") or "").strip()
    )
    if total_empty_label:
        print(
            f"[WARNING] 有 {total_empty_label} 条样本 category 或 query_text 为空，"
            "请人工检查 labels.json 后再正式使用实验结果。"
        )

    stats = {
        "total_samples": len(samples),
        "positive_count": counts["positive"],
        "near_positive_count": counts["near_positive"],
        "hard_negative_count": counts["hard_negative"],
        "negative_count": counts["negative"],
        "empty_category_count": empty_category,
        "empty_query_text_count": empty_query_text,
    }

    print("[INFO] labels.json 样本统计：")
    for key, value in stats.items():
        print(f"  {key}: {value}")

    return stats


def run_experiment(
    python_exe: str,
    dataset_dir: Path,
    output_dir: Path,
    thresholds: str,
) -> None:
    run_command(
        [
            python_exe,
            "experiment_cache_similarity.py",
            "--dataset-dir",
            str(dataset_dir),
            "--output-dir",
            str(output_dir),
            "--thresholds",
            thresholds,
        ]
    )


def read_threshold_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def group_line(name: str, metrics: Dict[str, Any]) -> str:
    return (
        f"| {name} | {metrics.get('count', 0)} | {metrics.get('hit_count', 0)} | "
        f"{metrics.get('miss_count', 0)} | {metrics.get('accuracy', 0)} | "
        f"{metrics.get('false_hit_rate', 0)} | {metrics.get('false_miss_rate', 0)} |"
    )


def conclusion_lines(summary: Dict[str, Any]) -> List[str]:
    grouped = summary.get("grouped_metrics", {})
    hard_negative = grouped.get("hard_negative", {})
    positive = grouped.get("positive", {})
    lines = []

    hard_false_hit = float(hard_negative.get("false_hit_rate") or 0.0)
    positive_false_miss = float(positive.get("false_miss_rate") or 0.0)
    false_hit = float(summary.get("false_hit_rate") or 0.0)
    recall = float(summary.get("recall") or 0.0)

    if hard_false_hit > 0:
        lines.append(
            "hard_negative 样本误命中率较高，说明当前阈值或融合权重可能偏宽松，"
            "后续需要提高阈值或降低图像相似度权重。"
        )
    if positive_false_miss > 0:
        lines.append(
            "positive 样本漏命中率较高，说明当前阈值可能偏保守，导致可复用模型未被充分利用。"
        )
    if false_hit == 0 and recall >= 0.8:
        lines.append(
            "当前阈值下误命中率较低，同时保持较好的缓存复用能力，可作为候选默认阈值。"
        )
    if not lines:
        lines.append("当前实验结果需要结合分组指标进一步分析阈值和图文权重。")
    return lines


def generate_report(output_dir: Path, dataset_stats: Dict[str, Any]) -> Path:
    summary_path = output_dir / "summary.json"
    threshold_path = output_dir / "threshold_summary.csv"
    summary = read_json(summary_path)
    _ = read_threshold_csv(threshold_path)
    grouped = summary.get("grouped_metrics", {})

    lines = [
        "# v2_hard 图文融合缓存命中实验",
        "",
        "## 数据集统计",
        "",
        f"- total samples: {dataset_stats.get('total_samples', summary.get('total_samples', 0))}",
        f"- positive count: {dataset_stats.get('positive_count', 0)}",
        f"- near_positive count: {dataset_stats.get('near_positive_count', 0)}",
        f"- hard_negative count: {dataset_stats.get('hard_negative_count', 0)}",
        f"- negative count: {dataset_stats.get('negative_count', 0)}",
        "",
        "## 总体指标",
        "",
        f"- accuracy: {summary.get('accuracy', 0)}",
        f"- precision: {summary.get('precision', 0)}",
        f"- recall: {summary.get('recall', 0)}",
        f"- f1: {summary.get('f1', 0)}",
        f"- false_hit_rate: {summary.get('false_hit_rate', 0)}",
        f"- false_miss_rate: {summary.get('false_miss_rate', 0)}",
        f"- avg_elapsed_ms: {summary.get('avg_elapsed_ms', 0)}",
        "",
        "## 推荐阈值",
        "",
        f"- recommended_threshold: {summary.get('recommended_threshold')}",
        f"- recommended_reason: {summary.get('recommended_reason', '')}",
        "",
        "## 分组指标",
        "",
        "| sample_type | count | hit_count | miss_count | accuracy | false_hit_rate | false_miss_rate |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]

    for sample_type in SAMPLE_TYPES:
        lines.append(group_line(sample_type, grouped.get(sample_type, {})))

    lines.extend(["", "## 解释性结论", ""])
    for line in conclusion_lines(summary):
        lines.append(f"- {line}")

    report_path = output_dir / "experiment_report.md"
    write_text(report_path, "\n".join(lines) + "\n")
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="One-click v2_hard cache similarity experiment runner."
    )
    parser.add_argument("--source-root", type=str, default="cache_test_v2")
    parser.add_argument("--dataset-dir", type=str, default="paper_repro_outputs/cache_similarity_dataset_v2_hard")
    parser.add_argument("--output-dir", type=str, default="paper_repro_outputs/cache_similarity_eval_v2_hard")
    parser.add_argument("--thresholds", type=str, default=DEFAULT_THRESHOLDS)
    parser.add_argument("--rebuild-labels", action="store_true")
    parser.add_argument("--python-exe", type=str, default=sys.executable)
    args = parser.parse_args()

    source_root = Path(args.source_root)
    dataset_dir = Path(args.dataset_dir)
    output_dir = Path(args.output_dir)

    ok, _ = check_source_dirs(source_root)
    if not ok:
        sys.exit(1)

    try:
        build_dataset_if_needed(
            python_exe=args.python_exe,
            source_root=source_root,
            dataset_dir=dataset_dir,
            rebuild_labels=args.rebuild_labels,
        )
        dataset_stats = inspect_labels(dataset_dir / "labels.json")
        run_experiment(
            python_exe=args.python_exe,
            dataset_dir=dataset_dir,
            output_dir=output_dir,
            thresholds=args.thresholds,
        )
        report_path = generate_report(output_dir, dataset_stats)
        summary = read_json(output_dir / "summary.json")
    except Exception as exc:
        print(f"[ERROR] v2_hard 实验运行失败：{exc}")
        sys.exit(1)

    print("=" * 72)
    print("v2_hard cache similarity experiment finished")
    print(f"Dataset: {dataset_dir}")
    print(f"Output: {output_dir}")
    print(f"summary.csv: {output_dir / 'summary.csv'}")
    print(f"threshold_summary.csv: {output_dir / 'threshold_summary.csv'}")
    print(f"experiment_report.md: {report_path}")
    print(f"recommended_threshold: {summary.get('recommended_threshold')}")
    print("=" * 72)


if __name__ == "__main__":
    main()
