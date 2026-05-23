from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SAMPLE_TYPES = ["positive", "near_positive", "hard_negative", "negative"]
DEFAULT_THRESHOLDS = "0.5,0.6,0.7,0.75,0.8,0.82,0.85,0.9"
DEFAULT_SOURCE_ROOT = Path("cache_test_v3_real")
DEFAULT_DATASET_DIR = Path("paper_repro_outputs/cache_similarity_dataset_v3_real")
DEFAULT_OUTPUT_DIR = Path("paper_repro_outputs/cache_similarity_eval_v3_real")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def list_images(path: Path) -> List[Path]:
    if not path.exists():
        return []
    return sorted(
        item
        for item in path.iterdir()
        if item.is_file() and item.suffix.lower() in IMAGE_EXTS
    )


def run_command(cmd: List[str]) -> None:
    print("[RUN]", " ".join(cmd))
    completed = subprocess.run(cmd, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"Command failed with code {completed.returncode}: {' '.join(cmd)}")


def check_source_dirs(root_dir: Path) -> Tuple[bool, Dict[str, int]]:
    counts: Dict[str, int] = {}
    missing: List[Path] = []

    for sample_type in SAMPLE_TYPES:
        target = root_dir / sample_type
        if not target.exists():
            missing.append(target)
            counts[sample_type] = 0
            continue
        counts[sample_type] = len(list_images(target))
        if counts[sample_type] == 0:
            print(f"[WARNING] {sample_type} 目录为空，相关分组指标可能无法稳定评估。")

    if missing:
        print("[ERROR] v3_real 四类样本目录尚未准备完整：")
        for item in missing:
            print(f"  - {item}")
        print("请先运行：")
        print(".\\.venv\\Scripts\\python.exe prepare_cache_v3_dirs.py")
        return False, counts

    total = sum(counts.values())
    print("[INFO] cache_test_v3_real 样本统计：")
    for sample_type in SAMPLE_TYPES:
        print(f"  {sample_type}: {counts[sample_type]}")
    print(f"  total: {total}")

    if total == 0:
        print("[WARNING] 当前还没有 v3_real 测试图片，请先向四类目录补充图片。")
    elif total < 50:
        print("[WARNING] 当前样本数少于建议目标约 50 条，可先流程验证，正式汇报建议继续扩充。")

    return True, counts


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
        sample_type = str(sample.get("sample_type") or "")
        if sample_type in counts:
            counts[sample_type] += 1
        if not str(sample.get("category") or "").strip():
            empty_category += 1
        if not str(sample.get("query_text") or "").strip():
            empty_query_text += 1

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

    if empty_category or empty_query_text:
        print("[WARNING] 存在空 category 或 query_text，请人工检查 labels.json 后再作为正式结果使用。")

    return stats


def run_label_check(python_exe: str, dataset_dir: Path) -> None:
    run_command(
        [
            python_exe,
            "check_cache_labels.py",
            "--labels",
            str(dataset_dir / "labels.json"),
            "--dataset-dir",
            str(dataset_dir),
            "--output",
            str(dataset_dir / "labels_check_report.md"),
        ]
    )


def run_experiment(python_exe: str, dataset_dir: Path, output_dir: Path, thresholds: str) -> None:
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


def print_v3_todo() -> None:
    print("")
    print("v3_real 跑完后建议依次重新运行：")
    print("1. 融合权重消融")
    print("2. 加权双阈值分析")
    print("3. 延迟对比分析")
    print("4. review 敏感性分析")
    print("5. 最终报告整合")


def main() -> None:
    parser = argparse.ArgumentParser(description="One-click v3_real cache similarity experiment runner.")
    parser.add_argument("--source-root", default=str(DEFAULT_SOURCE_ROOT))
    parser.add_argument("--dataset-dir", default=str(DEFAULT_DATASET_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--thresholds", default=DEFAULT_THRESHOLDS)
    parser.add_argument("--rebuild-labels", action="store_true")
    parser.add_argument("--python-exe", default=sys.executable)
    args = parser.parse_args()

    source_root = Path(args.source_root)
    dataset_dir = Path(args.dataset_dir)
    output_dir = Path(args.output_dir)

    ok, counts = check_source_dirs(source_root)
    if not ok:
        sys.exit(1)

    if sum(counts.values()) == 0 and not (dataset_dir / "labels.json").exists():
        print("[INFO] 未发现图片且 labels.json 不存在，暂不生成数据集。")
        print(f"v3_real 样本目录路径: {source_root}")
        print(f"v3_real 数据集输出路径: {dataset_dir}")
        print(f"v3_real 实验输出路径: {output_dir}")
        print("建议下一步：用户向四类目录补充图片，然后运行：")
        print(".\\.venv\\Scripts\\python.exe run_cache_v3_experiment.py --rebuild-labels")
        sys.exit(0)

    try:
        build_dataset_if_needed(
            python_exe=args.python_exe,
            source_root=source_root,
            dataset_dir=dataset_dir,
            rebuild_labels=args.rebuild_labels,
        )
        inspect_labels(dataset_dir / "labels.json")
        run_label_check(args.python_exe, dataset_dir)
        run_experiment(args.python_exe, dataset_dir, output_dir, args.thresholds)
        summary = read_json(output_dir / "summary.json")
    except Exception as exc:
        print(f"[ERROR] v3_real 实验运行失败：{exc}")
        sys.exit(1)

    print("=" * 72)
    print("v3_real cache similarity experiment finished")
    print(f"v3_real 样本目录路径: {source_root}")
    print(f"v3_real 数据集输出路径: {dataset_dir}")
    print(f"v3_real 实验输出路径: {output_dir}")
    print(f"summary.csv: {output_dir / 'summary.csv'}")
    print(f"summary.json: {output_dir / 'summary.json'}")
    print(f"threshold_summary.csv: {output_dir / 'threshold_summary.csv'}")
    print(f"recommended_threshold: {summary.get('recommended_threshold')}")
    print_v3_todo()
    print("=" * 72)


if __name__ == "__main__":
    main()
