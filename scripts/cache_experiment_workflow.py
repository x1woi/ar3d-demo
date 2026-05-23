from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SAMPLE_TYPES = ["positive", "near_positive", "hard_negative", "negative"]
ROOT_DIR = Path("cache_test_v2")
DATASET_DIR = Path("paper_repro_outputs/cache_similarity_dataset_v2_hard")
EVAL_DIR = Path("paper_repro_outputs/cache_similarity_eval_v2_hard")
ARCHIVE_INDEX = Path("paper_repro_outputs/cache_similarity_archives/archive_index.csv")


def print_workflow() -> None:
    print("=" * 72)
    print("v2_hard 缓存相似度实验流程")
    print("=" * 72)
    print("1. 准备目录：")
    print("   .\\.venv\\Scripts\\python.exe prepare_cache_v2_dirs.py")
    print()
    print("2. 放入图片：")
    print("   cache_test_v2/positive")
    print("   cache_test_v2/near_positive")
    print("   cache_test_v2/hard_negative")
    print("   cache_test_v2/negative")
    print()
    print("3. 生成数据集：")
    print("   .\\.venv\\Scripts\\python.exe run_cache_v2_experiment.py --rebuild-labels")
    print()
    print("4. 检查标签：")
    print("   .\\.venv\\Scripts\\python.exe check_cache_labels.py")
    print()
    print("5. 如果 query_text 为空，可运行：")
    print("   .\\.venv\\Scripts\\python.exe check_cache_labels.py --fix-empty-query-text")
    print()
    print("6. 人工检查：")
    print("   paper_repro_outputs/cache_similarity_dataset_v2_hard/labels.json")
    print()
    print("7. 正式跑实验：")
    print("   .\\.venv\\Scripts\\python.exe run_cache_v2_experiment.py")
    print()
    print("8. 生成导师摘要：")
    print("   .\\.venv\\Scripts\\python.exe analyze_cache_v2_report.py")
    print()
    print("9. 归档实验：")
    print("   .\\.venv\\Scripts\\python.exe archive_cache_experiment_result.py")
    print("=" * 72)
    print("提示：运行 --status 可以查看当前进度和下一步建议。")


def count_images(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(
        1
        for item in path.iterdir()
        if item.is_file() and item.suffix.lower() in IMAGE_EXTS
    )


def collect_status() -> Dict[str, object]:
    dir_exists = {name: (ROOT_DIR / name).exists() for name in SAMPLE_TYPES}
    image_counts = {name: count_images(ROOT_DIR / name) for name in SAMPLE_TYPES}
    return {
        "root_exists": ROOT_DIR.exists(),
        "dir_exists": dir_exists,
        "image_counts": image_counts,
        "labels_json": (DATASET_DIR / "labels.json").exists(),
        "labels_check_report": (DATASET_DIR / "labels_check_report.md").exists(),
        "summary_json": (EVAL_DIR / "summary.json").exists(),
        "threshold_summary_csv": (EVAL_DIR / "threshold_summary.csv").exists(),
        "teacher_summary_md": (EVAL_DIR / "teacher_summary.md").exists(),
        "archive_index_csv": ARCHIVE_INDEX.exists(),
    }


def next_suggestion(status: Dict[str, object]) -> str:
    image_counts: Dict[str, int] = status["image_counts"]  # type: ignore[assignment]
    total_images = sum(image_counts.values())

    if not status["root_exists"]:
        return "建议运行 prepare_cache_v2_dirs.py 创建四类目录。"

    if total_images == 0:
        return "四类目录均为空，建议先放入测试图片。"

    if not status["labels_json"]:
        return "已有图片但 labels.json 不存在，建议运行 run_cache_v2_experiment.py --rebuild-labels。"

    if not status["labels_check_report"]:
        return "labels.json 已存在但尚未检查，建议运行 check_cache_labels.py。"

    if not status["summary_json"]:
        return "labels 检查报告已存在但 summary.json 不存在，建议运行 run_cache_v2_experiment.py。"

    if not status["teacher_summary_md"]:
        return "summary.json 已存在但 teacher_summary.md 不存在，建议运行 analyze_cache_v2_report.py。"

    if not status["archive_index_csv"]:
        return "teacher_summary.md 已存在但 archive_index.csv 不存在，建议运行 archive_cache_experiment_result.py。"

    return "v2_hard 实验流程已完成，可以查看 teacher_summary.md 或 archive_index.csv。"


def print_status() -> None:
    status = collect_status()
    dir_exists: Dict[str, bool] = status["dir_exists"]  # type: ignore[assignment]
    image_counts: Dict[str, int] = status["image_counts"]  # type: ignore[assignment]

    print("=" * 72)
    print("v2_hard 实验状态检查")
    print("=" * 72)
    print(f"cache_test_v2 exists: {status['root_exists']}")
    for name in SAMPLE_TYPES:
        exists_text = "exists" if dir_exists[name] else "missing"
        print(f"{name}: {exists_text}, {image_counts[name]} images")
        if dir_exists[name] and image_counts[name] == 0:
            print(f"[WARNING] {name} 目录为空。")

    print()
    print(f"labels.json exists: {status['labels_json']}")
    print(f"labels_check_report.md exists: {status['labels_check_report']}")
    print(f"summary.json exists: {status['summary_json']}")
    print(f"threshold_summary.csv exists: {status['threshold_summary_csv']}")
    print(f"teacher_summary.md exists: {status['teacher_summary_md']}")
    print(f"archive_index.csv exists: {status['archive_index_csv']}")
    print()
    print("下一步建议：")
    print(next_suggestion(status))
    print("=" * 72)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Show v2_hard cache similarity experiment workflow or status."
    )
    parser.add_argument("--status", action="store_true")
    args = parser.parse_args()

    if args.status:
        print_status()
    else:
        print_workflow()


if __name__ == "__main__":
    main()
