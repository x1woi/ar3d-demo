from __future__ import annotations

import argparse
import csv
import json
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List


ARCHIVE_FILES = [
    "summary.csv",
    "summary.json",
    "threshold_summary.csv",
    "experiment_report.md",
    "teacher_summary.md",
]


INDEX_FIELDS = [
    "archive_id",
    "archived_at",
    "archive_dir",
    "total_samples",
    "recommended_threshold",
    "accuracy",
    "precision",
    "recall",
    "f1",
    "false_hit_rate",
    "false_miss_rate",
    "avg_elapsed_ms",
]


def read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception:
        return default


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def make_archive_dir(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    base = time.strftime("exp_%Y%m%d_%H%M%S")
    archive_dir = root / base
    counter = 1
    while archive_dir.exists():
        archive_dir = root / f"{base}_{counter:02d}"
        counter += 1
    archive_dir.mkdir(parents=True, exist_ok=False)
    return archive_dir


def copy_result_files(eval_dir: Path, archive_dir: Path) -> None:
    for filename in ARCHIVE_FILES:
        src = eval_dir / filename
        if not src.exists():
            print(f"[WARNING] 归档文件不存在，已跳过：{src}")
            continue
        shutil.copy2(src, archive_dir / filename)


def meta_from_summary(eval_dir: Path, archive_dir: Path, archived_at: str) -> Dict[str, Any]:
    summary = read_json(eval_dir / "summary.json", {})
    return {
        "archived_at": archived_at,
        "source_eval_dir": str(eval_dir),
        "archive_dir": str(archive_dir),
        "recommended_threshold": summary.get("recommended_threshold", ""),
        "total_samples": summary.get("total_samples", 0),
        "accuracy": summary.get("accuracy", 0),
        "precision": summary.get("precision", 0),
        "recall": summary.get("recall", 0),
        "f1": summary.get("f1", 0),
        "false_hit_rate": summary.get("false_hit_rate", 0),
        "false_miss_rate": summary.get("false_miss_rate", 0),
        "avg_elapsed_ms": summary.get("avg_elapsed_ms", 0),
    }


def append_archive_index(index_path: Path, archive_id: str, meta: Dict[str, Any]) -> None:
    index_path.parent.mkdir(parents=True, exist_ok=True)
    exists = index_path.exists()
    row = {
        "archive_id": archive_id,
        "archived_at": meta.get("archived_at", ""),
        "archive_dir": meta.get("archive_dir", ""),
        "total_samples": meta.get("total_samples", 0),
        "recommended_threshold": meta.get("recommended_threshold", ""),
        "accuracy": meta.get("accuracy", 0),
        "precision": meta.get("precision", 0),
        "recall": meta.get("recall", 0),
        "f1": meta.get("f1", 0),
        "false_hit_rate": meta.get("false_hit_rate", 0),
        "false_miss_rate": meta.get("false_miss_rate", 0),
        "avg_elapsed_ms": meta.get("avg_elapsed_ms", 0),
    }
    with index_path.open("a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=INDEX_FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def archive_experiment(eval_dir: Path, archive_root: Path) -> Dict[str, Any]:
    eval_dir = Path(eval_dir)
    archive_root = Path(archive_root)
    archive_dir = make_archive_dir(archive_root)
    archived_at = time.strftime("%Y-%m-%d %H:%M:%S")

    copy_result_files(eval_dir, archive_dir)
    meta = meta_from_summary(eval_dir, archive_dir, archived_at)
    meta_path = archive_dir / "archive_meta.json"
    write_json(meta_path, meta)

    index_path = archive_root / "archive_index.csv"
    append_archive_index(index_path, archive_dir.name, meta)

    return {
        "archive_dir": archive_dir,
        "meta_path": meta_path,
        "index_path": index_path,
        "meta": meta,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Archive current v2_hard cache similarity experiment results."
    )
    parser.add_argument(
        "--eval-dir",
        type=str,
        default="paper_repro_outputs/cache_similarity_eval_v2_hard",
    )
    parser.add_argument(
        "--archive-root",
        type=str,
        default="paper_repro_outputs/cache_similarity_archives",
    )
    args = parser.parse_args()

    result = archive_experiment(Path(args.eval_dir), Path(args.archive_root))
    meta = result["meta"]

    print("=" * 72)
    print(f"archive_dir: {result['archive_dir']}")
    print(f"archive_meta.json: {result['meta_path']}")
    print(f"archive_index.csv: {result['index_path']}")
    print(f"recommended_threshold: {meta.get('recommended_threshold', '')}")
    print(f"accuracy: {meta.get('accuracy', 0)}")
    print(f"false_hit_rate: {meta.get('false_hit_rate', 0)}")
    print(f"false_miss_rate: {meta.get('false_miss_rate', 0)}")
    print("=" * 72)


if __name__ == "__main__":
    main()
