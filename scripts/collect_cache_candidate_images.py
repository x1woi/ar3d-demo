from __future__ import annotations

import argparse
import csv
import hashlib
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from PIL import Image


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SKIP_DIRS = {".venv", "__pycache__", "node_modules"}
DEFAULT_SOURCE_DIRS = ["runtime_assets", "multiview_test", "paper_repro_outputs"]
GROUPS = ["maybe_glasses", "maybe_head", "maybe_text", "maybe_other"]


def should_skip(path: Path) -> bool:
    return any(part in SKIP_DIRS for part in path.parts)


def iter_images(source_dirs: Iterable[Path]) -> Iterable[Path]:
    for source_dir in source_dirs:
        if not source_dir.exists():
            continue
        for path in source_dir.rglob("*"):
            if should_skip(path):
                continue
            if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
                yield path


def image_size(path: Path) -> Tuple[int, int]:
    with Image.open(path) as img:
        return img.size


def suggest_group(path: Path) -> str:
    text = str(path).lower()
    if any(token in text for token in ("glasses", "eye", "yanjing", "眼镜")):
        return "maybe_glasses"
    if any(token in text for token in ("head", "face", "rentou", "人头", "人脸")):
        return "maybe_head"
    if any(token in text for token in ("text", "book", "page", "shu", "书", "ocr")):
        return "maybe_text"
    return "maybe_other"


def unique_name(path: Path, index: int) -> str:
    digest = hashlib.md5(str(path).encode("utf-8", errors="ignore")).hexdigest()[:10]
    stem = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in path.stem)
    stem = stem[:48] or "image"
    suffix = path.suffix.lower() or ".jpg"
    return f"{index:04d}_{stem}_{digest}{suffix}"


def ensure_output_dirs(output_dir: Path) -> Dict[str, Path]:
    dirs = {"all": output_dir / "all"}
    for group in GROUPS:
        dirs[group] = output_dir / group
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def collect_candidates(
    source_dirs: List[Path],
    output_dir: Path,
    min_size: int = 64,
    max_files: int = 300,
) -> Dict[str, int | Path]:
    output_dir = Path(output_dir)
    dirs = ensure_output_dirs(output_dir)
    rows = []
    scanned = 0
    copied = 0
    group_counts = {group: 0 for group in GROUPS}

    for path in iter_images(source_dirs):
        scanned += 1
        if copied >= max_files:
            break
        try:
            width, height = image_size(path)
        except Exception:
            continue
        if width < min_size or height < min_size:
            continue

        group = suggest_group(path)
        filename = unique_name(path, copied)
        all_path = dirs["all"] / filename
        group_path = dirs[group] / filename
        shutil.copy2(path, all_path)
        shutil.copy2(path, group_path)

        copied += 1
        group_counts[group] += 1
        rows.append(
            {
                "original_path": str(path),
                "copied_path": str(all_path),
                "suggested_group": group,
                "filename": filename,
                "width": width,
                "height": height,
                "file_size_kb": round(path.stat().st_size / 1024, 3),
            }
        )

    index_path = output_dir / "candidate_index.csv"
    with index_path.open("w", newline="", encoding="utf-8-sig") as f:
        fieldnames = [
            "original_path",
            "copied_path",
            "suggested_group",
            "filename",
            "width",
            "height",
            "file_size_kb",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return {
        "scanned": scanned,
        "copied": copied,
        "maybe_glasses": group_counts["maybe_glasses"],
        "maybe_head": group_counts["maybe_head"],
        "maybe_text": group_counts["maybe_text"],
        "maybe_other": group_counts["maybe_other"],
        "index_path": index_path,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect candidate images for manual v2_hard cache similarity dataset review."
    )
    parser.add_argument(
        "--source-dir",
        action="append",
        default=[],
        help="Source image directory. Can be specified multiple times.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="paper_repro_outputs/cache_candidate_images",
    )
    parser.add_argument("--min-size", type=int, default=64)
    parser.add_argument("--max-files", type=int, default=300)
    args = parser.parse_args()

    source_dirs = [Path(item) for item in (args.source_dir or DEFAULT_SOURCE_DIRS)]
    stats = collect_candidates(
        source_dirs=source_dirs,
        output_dir=Path(args.output_dir),
        min_size=args.min_size,
        max_files=args.max_files,
    )

    print("=" * 72)
    print("Cache candidate images collected")
    print(f"Scanned images: {stats['scanned']}")
    print(f"Copied images: {stats['copied']}")
    print(f"maybe_glasses: {stats['maybe_glasses']}")
    print(f"maybe_head: {stats['maybe_head']}")
    print(f"maybe_text: {stats['maybe_text']}")
    print(f"maybe_other: {stats['maybe_other']}")
    print(f"candidate_index.csv: {stats['index_path']}")
    print("=" * 72)


if __name__ == "__main__":
    main()
