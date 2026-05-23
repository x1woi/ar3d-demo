from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import List


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
TARGET_TYPES = {"positive", "near_positive", "hard_negative", "negative"}


def list_images(source_dir: Path) -> List[Path]:
    if not source_dir.exists():
        raise FileNotFoundError(f"source_dir does not exist: {source_dir}")
    images = [
        path for path in source_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS
    ]
    images.sort(key=lambda path: path.name.lower())
    return images


def print_image_list(source_dir: Path, limit: int = 50) -> None:
    images = list_images(source_dir)
    print("=" * 72)
    print(f"source_dir: {source_dir}")
    print(f"total_images: {len(images)}")
    print("=" * 72)
    for index, path in enumerate(images[:limit], start=1):
        size_kb = path.stat().st_size / 1024
        print(f"{index:03d}. {path.name}  {size_kb:.1f} KB")
    if len(images) > limit:
        print(f"... {len(images) - limit} more images not shown")


def parse_indices(text: str) -> List[int]:
    if not text.strip():
        return []
    result: List[int] = []
    for part in text.replace(";", ",").split(","):
        part = part.strip()
        if not part:
            continue
        result.append(int(part))
    return result


def resolve_selected_files(source_dir: Path, files: List[str], indices: str) -> List[Path]:
    images = list_images(source_dir)
    selected: List[Path] = []

    for filename in files:
        path = source_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"candidate file does not exist: {path}")
        if path.suffix.lower() not in IMAGE_EXTS:
            raise ValueError(f"not an image file: {path}")
        selected.append(path)

    for index in parse_indices(indices):
        if index < 1 or index > len(images):
            raise IndexError(f"index out of range: {index}; valid range is 1..{len(images)}")
        selected.append(images[index - 1])

    deduped: List[Path] = []
    seen = set()
    for path in selected:
        key = str(path.resolve()).lower()
        if key not in seen:
            seen.add(key)
            deduped.append(path)
    return deduped


def next_target_path(target_dir: Path, target_type: str, original_name: str) -> Path:
    suffix = Path(original_name).suffix.lower() or ".jpg"
    stem = Path(original_name).stem
    safe_stem = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in stem)
    safe_stem = safe_stem[:80] or "image"

    index = 1
    while True:
        candidate = target_dir / f"{target_type}_{index:03d}_{safe_stem}{suffix}"
        if not candidate.exists():
            return candidate
        index += 1


def copy_candidates(source_dir: Path, target_root: Path, target_type: str, files: List[str], indices: str) -> List[Path]:
    if target_type not in TARGET_TYPES:
        raise ValueError(f"target_type must be one of: {', '.join(sorted(TARGET_TYPES))}")
    selected = resolve_selected_files(source_dir, files, indices)
    if not selected:
        raise ValueError("no files selected; use --files or --indices")

    target_dir = target_root / target_type
    target_dir.mkdir(parents=True, exist_ok=True)

    copied: List[Path] = []
    for src in selected:
        dst = next_target_path(target_dir, target_type, src.name)
        shutil.copy2(src, dst)
        copied.append(dst)
    return copied


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Helper for manually sorting cache candidate images into cache_test_v2 folders."
    )
    parser.add_argument(
        "--source-dir",
        type=str,
        default="paper_repro_outputs/cache_candidate_images/all",
    )
    parser.add_argument("--target-type", type=str, choices=sorted(TARGET_TYPES))
    parser.add_argument("--files", nargs="*", default=[])
    parser.add_argument("--indices", type=str, default="")
    parser.add_argument("--target-root", type=str, default="cache_test_v2")
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()

    source_dir = Path(args.source_dir)
    if args.list:
        print_image_list(source_dir)
        return

    if not args.target_type:
        raise SystemExit("--target-type is required unless --list is used")

    copied = copy_candidates(
        source_dir=source_dir,
        target_root=Path(args.target_root),
        target_type=args.target_type,
        files=args.files,
        indices=args.indices,
    )

    print("=" * 72)
    print(f"source_dir: {source_dir}")
    print(f"target_type: {args.target_type}")
    print(f"target_dir: {Path(args.target_root) / args.target_type}")
    print(f"copied_count: {len(copied)}")
    print("copied_files:")
    for path in copied:
        print(f"- {path}")
    print("=" * 72)


if __name__ == "__main__":
    main()
