from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, List


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def list_images(image_dirs: List[Path], recursive: bool = False) -> List[Path]:
    images: List[Path] = []
    for image_dir in image_dirs:
        image_dir = Path(image_dir)
        if not image_dir.exists():
            raise FileNotFoundError(f"Image directory does not exist: {image_dir}")
        iterator = image_dir.rglob("*") if recursive else image_dir.iterdir()
        for path in iterator:
            if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
                images.append(path)
    images.sort(key=lambda path: str(path).lower())
    return images


def unique_output_name(path: Path, used_names: Dict[str, int]) -> str:
    stem = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in path.stem)
    stem = stem[:48] or "query"
    suffix = path.suffix.lower() or ".jpg"
    base = f"{stem}{suffix}"
    count = used_names.get(base, 0)
    used_names[base] = count + 1
    if count == 0:
        return base
    return f"{stem}_{count:03d}{suffix}"


def build_dataset(image_dirs: List[Path], output_dir: Path, recursive: bool = False) -> Path:
    output_dir = Path(output_dir)
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    images = list_images(image_dirs, recursive=recursive)
    used_names: Dict[str, int] = {}
    samples = []

    for src in images:
        dst_name = unique_output_name(src, used_names)
        dst = images_dir / dst_name
        shutil.copy2(src, dst)
        samples.append(
            {
                "image": f"images/{dst_name}",
                "category": "",
                "sample_type": "positive",
                "query_text": "",
                "should_hit": True,
                "source_path": str(src),
                "notes": "",
            }
        )

    labels = {
        "schema": "cache_similarity_dataset_labels.v1",
        "description": "Fill category, sample_type, query_text, and should_hit manually before running experiment_cache_similarity.py.",
        "sample_type_options": ["positive", "near_positive", "negative"],
        "samples": samples,
    }

    labels_path = output_dir / "labels_template.json"
    labels_path.write_text(
        json.dumps(labels, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return labels_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a cache similarity evaluation dataset template from query image directories."
    )
    parser.add_argument(
        "--image-dir",
        action="append",
        required=True,
        help="Input query image directory. Can be specified multiple times.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="paper_repro_outputs/cache_similarity_dataset_v1",
        help="Output dataset directory.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively scan image directories.",
    )

    args = parser.parse_args()
    labels_path = build_dataset(
        image_dirs=[Path(item) for item in args.image_dir],
        output_dir=Path(args.output_dir),
        recursive=args.recursive,
    )

    print("=" * 72)
    print("Cache similarity dataset template created")
    print(f"Output dir: {args.output_dir}")
    print(f"Images dir: {Path(args.output_dir) / 'images'}")
    print(f"Labels template: {labels_path}")
    print("Next: copy labels_template.json to labels.json and fill category/sample_type/query_text/should_hit.")
    print("=" * 72)


if __name__ == "__main__":
    main()
