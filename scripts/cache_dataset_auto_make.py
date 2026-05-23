from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


LABEL_RULES = [
    (("glasses", "eye", "yanjing", "眼镜"), "眼镜", "眼镜 帮助人看清东西的工具"),
    (("head", "face", "rentou", "人头", "人脸"), "人头", "人头 人的头部或脸部"),
    (("cup", "beizi", "杯子"), "杯子", "杯子 用来喝水的容器"),
    (("book", "shu", "书"), "书", "书 用来阅读的物品"),
    (("hand", "shou", "手"), "手", "手 人体用于抓握和操作的部位"),
]


def read_image(path: Path) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Could not read image: {path}")
    return img


def save_image(path: Path, img: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix or ".jpg"
    ok, buffer = cv2.imencode(ext, img)
    if not ok:
        raise RuntimeError(f"Could not encode image: {path}")
    buffer.tofile(str(path))


def safe_name(text: str) -> str:
    name = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in text.strip())
    return name[:48] or "sample"


def unique_path(path: Path) -> Path:
    if not path.exists():
        return path
    stem = path.stem
    suffix = path.suffix
    index = 1
    while True:
        candidate = path.with_name(f"{stem}_{index:03d}{suffix}")
        if not candidate.exists():
            return candidate
        index += 1


def writable_images_dir(output_dir: Path, preferred_name: str = "images") -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    preferred = output_dir / preferred_name
    preferred.mkdir(parents=True, exist_ok=True)
    probe = preferred / f".write_probe_{int(time.time() * 1000)}.tmp"
    try:
        probe.write_text("probe", encoding="utf-8")
        probe.unlink(missing_ok=True)
        return preferred
    except OSError:
        fallback = output_dir / f"{preferred_name}_rebuild_{time.strftime('%Y%m%d_%H%M%S')}"
        fallback.mkdir(parents=True, exist_ok=True)
        return fallback


def auto_label_from_filename(path: Path) -> Tuple[str, str]:
    name = path.stem.lower()
    for tokens, category, query_text in LABEL_RULES:
        if any(token.lower() in name for token in tokens):
            return category, query_text
    return "", ""


def rotate_image(img: np.ndarray, degrees: float) -> np.ndarray:
    h, w = img.shape[:2]
    center = (w / 2.0, h / 2.0)
    matrix = cv2.getRotationMatrix2D(center, degrees, 1.0)
    return cv2.warpAffine(
        img,
        matrix,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )


def adjust_brightness(img: np.ndarray, factor: float) -> np.ndarray:
    out = img.astype(np.float32) * factor
    return np.clip(out, 0, 255).astype(np.uint8)


def center_crop_resize(img: np.ndarray, crop_ratio: float = 0.86) -> np.ndarray:
    h, w = img.shape[:2]
    crop_w = max(1, int(w * crop_ratio))
    crop_h = max(1, int(h * crop_ratio))
    x1 = max(0, (w - crop_w) // 2)
    y1 = max(0, (h - crop_h) // 2)
    crop = img[y1 : y1 + crop_h, x1 : x1 + crop_w]
    return cv2.resize(crop, (w, h), interpolation=cv2.INTER_CUBIC)


def augment_images(img: np.ndarray) -> List[Tuple[str, np.ndarray]]:
    return [
        ("original", img.copy()),
        ("rotate_pos", rotate_image(img, 5.0)),
        ("rotate_neg", rotate_image(img, -5.0)),
        ("bright", adjust_brightness(img, 1.18)),
        ("dark", adjust_brightness(img, 0.78)),
        ("blur", cv2.GaussianBlur(img, (5, 5), 0)),
        ("center_crop", center_crop_resize(img, 0.86)),
    ]


def parse_base_image_specs(specs: List[str]) -> List[Tuple[Path, str]]:
    result: List[Tuple[Path, str]] = []
    for spec in specs:
        if "=" in spec:
            category, path_text = spec.split("=", 1)
        elif "|" in spec:
            path_text, category = spec.split("|", 1)
        else:
            raise ValueError(
                "Base image must use category=path or path|category format: "
                f"{spec}"
            )
        category = category.strip()
        path = Path(path_text.strip())
        if not category:
            raise ValueError(f"Base image category is empty: {spec}")
        if not path.exists():
            raise FileNotFoundError(f"Base image does not exist: {path}")
        result.append((path, category))
    return result


def list_images(image_dirs: List[Path], recursive: bool = False) -> List[Path]:
    images: List[Path] = []
    for image_dir in image_dirs:
        if not image_dir.exists():
            raise FileNotFoundError(f"Image directory does not exist: {image_dir}")
        iterator = image_dir.rglob("*") if recursive else image_dir.iterdir()
        for path in iterator:
            if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
                images.append(path)
    return sorted(images, key=lambda path: str(path).lower())


def build_auto_dataset(
    base_specs: List[str],
    output_dir: Path,
    negative_dirs: List[Path],
    recursive_negative: bool = False,
) -> Path:
    output_dir = Path(output_dir)
    images_dir = writable_images_dir(output_dir)

    samples = []

    for base_index, (base_path, category) in enumerate(parse_base_image_specs(base_specs)):
        img = read_image(base_path)
        category_name = safe_name(category)
        base_name = safe_name(base_path.stem)
        for aug_name, aug_img in augment_images(img):
            filename = f"pos_{base_index:02d}_{category_name}_{base_name}_{aug_name}.jpg"
            dst = images_dir / filename
            save_image(dst, aug_img)
            samples.append(
                {
                    "image": f"{images_dir.name}/{filename}",
                    "category": category,
                    "sample_type": "positive",
                    "query_text": category,
                    "should_hit": True,
                    "source_path": str(base_path),
                    "augmentation": aug_name,
                    "notes": "",
                }
            )

    negative_images = list_images(negative_dirs, recursive=recursive_negative) if negative_dirs else []
    for neg_index, src in enumerate(negative_images):
        filename = f"neg_{neg_index:03d}_{safe_name(src.stem)}{src.suffix.lower() or '.jpg'}"
        dst = images_dir / filename
        shutil.copy2(src, dst)
        samples.append(
            {
                "image": f"{images_dir.name}/{filename}",
                "category": "",
                "sample_type": "negative",
                "query_text": "",
                "should_hit": False,
                "source_path": str(src),
                "augmentation": "negative_copy",
                "notes": "",
            }
        )

    labels = {
        "schema": "cache_similarity_dataset_labels.v1",
        "description": "Automatically generated cache similarity dataset. Edit category/query_text for negative samples if needed.",
        "samples": samples,
    }
    labels_path = output_dir / "labels.json"
    labels_path.write_text(
        json.dumps(labels, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return labels_path


def add_labeled_directory_samples(
    samples: list,
    images_dir: Path,
    image_dirs: List[Path],
    sample_type: str,
    should_hit: bool,
    prefix: str,
    recursive: bool = False,
    auto_fill_labels: bool = False,
) -> None:
    for index, src in enumerate(list_images(image_dirs, recursive=recursive)):
        suffix = src.suffix.lower() or ".jpg"
        filename = f"{prefix}_{index:03d}_{safe_name(src.stem)}{suffix}"
        dst = unique_path(images_dir / filename)
        filename = dst.name
        shutil.copy2(src, dst)
        category, query_text = auto_label_from_filename(src) if auto_fill_labels else ("", "")
        samples.append(
            {
                "image": f"{images_dir.name}/{filename}",
                "category": category,
                "sample_type": sample_type,
                "query_text": query_text,
                "should_hit": should_hit,
                "source_path": str(src),
                "augmentation": "directory_copy",
                "notes": "",
            }
        )


def build_hard_dataset(
    output_dir: Path,
    positive_dirs: List[Path],
    near_positive_dirs: List[Path],
    hard_negative_dirs: List[Path],
    negative_dirs: List[Path],
    recursive: bool = False,
    auto_fill_labels: bool = False,
) -> Path:
    output_dir = Path(output_dir)
    images_dir = writable_images_dir(output_dir)
    samples = []

    add_labeled_directory_samples(
        samples,
        images_dir,
        positive_dirs,
        sample_type="positive",
        should_hit=True,
        prefix="pos",
        recursive=recursive,
        auto_fill_labels=auto_fill_labels,
    )
    add_labeled_directory_samples(
        samples,
        images_dir,
        near_positive_dirs,
        sample_type="near_positive",
        should_hit=True,
        prefix="near",
        recursive=recursive,
        auto_fill_labels=auto_fill_labels,
    )
    add_labeled_directory_samples(
        samples,
        images_dir,
        hard_negative_dirs,
        sample_type="hard_negative",
        should_hit=False,
        prefix="hardneg",
        recursive=recursive,
        auto_fill_labels=auto_fill_labels,
    )
    add_labeled_directory_samples(
        samples,
        images_dir,
        negative_dirs,
        sample_type="negative",
        should_hit=False,
        prefix="neg",
        recursive=recursive,
        auto_fill_labels=auto_fill_labels,
    )

    labels = {
        "schema": "cache_similarity_dataset_labels.v1",
        "description": "Hard cache similarity dataset template. Fill category and query_text manually before evaluation.",
        "samples": samples,
    }
    labels_path = output_dir / "labels.json"
    labels_path.write_text(
        json.dumps(labels, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return labels_path


def print_next_steps(output_dir: str) -> None:
    print("Next steps:")
    print(f"1. 检查并补全 {Path(output_dir) / 'labels.json'} 中的 category 和 query_text。")
    print("2. 运行缓存相似度实验：")
    print(".\\.venv\\Scripts\\python.exe experiment_cache_similarity.py `")
    print(f"  --dataset-dir {output_dir} `")
    print("  --output-dir paper_repro_outputs\\cache_similarity_eval_v2_hard `")
    print("  --thresholds 0.5,0.6,0.7,0.75,0.8,0.82,0.85,0.9")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Automatically build cache similarity dataset with positive augmentations and negative images."
    )
    parser.add_argument(
        "--base-image",
        action="append",
        default=[],
        help="Base positive image. Use category=path or path|category. Can be repeated.",
    )
    parser.add_argument("--positive-dir", action="append", default=[])
    parser.add_argument("--near-positive-dir", action="append", default=[])
    parser.add_argument("--hard-negative-dir", action="append", default=[])
    parser.add_argument(
        "--negative-dir",
        action="append",
        default=[],
        help="Directory of negative query images. Can be repeated.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="paper_repro_outputs/cache_similarity_dataset_v1_auto",
    )
    parser.add_argument(
        "--recursive-negative",
        action="store_true",
        help="Recursively scan negative directories.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively scan positive/near-positive/hard-negative/negative directories.",
    )
    parser.add_argument(
        "--auto-fill-labels",
        action="store_true",
        help="Prefill category and query_text from simple filename rules.",
    )

    args = parser.parse_args()
    directory_mode = bool(
        args.positive_dir
        or args.near_positive_dir
        or args.hard_negative_dir
    )
    if directory_mode:
        labels_path = build_hard_dataset(
            output_dir=Path(args.output_dir),
            positive_dirs=[Path(item) for item in args.positive_dir],
            near_positive_dirs=[Path(item) for item in args.near_positive_dir],
            hard_negative_dirs=[Path(item) for item in args.hard_negative_dir],
            negative_dirs=[Path(item) for item in args.negative_dir],
            recursive=args.recursive or args.recursive_negative,
            auto_fill_labels=args.auto_fill_labels,
        )
    else:
        if not args.base_image:
            raise ValueError("Either --base-image or one of --positive-dir/--near-positive-dir/--hard-negative-dir is required.")
        labels_path = build_auto_dataset(
            base_specs=args.base_image,
            output_dir=Path(args.output_dir),
            negative_dirs=[Path(item) for item in args.negative_dir],
            recursive_negative=args.recursive_negative or args.recursive,
        )

    print("=" * 72)
    print("Automatic cache similarity dataset created")
    print(f"Output dir: {args.output_dir}")
    print(f"Images dir: {Path(args.output_dir) / 'images'}")
    print(f"Labels: {labels_path}")
    print_next_steps(args.output_dir)
    print("=" * 72)


if __name__ == "__main__":
    main()
