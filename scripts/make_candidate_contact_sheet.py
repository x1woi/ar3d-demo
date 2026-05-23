from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import List, Tuple

from PIL import Image, ImageDraw, ImageFont


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def list_images(source_dir: Path) -> List[Path]:
    if not source_dir.exists():
        raise FileNotFoundError(f"source_dir does not exist: {source_dir}")
    images = [
        path for path in source_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS
    ]
    images.sort(key=lambda path: path.name.lower())
    return images


def fit_image(img: Image.Image, thumb_size: int) -> Image.Image:
    img = img.convert("RGB")
    img.thumbnail((thumb_size, thumb_size), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (thumb_size, thumb_size), "white")
    x = (thumb_size - img.width) // 2
    y = (thumb_size - img.height) // 2
    canvas.paste(img, (x, y))
    return canvas


def truncate_text(text: str, max_chars: int = 26) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."


def safe_sheet_name(source_dir: Path) -> str:
    name = source_dir.name.strip() or "images"
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in name)


def image_dimensions(path: Path) -> Tuple[int, int]:
    with Image.open(path) as img:
        return img.size


def make_contact_sheet(
    source_dir: Path,
    output_dir: Path,
    max_images: int = 100,
    thumb_size: int = 160,
    cols: int = 5,
) -> Tuple[Path, Path, int, int]:
    source_dir = Path(source_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_images = list_images(source_dir)
    images = all_images[:max_images]

    cell_w = thumb_size + 28
    cell_h = thumb_size + 54
    margin = 18
    rows = max(1, math.ceil(len(images) / max(1, cols)))
    sheet_w = margin * 2 + cols * cell_w
    sheet_h = margin * 2 + rows * cell_h

    sheet = Image.new("RGB", (sheet_w, sheet_h), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()

    rows_for_csv = []
    for idx, path in enumerate(images, start=1):
        row = (idx - 1) // cols
        col = (idx - 1) % cols
        x = margin + col * cell_w
        y = margin + row * cell_h

        try:
            with Image.open(path) as img:
                fitted = fit_image(img, thumb_size)
                width, height = img.size
        except Exception:
            fitted = Image.new("RGB", (thumb_size, thumb_size), (240, 240, 240))
            width, height = 0, 0

        sheet.paste(fitted, (x, y + 24))
        draw.text((x, y), f"[{idx}] {truncate_text(path.name)}", fill="black", font=font)

        rows_for_csv.append(
            {
                "index": idx,
                "filename": path.name,
                "source_path": str(path),
                "width": width,
                "height": height,
                "file_size_kb": round(path.stat().st_size / 1024, 3),
            }
        )

    sheet_name = safe_sheet_name(source_dir)
    sheet_path = output_dir / f"contact_sheet_{sheet_name}.png"
    csv_path = output_dir / f"contact_sheet_{sheet_name}.csv"
    sheet.save(sheet_path)

    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        fieldnames = ["index", "filename", "source_path", "width", "height", "file_size_kb"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_for_csv)

    return sheet_path, csv_path, len(all_images), len(images)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a contact sheet for cache candidate images."
    )
    parser.add_argument(
        "--source-dir",
        type=str,
        default="paper_repro_outputs/cache_candidate_images/all",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="paper_repro_outputs/cache_candidate_sheets",
    )
    parser.add_argument("--max-images", type=int, default=100)
    parser.add_argument("--thumb-size", type=int, default=160)
    parser.add_argument("--cols", type=int, default=5)
    args = parser.parse_args()

    sheet_path, csv_path, total, used = make_contact_sheet(
        source_dir=Path(args.source_dir),
        output_dir=Path(args.output_dir),
        max_images=args.max_images,
        thumb_size=args.thumb_size,
        cols=args.cols,
    )

    print("=" * 72)
    print(f"source_dir: {args.source_dir}")
    print(f"total_images_found: {total}")
    print(f"images_in_sheet: {used}")
    print(f"contact_sheet_path: {sheet_path}")
    print(f"csv_path: {csv_path}")
    print("=" * 72)


if __name__ == "__main__":
    main()
