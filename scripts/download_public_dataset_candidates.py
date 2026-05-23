from __future__ import annotations

import argparse
import csv
import json
import math
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from PIL import Image, ImageDraw, ImageFont


OUTPUT_ROOT = Path("public_dataset_samples")
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEFAULT_CLASSES = ["face", "glasses", "hand", "cup", "book", "keyboard", "background"]
SUBDIRS = ["face", "glasses", "hard_negative", "negative", "objects"]

CLASS_QUERIES = {
    "face": "human face portrait",
    "glasses": "eyeglasses isolated",
    "hand": "human hand close up",
    "cup": "drinking cup object",
    "book": "book object",
    "keyboard": "computer keyboard",
    "background": "desktop background object",
}


def safe_name(text: str) -> str:
    text = re.sub(r"[^0-9A-Za-z._-]+", "_", text.strip())
    text = text.strip("._")
    return text[:80] or "image"


def parse_csv_list(text: str) -> List[str]:
    return [item.strip() for item in text.split(",") if item.strip()]


def class_to_subdir(class_name: str) -> str:
    if class_name == "face":
        return "face"
    if class_name == "glasses":
        return "glasses"
    if class_name == "hand":
        return "hard_negative"
    if class_name in {"cup", "book", "keyboard", "background"}:
        return "negative"
    return "objects"


def suggested_usage(class_name: str) -> str:
    if class_name in {"face", "glasses"}:
        return "likely_near_positive"
    if class_name == "hand":
        return "likely_hard_negative"
    if class_name in {"cup", "book", "keyboard", "background"}:
        return "likely_negative"
    return "unknown"


def ensure_dirs(output_root: Path) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    for name in SUBDIRS:
        (output_root / name).mkdir(parents=True, exist_ok=True)


def write_readme(output_root: Path) -> None:
    readme = output_root / "README.md"
    readme.write_text(
        """# public_dataset_samples

这个目录只用于保存公开来源的小规模候选图片，不是正式 v3_real 数据集。

- `face/`：人脸、人头候选图，通常只作为 near_positive 候选。
- `glasses/`：眼镜候选图，通常只作为 near_positive 候选。
- `hard_negative/`：手、遮挡、相似但不应复用的候选图。
- `negative/`：杯子、书、键盘、背景等明显负样本候选图。
- `objects/`：其他公开候选图。

这些图片必须人工检查后，才能使用 `sort_cache_candidates_helper.py` 或手动复制到 `cache_test_v3_real/`。
不要直接把本目录当作正式实验标签。
""",
        encoding="utf-8",
    )


def write_download_instructions(output_root: Path) -> Path:
    ensure_dirs(output_root)
    write_readme(output_root)
    path = output_root / "download_instructions.md"
    path.write_text(
        """# 公开候选样本手动下载说明

当前环境无法联网，或公开图片下载失败。请手动下载少量公开样本后放入 `public_dataset_samples/` 对应目录。

建议来源：

- face：可少量参考 WIDER FACE / LFW 或 Wikimedia Commons 中公开许可的人脸图片。
- glasses / hand / cup / book / keyboard：可少量参考 Open Images、COCO 或 Wikimedia Commons 中公开许可图片。
- background：可使用项目中的真实背景、桌面、非目标区域截图，也可以少量参考公开许可图片。

注意：

- 每类只建议 10-20 张，不要下载完整 Open Images / COCO / WIDER FACE。
- 这些图片只作为候选样本池。
- 必须人工确认后，才允许分拣到 `cache_test_v3_real/`。
- 不自动设置 positive / near_positive / hard_negative / negative。
- 不自动设置 should_hit。
""",
        encoding="utf-8",
    )
    return path


def commons_api_url(query: str, limit: int) -> str:
    params = {
        "action": "query",
        "format": "json",
        "generator": "search",
        "gsrnamespace": "6",
        "gsrlimit": str(limit),
        "gsrsearch": query,
        "prop": "imageinfo",
        "iiprop": "url|mime|size|extmetadata",
        "iiurlwidth": "900",
        "origin": "*",
    }
    return "https://commons.wikimedia.org/w/api.php?" + urllib.parse.urlencode(params)


def fetch_commons_candidates(class_name: str, max_items: int) -> List[Dict[str, str]]:
    query = CLASS_QUERIES.get(class_name, class_name)
    url = commons_api_url(query, max(max_items * 4, 20))
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "AR-cache-similarity-research/1.0 (small candidate download)",
        },
    )
    with urllib.request.urlopen(request, timeout=20) as response:
        payload = json.loads(response.read().decode("utf-8"))

    pages = payload.get("query", {}).get("pages", {})
    results: List[Dict[str, str]] = []
    for page in pages.values():
        title = page.get("title", "")
        info_list = page.get("imageinfo") or []
        if not info_list:
            continue
        info = info_list[0]
        mime = str(info.get("mime", "")).lower()
        image_url = info.get("thumburl") or info.get("url")
        if not image_url:
            continue
        suffix = Path(urllib.parse.urlparse(image_url).path).suffix.lower()
        if mime not in {"image/jpeg", "image/png", "image/webp"} and suffix not in IMAGE_EXTS:
            continue
        metadata = info.get("extmetadata", {}) or {}
        license_short = metadata.get("LicenseShortName", {}).get("value", "")
        artist = metadata.get("Artist", {}).get("value", "")
        license_note = f"Wikimedia Commons; {license_short}".strip("; ")
        results.append(
            {
                "title": title,
                "url": image_url,
                "source": "Wikimedia Commons",
                "license_note": license_note,
                "artist": re.sub(r"<[^>]+>", "", str(artist))[:120],
            }
        )
        if len(results) >= max_items:
            break
    return results


def download_image(url: str, target_path: Path) -> None:
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "AR-cache-similarity-research/1.0 (small candidate download)",
        },
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        data = response.read()
    target_path.write_bytes(data)
    with Image.open(target_path) as img:
        img.verify()


def image_size(path: Path) -> Tuple[int, int]:
    with Image.open(path) as img:
        return img.size


def truncate(text: str, max_len: int = 26) -> str:
    return text if len(text) <= max_len else text[: max_len - 3] + "..."


def fit_image(img: Image.Image, thumb_size: int) -> Image.Image:
    img = img.convert("RGB")
    img.thumbnail((thumb_size, thumb_size), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (thumb_size, thumb_size), "white")
    canvas.paste(img, ((thumb_size - img.width) // 2, (thumb_size - img.height) // 2))
    return canvas


def make_contact_sheet(rows: List[Dict[str, str]], output_root: Path) -> Tuple[Path, Path]:
    sheet_path = output_root / "contact_sheet_public_samples.png"
    csv_path = output_root / "contact_sheet_public_samples.csv"
    thumb_size = 150
    cols = 5
    cell_w = 210
    cell_h = 210
    margin = 18
    rows_count = max(1, math.ceil(len(rows) / cols))
    sheet = Image.new("RGB", (margin * 2 + cols * cell_w, margin * 2 + rows_count * cell_h), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()

    sheet_rows = []
    for index, row in enumerate(rows, start=1):
        path = Path(row["local_path"])
        grid_row = (index - 1) // cols
        grid_col = (index - 1) % cols
        x = margin + grid_col * cell_w
        y = margin + grid_row * cell_h
        width = height = 0
        try:
            with Image.open(path) as img:
                width, height = img.size
                fitted = fit_image(img, thumb_size)
        except Exception:
            fitted = Image.new("RGB", (thumb_size, thumb_size), (235, 235, 235))

        sheet.paste(fitted, (x, y + 42))
        label = f"[{index}] {row['class_name']} {truncate(path.name, 20)}"
        usage = truncate(row["suggested_usage"], 28)
        draw.text((x, y), label, fill="black", font=font)
        draw.text((x, y + 18), usage, fill="black", font=font)
        sheet_rows.append(
            {
                "index": index,
                "class_name": row["class_name"],
                "filename": path.name,
                "source_path": str(path),
                "width": width,
                "height": height,
                "file_size_kb": round(path.stat().st_size / 1024, 3) if path.exists() else 0,
                "suggested_usage": row["suggested_usage"],
            }
        )

    sheet.save(sheet_path)
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        fieldnames = [
            "index",
            "class_name",
            "filename",
            "source_path",
            "width",
            "height",
            "file_size_kb",
            "suggested_usage",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(sheet_rows)
    return sheet_path, csv_path


def write_labels_draft(rows: List[Dict[str, str]], output_root: Path) -> Path:
    path = output_root / "labels_draft.json"
    samples = []
    for row in rows:
        samples.append(
            {
                "image": str(Path(row["local_path"]).relative_to(output_root)).replace("\\", "/"),
                "class_name": row["class_name"],
                "category": "",
                "sample_type": "",
                "query_text": "",
                "should_hit": None,
                "suggested_usage": row["suggested_usage"],
                "needs_manual_review": True,
                "source": row["source"],
                "original_url": row["original_url"],
                "license_note": row["license_note"],
            }
        )
    path.write_text(json.dumps({"samples": samples}, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def write_candidate_index(rows: List[Dict[str, str]], output_root: Path) -> Path:
    path = output_root / "candidate_index.csv"
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        fieldnames = [
            "index",
            "source",
            "class_name",
            "local_path",
            "original_url",
            "license_note",
            "suggested_usage",
            "needs_manual_review",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for index, row in enumerate(rows, start=1):
            item = dict(row)
            item["index"] = index
            writer.writerow(item)
    return path


def write_download_log(rows: List[Dict[str, str]], output_root: Path) -> Path:
    path = output_root / "download_log.csv"
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        fieldnames = [
            "source",
            "class_name",
            "local_path",
            "original_url",
            "license_note",
            "suggested_usage",
            "needs_manual_review",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def print_plan(classes: Iterable[str], max_per_class: int, output_root: Path) -> None:
    print("=" * 72)
    print("Public dataset candidate download dry-run")
    print(f"output_dir: {output_root}")
    print(f"max_per_class: {max_per_class}")
    print("classes:")
    for class_name in classes:
        print(
            f"  - {class_name}: query='{CLASS_QUERIES.get(class_name, class_name)}', "
            f"target_dir={class_to_subdir(class_name)}, suggested_usage={suggested_usage(class_name)}"
        )
    print("No files will be downloaded in dry-run mode.")
    print("=" * 72)


def download_candidates(classes: List[str], max_per_class: int, output_root: Path) -> Dict[str, object]:
    ensure_dirs(output_root)
    write_readme(output_root)
    rows: List[Dict[str, str]] = []
    online_ok = True
    errors: List[str] = []

    for class_name in classes:
        try:
            candidates = fetch_commons_candidates(class_name, max_per_class)
        except Exception as exc:
            online_ok = False
            errors.append(f"{class_name}: {exc}")
            continue

        target_dir = output_root / class_to_subdir(class_name)
        usage = suggested_usage(class_name)
        for idx, candidate in enumerate(candidates, start=1):
            parsed_path = Path(urllib.parse.urlparse(candidate["url"]).path)
            suffix = parsed_path.suffix.lower()
            if suffix not in IMAGE_EXTS:
                suffix = ".jpg"
            filename = f"{class_name}_{idx:03d}_{safe_name(candidate['title'])}{suffix}"
            local_path = target_dir / filename
            try:
                download_image(candidate["url"], local_path)
                image_size(local_path)
            except Exception as exc:
                if local_path.exists():
                    local_path.unlink()
                errors.append(f"{class_name}/{idx}: {exc}")
                continue

            rows.append(
                {
                    "source": candidate["source"],
                    "class_name": class_name,
                    "local_path": str(local_path),
                    "original_url": candidate["url"],
                    "license_note": candidate["license_note"],
                    "suggested_usage": usage,
                    "needs_manual_review": "true",
                }
            )
            time.sleep(0.15)

    log_path = write_download_log(rows, output_root)
    index_path = write_candidate_index(rows, output_root)
    labels_path = write_labels_draft(rows, output_root)
    sheet_path, sheet_csv_path = make_contact_sheet(rows, output_root)
    if not rows:
        write_download_instructions(output_root)
    return {
        "online_ok": online_ok and bool(rows),
        "download_count": len(rows),
        "download_log": log_path,
        "candidate_index": index_path,
        "labels_draft": labels_path,
        "sheet": sheet_path,
        "sheet_csv": sheet_csv_path,
        "errors": errors,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download a small public-image candidate pool for v3_real manual review."
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-per-class", type=int, default=10)
    parser.add_argument(
        "--classes",
        type=str,
        default=",".join(DEFAULT_CLASSES),
        help="Comma-separated class list.",
    )
    parser.add_argument("--output-dir", type=str, default=str(OUTPUT_ROOT))
    args = parser.parse_args()

    classes = parse_csv_list(args.classes) or DEFAULT_CLASSES
    max_per_class = max(1, min(args.max_per_class, 20))
    output_root = Path(args.output_dir)

    if args.dry_run:
        print_plan(classes, max_per_class, output_root)
        return

    try:
        stats = download_candidates(classes, max_per_class, output_root)
    except Exception as exc:
        instructions = write_download_instructions(output_root)
        print("=" * 72)
        print("当前环境无法联网，请手动下载少量公开数据集样本放入 public_dataset_samples/ 对应目录。")
        print(f"error: {exc}")
        print(f"download_instructions.md: {instructions}")
        print("=" * 72)
        return

    if not stats["online_ok"]:
        instructions = write_download_instructions(output_root)
        print("当前环境无法联网，请手动下载少量公开数据集样本放入 public_dataset_samples/ 对应目录。")
        print(f"download_instructions.md: {instructions}")

    print("=" * 72)
    print(f"联网成功: {bool(stats['online_ok'])}")
    print(f"下载数量: {stats['download_count']}")
    print(f"public_dataset_samples 路径: {output_root}")
    print(f"download_log.csv 路径: {stats['download_log']}")
    print(f"candidate_index.csv 路径: {stats['candidate_index']}")
    print(f"labels_draft.json 路径: {stats['labels_draft']}")
    print(f"contact_sheet_public_samples.png 路径: {stats['sheet']}")
    print(f"contact_sheet_public_samples.csv 路径: {stats['sheet_csv']}")
    errors = stats.get("errors") or []
    if errors:
        print(f"warnings: {len(errors)} download/search errors; see terminal output for details.")
        for item in errors[:8]:
            print(f"  - {item}")
    print("=" * 72)


if __name__ == "__main__":
    main()
