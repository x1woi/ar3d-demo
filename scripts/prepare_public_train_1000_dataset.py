#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Prepare a small public_train_1000 ROI dataset from public detection datasets.

The script is conservative by design. If FiftyOne is not installed, it writes
an actionable report instead of installing dependencies or touching existing
v3_real_70 results.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from PIL import Image, ImageDraw, ImageFont


OUTPUT_DIR = Path("paper_repro_outputs/cache_similarity_eval_public_train_1000")
ROI_DIR = OUTPUT_DIR / "roi_images"
METADATA_CSV = OUTPUT_DIR / "public_train_1000_metadata.csv"
CONTACT_SHEET = OUTPUT_DIR / "public_train_1000_contact_sheet.jpg"
PREPARE_REPORT = OUTPUT_DIR / "public_train_1000_prepare_report.md"
PREPARE_JSON = OUTPUT_DIR / "public_train_1000_prepare_report.json"
RANDOM_STATE = 42

TARGET_COUNTS = {
    "positive": 200,
    "near_positive": 300,
    "hard_negative": 300,
    "negative": 200,
}

TARGET_BY_CANDIDATE = {
    "glasses": 250,
    "keyboard": 250,
    "tissue_box": 250,
    "tennis_racket": 250,
}

CSV_FIELDS = [
    "sample_id",
    "roi_image_path",
    "source_dataset",
    "source_image_path",
    "source_image_id",
    "bbox",
    "query_text",
    "keyword",
    "candidate_cache_name",
    "candidate_glb_path",
    "sample_type",
    "should_hit",
    "object_category",
    "matched_dataset_class",
    "license_note",
    "split",
    "notes",
]


CLASS_MAP = {
    "glasses": {
        "display_name": "眼镜",
        "candidate_glb_paths": [
            "runtime_assets/model_cache/眼镜_db9b92eef7.glb",
            "runtime_assets/competition_demo_models/glasses.glb",
        ],
        "positive": ["Glasses", "Sunglasses", "Eyeglasses"],
        "near_positive": ["Goggles", "Safety glasses"],
        "hard_negative": ["Headphones", "Necklace", "Watch", "Bracelet", "Scissors", "Key"],
        "negative": ["Bottle", "Cup", "Book", "Cell phone", "Mouse"],
    },
    "keyboard": {
        "display_name": "键盘",
        "candidate_glb_paths": [
            "runtime_assets/model_cache/电脑键盘_1e24316d44.glb",
            "runtime_assets/competition_demo_models/keyboard.glb",
        ],
        "positive": ["Computer keyboard", "Keyboard"],
        "near_positive": ["Laptop keyboard", "Computer"],
        "hard_negative": ["Laptop", "Calculator", "Remote control", "Book", "Tablet computer"],
        "negative": ["Cup", "Bottle", "Chair", "Handbag", "Spoon"],
    },
    "tissue_box": {
        "display_name": "纸巾盒",
        "candidate_glb_paths": [
            "runtime_assets/model_cache/纸巾盒_ea7bcc88a0.glb",
            "runtime_assets/competition_demo_models/tissue_box.glb",
        ],
        "positive": ["Box", "Tissue box", "Container"],
        "near_positive": ["Carton", "Package", "Plastic container"],
        "hard_negative": ["Book", "Laptop", "Suitcase", "Handbag", "Microwave"],
        "negative": ["Cup", "Bottle", "Keyboard", "Tennis racket", "Mouse"],
    },
    "tennis_racket": {
        "display_name": "网球拍",
        "candidate_glb_paths": [
            "runtime_assets/model_cache/网球拍_efa3e2b1e3.glb",
            "runtime_assets/competition_demo_models/tennis_racket.glb",
        ],
        "positive": ["Tennis racket", "Racket"],
        "near_positive": ["Badminton racket", "Squash racket"],
        "hard_negative": ["Baseball bat", "Skateboard", "Surfboard", "Umbrella", "Ski"],
        "negative": ["Cup", "Bottle", "Book", "Keyboard", "Mouse"],
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare public_train_1000 from Open Images/COCO via FiftyOne.")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--max-samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=RANDOM_STATE)
    parser.add_argument("--dataset-priority", default="open-images-v7,coco-2017")
    parser.add_argument("--dry-run", action="store_true", help="Only write plan/report; do not download.")
    return parser.parse_args()


def ensure_dirs(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    for sample_type in TARGET_COUNTS:
        (root / "roi_images" / sample_type).mkdir(parents=True, exist_ok=True)


def normalize_label(text: str) -> str:
    return "".join(ch.lower() for ch in text if ch.isalnum())


def resolve_glb(paths: List[str]) -> Tuple[str, bool]:
    for path in paths:
        if Path(path).exists():
            return path, True
    return paths[0] if paths else "", False


def all_requested_classes() -> List[str]:
    classes: List[str] = []
    for spec in CLASS_MAP.values():
        for sample_type in TARGET_COUNTS:
            classes.extend(spec[sample_type])
    seen = set()
    out = []
    for cls in classes:
        key = normalize_label(cls)
        if key not in seen:
            seen.add(key)
            out.append(cls)
    return out


def class_to_candidates() -> Dict[str, List[Tuple[str, str, str]]]:
    mapping: Dict[str, List[Tuple[str, str, str]]] = defaultdict(list)
    for candidate, spec in CLASS_MAP.items():
        for sample_type in TARGET_COUNTS:
            for cls in spec[sample_type]:
                mapping[normalize_label(cls)].append((candidate, sample_type, cls))
    return mapping


def write_empty_metadata(path: Path) -> None:
    if path.exists():
        return
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        csv.DictWriter(f, fieldnames=CSV_FIELDS).writeheader()


def make_placeholder_contact_sheet(path: Path, message: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (1200, 500), "white")
    draw = ImageDraw.Draw(img)
    lines = ["public_train_1000 contact sheet", message]
    y = 180
    for line in lines:
        draw.text((60, y), line, fill=(30, 30, 30))
        y += 48
    img.save(path, quality=92)


def write_report(root: Path, payload: Dict[str, Any]) -> None:
    counts = payload.get("count_by_sample_type", {})
    by_candidate = payload.get("count_by_candidate_cache_name", {})
    missing_classes = payload.get("missing_classes", [])
    missing_glb = payload.get("missing_glb", [])
    failed_reason = payload.get("failed_download_reason", "")
    text = f"""# public_train_1000 数据集构造报告

## 1. 构造目的

本报告记录公开数据集训练分支 public_train_1000 的构造状态。该分支用于后续训练学习式图文融合分类器；公开数据用于训练，真实摄像头 ROI 后续用于测试。

## 2. 数据来源

- 优先数据集：Open Images V7 / V6
- 备选数据集：COCO 2017
- 实际使用：{payload.get("source_dataset_used", "none")}

## 3. 样本统计

- total_samples: {payload.get("total_samples", 0)}
- count_by_sample_type: {counts}
- count_by_candidate_cache_name: {by_candidate}
- count_by_source_dataset: {payload.get("count_by_source_dataset", {})}

## 4. 缺失与失败信息

- missing_classes: {missing_classes}
- missing_glb: {missing_glb}
- failed_download_reason: {failed_reason or "无"}

## 5. 依赖状态

- FiftyOne 可用：{payload.get("fiftyone_available", False)}
- 如果未安装，可选安装命令：`pip install fiftyone`

## 6. 注意事项

本脚本不会安装 FiftyOne，不会下载全量数据集，不会调用 Qwen / TripoSR / Stable Fast 3D，也不会修改 plus.py。公开数据集没有天然 should_hit 标签，本分支的标签来自候选缓存模型与类别映射，后续需要真实 ROI 测试验证。
"""
    (root / "public_train_1000_prepare_report.md").write_text(text, encoding="utf-8-sig")
    (root / "public_train_1000_prepare_report.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def crop_roi(image_path: Path, bbox: List[float], output_path: Path) -> bool:
    try:
        with Image.open(image_path) as img:
            width, height = img.size
            x, y, w, h = bbox
            left = max(0, int(x * width))
            top = max(0, int(y * height))
            right = min(width, int((x + w) * width))
            bottom = min(height, int((y + h) * height))
            if right - left < 16 or bottom - top < 16:
                return False
            roi = img.crop((left, top, right, bottom)).convert("RGB")
            output_path.parent.mkdir(parents=True, exist_ok=True)
            roi.save(output_path, quality=92)
            return True
    except Exception:
        return False


def make_contact_sheet(rows: List[Dict[str, str]], path: Path, max_images: int = 80) -> None:
    if not rows:
        make_placeholder_contact_sheet(path, "No ROI images were generated.")
        return
    thumbs = []
    for row in rows[:max_images]:
        img_path = Path(row["roi_image_path"])
        if not img_path.exists():
            continue
        try:
            with Image.open(img_path) as img:
                thumb = img.convert("RGB")
                thumb.thumbnail((140, 120))
                thumbs.append((thumb.copy(), row))
        except Exception:
            continue
    if not thumbs:
        make_placeholder_contact_sheet(path, "No readable ROI images were generated.")
        return
    cols = 5
    cell_w, cell_h = 220, 170
    rows_n = (len(thumbs) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * cell_w, rows_n * cell_h), "white")
    draw = ImageDraw.Draw(sheet)
    for i, (thumb, row) in enumerate(thumbs):
        x = (i % cols) * cell_w
        y = (i // cols) * cell_h
        sheet.paste(thumb, (x + 8, y + 8))
        label = f"{row['sample_id']} {row['candidate_cache_name']} {row['sample_type']}"
        draw.text((x + 8, y + 132), label[:32], fill=(0, 0, 0))
        draw.text((x + 8, y + 150), row["matched_dataset_class"][:32], fill=(80, 80, 80))
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path, quality=92)


def try_load_fiftyone_dataset(dataset_name: str, classes: List[str], max_samples: int, seed: int):
    import fiftyone.zoo as foz

    if dataset_name.startswith("open-images"):
        return foz.load_zoo_dataset(
            dataset_name,
            split="validation",
            label_types=["detections"],
            classes=classes,
            max_samples=max_samples,
            shuffle=True,
            seed=seed,
            dataset_name=f"public_train_1000_{dataset_name.replace('-', '_')}",
        )
    if dataset_name == "coco-2017":
        return foz.load_zoo_dataset(
            "coco-2017",
            split="validation",
            label_types=["detections"],
            classes=classes,
            max_samples=max_samples,
            shuffle=True,
            seed=seed,
            dataset_name="public_train_1000_coco_2017",
        )
    raise ValueError(f"unsupported dataset: {dataset_name}")


def detection_iter(sample) -> Iterable[Tuple[str, List[float]]]:  # noqa: ANN001
    detections = None
    for field in ("ground_truth", "detections"):
        if hasattr(sample, field):
            detections = getattr(sample, field)
            break
    if not detections or not getattr(detections, "detections", None):
        return []
    out = []
    for det in detections.detections:
        label = getattr(det, "label", "")
        bbox = getattr(det, "bounding_box", None)
        if label and bbox:
            out.append((label, bbox))
    return out


def build_with_fiftyone(root: Path, dataset_priorities: List[str], max_samples: int, seed: int) -> Dict[str, Any]:
    requested_classes = all_requested_classes()
    class_map = class_to_candidates()
    selected_dataset = None
    failed_reasons = []
    for dataset_name in dataset_priorities:
        try:
            selected_dataset = try_load_fiftyone_dataset(dataset_name, requested_classes, max_samples * 3, seed)
            source_dataset_used = dataset_name
            break
        except Exception as exc:
            failed_reasons.append(f"{dataset_name}: {type(exc).__name__}: {exc}")
    if selected_dataset is None:
        raise RuntimeError("; ".join(failed_reasons))

    rng = random.Random(seed)
    sample_rows: List[Dict[str, str]] = []
    counts_by_type = Counter()
    counts_by_candidate = Counter()
    counts_by_source_image = Counter()
    counts_by_source_dataset = Counter()
    missing_glb = []
    sample_num = 1

    samples = list(selected_dataset)
    rng.shuffle(samples)
    for sample in samples:
        source_image_path = Path(sample.filepath)
        source_image_id = str(getattr(sample, "id", ""))
        if counts_by_source_image[source_image_id] >= 2:
            continue
        detections = list(detection_iter(sample))
        rng.shuffle(detections)
        for label, bbox in detections:
            key = normalize_label(label)
            options = class_map.get(key, [])
            if not options:
                continue
            rng.shuffle(options)
            for candidate_name, sample_type, matched_class in options:
                if counts_by_type[sample_type] >= TARGET_COUNTS[sample_type]:
                    continue
                if counts_by_candidate[candidate_name] >= TARGET_BY_CANDIDATE[candidate_name]:
                    continue
                spec = CLASS_MAP[candidate_name]
                glb_path, glb_exists = resolve_glb(spec["candidate_glb_paths"])
                if not glb_exists and candidate_name not in missing_glb:
                    missing_glb.append(candidate_name)
                sample_id = f"pub_{sample_num:06d}"
                roi_path = root / "roi_images" / sample_type / f"{sample_id}_{candidate_name}.jpg"
                if not crop_roi(source_image_path, bbox, roi_path):
                    continue
                row = {
                    "sample_id": sample_id,
                    "roi_image_path": str(roi_path),
                    "source_dataset": source_dataset_used,
                    "source_image_path": str(source_image_path),
                    "source_image_id": source_image_id,
                    "bbox": json.dumps([round(float(v), 6) for v in bbox]),
                    "query_text": f"{spec['display_name']} {matched_class}",
                    "keyword": spec["display_name"],
                    "candidate_cache_name": candidate_name,
                    "candidate_glb_path": glb_path,
                    "sample_type": sample_type,
                    "should_hit": "1" if sample_type in {"positive", "near_positive"} else "0",
                    "object_category": spec["display_name"],
                    "matched_dataset_class": matched_class,
                    "license_note": "public dataset sample; verify original dataset license before redistribution",
                    "split": "train_candidate",
                    "notes": "constructed from public detection bbox by class mapping",
                }
                sample_rows.append(row)
                counts_by_type[sample_type] += 1
                counts_by_candidate[candidate_name] += 1
                counts_by_source_dataset[source_dataset_used] += 1
                counts_by_source_image[source_image_id] += 1
                sample_num += 1
                break
            if sum(counts_by_type.values()) >= max_samples:
                break
        if sum(counts_by_type.values()) >= max_samples:
            break
        if all(counts_by_type[t] >= TARGET_COUNTS[t] for t in TARGET_COUNTS):
            break

    with METADATA_CSV.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(sample_rows)
    make_contact_sheet(sample_rows, CONTACT_SHEET)

    existing_labels = set()
    for sample in samples:
        for label, _bbox in detection_iter(sample):
            existing_labels.add(normalize_label(label))
    missing_classes = [cls for cls in requested_classes if normalize_label(cls) not in existing_labels]

    return {
        "fiftyone_available": True,
        "source_dataset_used": source_dataset_used,
        "total_samples": len(sample_rows),
        "count_by_sample_type": dict(counts_by_type),
        "count_by_candidate_cache_name": dict(counts_by_candidate),
        "count_by_source_dataset": dict(counts_by_source_dataset),
        "missing_classes": missing_classes,
        "missing_glb": missing_glb,
        "failed_download_reason": "; ".join(failed_reasons),
        "metadata_csv": str(METADATA_CSV),
        "contact_sheet": str(CONTACT_SHEET),
    }


def main() -> None:
    args = parse_args()
    root = Path(args.output_dir)
    global OUTPUT_DIR, ROI_DIR, METADATA_CSV, CONTACT_SHEET
    OUTPUT_DIR = root
    ROI_DIR = OUTPUT_DIR / "roi_images"
    METADATA_CSV = OUTPUT_DIR / "public_train_1000_metadata.csv"
    CONTACT_SHEET = OUTPUT_DIR / "public_train_1000_contact_sheet.jpg"
    ensure_dirs(root)

    payload: Dict[str, Any]
    try:
        import fiftyone  # noqa: F401

        fiftyone_available = True
    except Exception as exc:
        fiftyone_available = False
        failed_reason = f"FiftyOne not installed or unavailable: {type(exc).__name__}: {exc}"

    if args.dry_run:
        payload = {
            "fiftyone_available": fiftyone_available,
            "source_dataset_used": "none",
            "total_samples": 0,
            "count_by_sample_type": {},
            "count_by_candidate_cache_name": {},
            "count_by_source_dataset": {},
            "missing_classes": [],
            "missing_glb": [
                name
                for name, spec in CLASS_MAP.items()
                if not resolve_glb(spec["candidate_glb_paths"])[1]
            ],
            "failed_download_reason": "dry-run requested; no dataset download attempted",
            "metadata_csv": str(METADATA_CSV),
            "contact_sheet": str(CONTACT_SHEET),
            "optional_install_command": "pip install fiftyone",
        }
        write_empty_metadata(METADATA_CSV)
        make_placeholder_contact_sheet(CONTACT_SHEET, "Dry run only. No public dataset images downloaded.")
    elif not fiftyone_available:
        payload = {
            "fiftyone_available": False,
            "source_dataset_used": "none",
            "total_samples": 0,
            "count_by_sample_type": {},
            "count_by_candidate_cache_name": {},
            "count_by_source_dataset": {},
            "missing_classes": [],
            "missing_glb": [
                name
                for name, spec in CLASS_MAP.items()
                if not resolve_glb(spec["candidate_glb_paths"])[1]
            ],
            "failed_download_reason": failed_reason,
            "metadata_csv": str(METADATA_CSV),
            "contact_sheet": str(CONTACT_SHEET),
            "optional_install_command": "pip install fiftyone",
        }
        write_empty_metadata(METADATA_CSV)
        make_placeholder_contact_sheet(CONTACT_SHEET, "FiftyOne is not installed. No public dataset images downloaded.")
    else:
        try:
            payload = build_with_fiftyone(
                root=root,
                dataset_priorities=[x.strip() for x in args.dataset_priority.split(",") if x.strip()],
                max_samples=args.max_samples,
                seed=args.seed,
            )
        except Exception as exc:
            payload = {
                "fiftyone_available": True,
                "source_dataset_used": "none",
                "total_samples": 0,
                "count_by_sample_type": {},
                "count_by_candidate_cache_name": {},
                "count_by_source_dataset": {},
                "missing_classes": [],
                "missing_glb": [
                    name
                    for name, spec in CLASS_MAP.items()
                    if not resolve_glb(spec["candidate_glb_paths"])[1]
                ],
                "failed_download_reason": f"{type(exc).__name__}: {exc}",
                "metadata_csv": str(METADATA_CSV),
                "contact_sheet": str(CONTACT_SHEET),
                "optional_install_command": "pip install fiftyone",
            }
            write_empty_metadata(METADATA_CSV)
            make_placeholder_contact_sheet(CONTACT_SHEET, "Dataset download failed. See report JSON.")

    write_report(root, payload)
    print("=" * 72)
    print(f"prepare_script_path: {Path(__file__)}")
    print(f"public_train_total_samples: {payload.get('total_samples', 0)}")
    print(f"count_by_sample_type: {payload.get('count_by_sample_type', {})}")
    print(f"count_by_candidate_cache_name: {payload.get('count_by_candidate_cache_name', {})}")
    print(f"source_dataset_used: {payload.get('source_dataset_used', 'none')}")
    print(f"failed_download_reason: {payload.get('failed_download_reason', '')}")
    print(f"metadata_csv: {METADATA_CSV}")
    print(f"contact_sheet: {CONTACT_SHEET}")
    print("=" * 72)


if __name__ == "__main__":
    main()
