#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Compute real_add_130 features using the real cache similarity scorer.

The script blocks unless metadata and ROI images are complete. It reuses
cache_similarity.text_similarity and cache_similarity.image_similarity through
score_cache_entries. It never fabricates proxy image scores from sample_type.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

from PIL import Image, ImageDraw

from cache_similarity import build_similarity_index, load_similarity_index, safe_text, score_cache_entries


ROOT = Path("paper_repro_outputs/cache_similarity_eval_real_add_130")
METADATA_CSV = ROOT / "metadata" / "real_add_130_metadata.csv"
FEATURES_CSV = ROOT / "real_add_130_features.csv"
REPORTS_DIR = ROOT / "reports"
COLLECTION_CHECK_MD = REPORTS_DIR / "real_add_130_collection_check.md"
FEATURE_CHECK_MD = REPORTS_DIR / "real_add_130_feature_check.md"
CONTACT_SHEET = ROOT / "real_add_130_contact_sheet.jpg"
DEFAULT_CACHE_DIR = Path("paper_repro_outputs/cache_similarity_model_cache_eval_70")
DEFAULT_INDEX_PATH = DEFAULT_CACHE_DIR / "cache_similarity_index.json"

SAMPLE_TYPES = ["positive", "near_positive", "hard_negative", "negative"]
TARGET_BY_TYPE = {
    "positive": 28,
    "near_positive": 36,
    "hard_negative": 36,
    "negative": 30,
}
TARGET_TOTAL = 130
FEATURE_FIELDS = [
    "sample_id",
    "roi_image_path",
    "query_text",
    "keyword",
    "candidate_cache_name",
    "candidate_glb_path",
    "sample_type",
    "should_hit",
    "text_score",
    "image_score",
    "fusion_score",
    "score_abs_diff",
    "score_max",
    "score_min",
    "object_category",
    "source_video",
    "frame_index",
    "timestamp_sec",
    "viewpoint",
    "distance",
    "lighting",
    "background",
    "occlusion",
    "notes",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute real_add_130 real similarity features.")
    parser.add_argument("--metadata-csv", default=str(METADATA_CSV))
    parser.add_argument("--output-csv", default=str(FEATURES_CSV))
    parser.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR))
    parser.add_argument("--index-path", default=str(DEFAULT_INDEX_PATH))
    parser.add_argument("--train-if-complete", action="store_true")
    return parser.parse_args()


def read_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return [row for row in csv.DictReader(f) if any((v or "").strip() for v in row.values())]


def as_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y", "hit"}


def expected_should_hit(sample_type: str) -> bool:
    return sample_type in {"positive", "near_positive"}


def image_exists(row: Dict[str, str]) -> bool:
    path = (row.get("roi_image_path") or "").strip()
    return bool(path) and Path(path).exists()


def validate_collection(rows: List[Dict[str, str]]) -> Dict[str, Any]:
    counts = Counter(row.get("sample_type", "") for row in rows)
    by_object_type = defaultdict(Counter)
    missing_images = []
    missing_glb_paths = []
    empty_fields = defaultdict(list)
    invalid_labels = []
    required = [
        "sample_id",
        "roi_image_path",
        "query_text",
        "keyword",
        "candidate_cache_name",
        "candidate_glb_path",
        "sample_type",
        "should_hit",
        "object_category",
    ]
    for row in rows:
        sample_id = row.get("sample_id", "") or row.get("roi_image_path", "")
        sample_type = row.get("sample_type", "")
        by_object_type[row.get("object_category", "")][sample_type] += 1
        if not image_exists(row):
            missing_images.append(sample_id)
        glb_path = (row.get("candidate_glb_path") or "").strip()
        if not glb_path or not Path(glb_path).exists():
            missing_glb_paths.append(sample_id)
        for field in required:
            if not (row.get(field) or "").strip():
                empty_fields[field].append(sample_id)
        if sample_type in SAMPLE_TYPES and as_bool(row.get("should_hit")) != expected_should_hit(sample_type):
            invalid_labels.append(sample_id)

    complete = (
        len(rows) == TARGET_TOTAL
        and all(counts.get(sample_type, 0) == TARGET_BY_TYPE[sample_type] for sample_type in SAMPLE_TYPES)
        and not missing_images
        and not missing_glb_paths
        and not invalid_labels
        and not any(empty_fields.values())
    )
    return {
        "total_samples": len(rows),
        "count_by_sample_type": {sample_type: counts.get(sample_type, 0) for sample_type in SAMPLE_TYPES},
        "count_by_object_and_type": {
            obj: {sample_type: by_object_type[obj].get(sample_type, 0) for sample_type in SAMPLE_TYPES}
            for obj in sorted(by_object_type)
        },
        "missing_image_count": len(missing_images),
        "missing_images": missing_images[:50],
        "missing_glb_path_count": len(missing_glb_paths),
        "missing_glb_paths": missing_glb_paths[:50],
        "invalid_label_count": len(invalid_labels),
        "invalid_labels": invalid_labels[:50],
        "empty_field_counts": {field: len(values) for field, values in empty_fields.items()},
        "complete": complete,
    }


def write_collection_check(stats: Dict[str, Any]) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    rows = ["| sample_type | current | target |", "| --- | ---: | ---: |"]
    counts = stats["count_by_sample_type"]
    for sample_type in SAMPLE_TYPES:
        rows.append(f"| {sample_type} | {counts.get(sample_type, 0)} | {TARGET_BY_TYPE[sample_type]} |")

    empty_rows = ["| field | empty_count |", "| --- | ---: |"]
    for field, count in stats["empty_field_counts"].items():
        empty_rows.append(f"| {field} | {count} |")

    text = f"""# real_add_130 采集状态检查报告

## 1. 当前数量

- total_samples: {stats['total_samples']} / {TARGET_TOTAL}
- missing_image_count: {stats['missing_image_count']}
- missing_glb_path_count: {stats['missing_glb_path_count']}
- invalid_label_count: {stats['invalid_label_count']}
- complete: {stats['complete']}

## 2. 按样本类型统计

{chr(10).join(rows)}

## 3. 空字段检查

{chr(10).join(empty_rows)}

## 4. 结论

只有 total=130、四类数量达标、ROI 图片存在、candidate_glb_path 存在、标签规则一致时，才会计算真实 text_score / image_score。
"""
    COLLECTION_CHECK_MD.write_text(text, encoding="utf-8-sig")


def make_contact_sheet(rows: List[Dict[str, str]], output_path: Path) -> None:
    valid = [row for row in rows if image_exists(row)]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not valid:
        img = Image.new("RGB", (1000, 420), "white")
        draw = ImageDraw.Draw(img)
        draw.text((40, 180), "No real_add_130 ROI images available yet.", fill=(30, 30, 30))
        img.save(output_path, quality=92)
        return

    cells = []
    for row in valid[:160]:
        try:
            with Image.open(row["roi_image_path"]) as im:
                thumb = im.convert("RGB")
                thumb.thumbnail((150, 115))
                cells.append((thumb.copy(), row))
        except Exception:
            continue
    cols = 5
    cell_w, cell_h = 245, 172
    sheet = Image.new("RGB", (cols * cell_w, max(1, math.ceil(len(cells) / cols)) * cell_h), "white")
    draw = ImageDraw.Draw(sheet)
    for idx, (thumb, row) in enumerate(cells):
        x = (idx % cols) * cell_w
        y = (idx // cols) * cell_h
        sheet.paste(thumb, (x + 8, y + 8))
        draw.text((x + 8, y + 127), f"{row.get('sample_id','')} {row.get('sample_type','')}"[:36], fill=(0, 0, 0))
        draw.text((x + 8, y + 146), f"{row.get('object_category','')} -> {row.get('candidate_cache_name','')}"[:36], fill=(80, 80, 80))
    sheet.save(output_path, quality=92)


def load_entries(cache_dir: Path, index_path: Path):
    if index_path.exists():
        return load_similarity_index(index_path)
    return build_similarity_index(cache_dir, output_path=index_path)


def find_candidate_result(results, row: Dict[str, str]):  # noqa: ANN001
    candidate_name = safe_text(row.get("candidate_cache_name", ""))
    candidate_path = safe_text(Path(row.get("candidate_glb_path", "")).stem)
    keyword = safe_text(row.get("keyword", ""))
    object_category = safe_text(row.get("object_category", ""))
    for result in results:
        probes = {
            safe_text(result.cache_id),
            safe_text(result.keyword),
            safe_text(result.filename),
            safe_text(Path(result.model_path).stem),
        }
        if candidate_name and any(candidate_name in probe or probe in candidate_name for probe in probes if probe):
            return result
        if candidate_path and any(candidate_path in probe or probe in candidate_path for probe in probes if probe):
            return result
        if keyword and any(keyword in probe or probe in keyword for probe in probes if probe):
            return result
        if object_category and any(object_category in probe or probe in object_category for probe in probes if probe):
            return result
    return None


def compute_features(rows: List[Dict[str, str]], cache_dir: Path, index_path: Path) -> Tuple[List[Dict[str, Any]], str, str]:
    entries = load_entries(cache_dir, index_path)
    if not entries:
        raise RuntimeError(f"No cache entries available from {cache_dir}")
    if not any(entry.reference_image and Path(entry.reference_image).exists() for entry in entries):
        raise RuntimeError("No cache reference images available; cannot compute real image_score.")

    out: List[Dict[str, Any]] = []
    for row in rows:
        query_image = Path(row.get("roi_image_path", ""))
        if not query_image.exists():
            raise FileNotFoundError(f"Missing ROI image: {query_image}")
        query_text = row.get("query_text") or row.get("keyword") or row.get("object_category") or ""
        results = score_cache_entries(
            entries,
            query_text=query_text,
            query_image=query_image,
            text_weight=0.5,
            image_weight=0.5,
            threshold=0.6,
        )
        candidate = find_candidate_result(results, row)
        if candidate is None or candidate.image_score is None:
            raise RuntimeError(f"Missing real image_score for sample {row.get('sample_id')}")
        text_score = float(candidate.text_score)
        image_score = float(candidate.image_score)
        fusion_score = round(0.5 * text_score + 0.5 * image_score, 4)
        out.append(
            {
                "sample_id": row.get("sample_id", ""),
                "roi_image_path": row.get("roi_image_path", ""),
                "query_text": query_text,
                "keyword": row.get("keyword", ""),
                "candidate_cache_name": row.get("candidate_cache_name", candidate.keyword),
                "candidate_glb_path": row.get("candidate_glb_path", candidate.model_path),
                "sample_type": row.get("sample_type", ""),
                "should_hit": "1" if as_bool(row.get("should_hit")) else "0",
                "text_score": round(text_score, 4),
                "image_score": round(image_score, 4),
                "fusion_score": fusion_score,
                "score_abs_diff": round(abs(text_score - image_score), 4),
                "score_max": round(max(text_score, image_score), 4),
                "score_min": round(min(text_score, image_score), 4),
                "object_category": row.get("object_category", ""),
                "source_video": row.get("source_video", ""),
                "frame_index": row.get("frame_index", ""),
                "timestamp_sec": row.get("timestamp_sec", ""),
                "viewpoint": row.get("viewpoint", ""),
                "distance": row.get("distance", ""),
                "lighting": row.get("lighting", ""),
                "background": row.get("background", ""),
                "occlusion": row.get("occlusion", ""),
                "notes": row.get("notes", ""),
            }
        )
    return (
        out,
        "cache_similarity.text_similarity",
        "cache_similarity.image_similarity histogram+edge+aHash signature cosine",
    )


def write_features(path: Path, rows: List[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FEATURE_FIELDS)
        writer.writeheader()
        writer.writerows([{field: row.get(field, "") for field in FEATURE_FIELDS} for row in rows])


def feature_check(rows: List[Dict[str, Any]], text_method: str, image_method: str) -> Dict[str, Any]:
    counts = Counter(row.get("sample_type", "") for row in rows)
    empty_text = [row.get("sample_id", "") for row in rows if row.get("text_score") in ("", None)]
    empty_image = [row.get("sample_id", "") for row in rows if row.get("image_score") in ("", None)]
    bad_fusion = []
    for row in rows:
        try:
            text_score = float(row["text_score"])
            image_score = float(row["image_score"])
            fusion_score = float(row["fusion_score"])
            if abs(fusion_score - (0.5 * text_score + 0.5 * image_score)) > 1e-3:
                bad_fusion.append(row.get("sample_id", ""))
        except Exception:
            bad_fusion.append(row.get("sample_id", ""))

    complete = (
        len(rows) == TARGET_TOTAL
        and all(counts.get(sample_type, 0) == TARGET_BY_TYPE[sample_type] for sample_type in SAMPLE_TYPES)
        and not empty_text
        and not empty_image
        and not bad_fusion
    )
    stats = {
        "feature_rows": len(rows),
        "count_by_sample_type": {sample_type: counts.get(sample_type, 0) for sample_type in SAMPLE_TYPES},
        "empty_text_score_count": len(empty_text),
        "empty_image_score_count": len(empty_image),
        "bad_fusion_count": len(bad_fusion),
        "text_score_method": text_method,
        "image_score_method": image_method,
        "whether_proxy_score_used": False,
        "suggest_training": complete,
    }

    table = ["| sample_type | current | target |", "| --- | ---: | ---: |"]
    for sample_type in SAMPLE_TYPES:
        table.append(f"| {sample_type} | {counts.get(sample_type, 0)} | {TARGET_BY_TYPE[sample_type]} |")

    text = f"""# real_add_130 特征质量检查报告

## 1. 数量检查

- feature_rows: {len(rows)} / {TARGET_TOTAL}

{chr(10).join(table)}

## 2. 分数检查

- empty_text_score_count: {len(empty_text)}
- empty_image_score_count: {len(empty_image)}
- bad_fusion_count: {len(bad_fusion)}

## 3. 方法检查

- text_score_method: {text_method}
- image_score_method: {image_method}
- whether_proxy_score_used: False

## 4. 结论

- 是否建议进入 real_eval_200 训练: {complete}
"""
    FEATURE_CHECK_MD.write_text(text, encoding="utf-8-sig")
    return stats


def write_blocked_feature_check(reason: str) -> Dict[str, Any]:
    stats = {
        "feature_rows": 0,
        "count_by_sample_type": {sample_type: 0 for sample_type in SAMPLE_TYPES},
        "text_score_method": "",
        "image_score_method": "",
        "whether_proxy_score_used": False,
        "suggest_training": False,
        "blocked_reason": reason,
    }
    FEATURE_CHECK_MD.write_text(
        f"""# real_add_130 特征质量检查报告

## 当前状态

特征未生成。

blocked_reason: {reason}

是否建议进入 real_eval_200 训练: False
""",
        encoding="utf-8-sig",
    )
    return stats


def maybe_train(complete: bool, train_if_complete: bool) -> bool:
    if not complete or not train_if_complete:
        return False
    completed = subprocess.run([sys.executable, "train_real_eval_200_conservative_classifier.py"], check=False)
    return completed.returncode == 0


def main() -> None:
    args = parse_args()
    metadata_path = Path(args.metadata_csv)
    output_csv = Path(args.output_csv)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    rows = read_rows(metadata_path)
    collection_stats = validate_collection(rows)
    write_collection_check(collection_stats)
    make_contact_sheet(rows, CONTACT_SHEET)

    features_generated = False
    training_executed = False
    feature_rows = 0
    text_method = ""
    image_method = ""

    if not collection_stats["complete"]:
        reason = (
            "collection incomplete or invalid: "
            f"total={collection_stats['total_samples']}, "
            f"counts={collection_stats['count_by_sample_type']}, "
            f"missing_images={collection_stats['missing_image_count']}, "
            f"missing_glb_paths={collection_stats['missing_glb_path_count']}, "
            f"invalid_labels={collection_stats['invalid_label_count']}"
        )
        feature_stats = write_blocked_feature_check(reason)
    else:
        try:
            features, text_method, image_method = compute_features(
                rows, Path(args.cache_dir), Path(args.index_path)
            )
            write_features(output_csv, features)
            features_generated = True
            feature_rows = len(features)
            feature_stats = feature_check(features, text_method, image_method)
            training_executed = maybe_train(bool(feature_stats["suggest_training"]), args.train_if_complete)
        except Exception as exc:
            feature_stats = write_blocked_feature_check(f"{type(exc).__name__}: {exc}")

    print("=" * 72)
    print(f"real_add_130_total_samples: {collection_stats['total_samples']}")
    print(f"count_by_sample_type: {collection_stats['count_by_sample_type']}")
    print(f"real_add_130_features.csv generated: {features_generated}")
    print(f"feature_rows: {feature_rows}")
    print(f"image_score_method: {feature_stats.get('image_score_method', image_method)}")
    print(f"text_score_method: {feature_stats.get('text_score_method', text_method)}")
    print(f"whether_proxy_score_used: {feature_stats.get('whether_proxy_score_used', False)}")
    print(f"training_executed: {training_executed}")
    print("real_eval_200_conservative_classifier_report.md: "
          "paper_repro_outputs/cache_similarity_eval_real_eval_200/conservative_classifier/"
          "real_eval_200_conservative_classifier_report.md")
    print("=" * 72)


if __name__ == "__main__":
    main()
