#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Compute real_challenge_50 similarity features with the real cache scorer.

This script reuses `cache_similarity.py` text/image scoring logic. It does not
call Qwen, TripoSR, Stable Fast 3D, or modify plus.py. If the challenge set is
incomplete or image scoring is unavailable, it writes blocked reports instead
of fabricating scores.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image, ImageDraw

from cache_similarity import build_similarity_index, load_similarity_index, safe_text, score_cache_entries


ROOT = Path("paper_repro_outputs/cache_similarity_eval_real_challenge_50")
METADATA_CSV = ROOT / "metadata" / "real_challenge_50_metadata.csv"
FEATURES_CSV = ROOT / "real_challenge_50_features.csv"
COLLECTION_CHECK_MD = ROOT / "real_challenge_50_collection_check.md"
FEATURE_CHECK_MD = ROOT / "real_challenge_50_feature_check.md"
CONTACT_SHEET = ROOT / "real_challenge_50_contact_sheet.jpg"
DEFAULT_CACHE_DIR = Path("paper_repro_outputs/cache_similarity_model_cache_eval_70")
DEFAULT_INDEX_PATH = DEFAULT_CACHE_DIR / "cache_similarity_index.json"
SAMPLE_TYPES = ["near_positive", "hard_negative"]
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
    "notes",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute real_challenge_50 features.")
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
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def image_exists(row: Dict[str, str]) -> bool:
    return bool(row.get("roi_image_path")) and Path(row["roi_image_path"]).exists()


def validate_collection(rows: List[Dict[str, str]]) -> Dict[str, Any]:
    counts = Counter(row.get("sample_type", "") for row in rows)
    missing_images = [row.get("sample_id", "") or row.get("roi_image_path", "") for row in rows if not image_exists(row)]
    empty_query = [row.get("sample_id", "") for row in rows if not (row.get("query_text") or "").strip()]
    empty_candidate = [row.get("sample_id", "") for row in rows if not (row.get("candidate_cache_name") or "").strip()]
    empty_glb = [row.get("sample_id", "") for row in rows if not (row.get("candidate_glb_path") or "").strip()]
    inconsistent = []
    for row in rows:
        st = row.get("sample_type", "")
        should = as_bool(row.get("should_hit"))
        if st == "near_positive" and not should:
            inconsistent.append(row.get("sample_id", ""))
        if st == "hard_negative" and should:
            inconsistent.append(row.get("sample_id", ""))
    return {
        "total": len(rows),
        "near_positive": counts.get("near_positive", 0),
        "hard_negative": counts.get("hard_negative", 0),
        "missing_images": missing_images,
        "empty_query_text": empty_query,
        "empty_candidate_cache_name": empty_candidate,
        "empty_candidate_glb_path": empty_glb,
        "inconsistent_should_hit": inconsistent,
    }


def write_collection_check(stats: Dict[str, Any]) -> None:
    text = f"""# real_challenge_50 采集状态检查报告

## 1. 当前数量

- total: {stats['total']}
- near_positive: {stats['near_positive']}
- hard_negative: {stats['hard_negative']}

## 2. 字段检查

- missing ROI image count: {len(stats['missing_images'])}
- empty query_text count: {len(stats['empty_query_text'])}
- empty candidate_cache_name count: {len(stats['empty_candidate_cache_name'])}
- empty candidate_glb_path count: {len(stats['empty_candidate_glb_path'])}
- inconsistent_should_hit_count: {len(stats['inconsistent_should_hit'])}

## 3. 结论

目标是 near_positive=25、hard_negative=25、total=50。只有采集和字段检查都通过后，才能计算真实相似度特征并进入 real_eval_120 训练。
"""
    COLLECTION_CHECK_MD.write_text(text, encoding="utf-8-sig")


def make_contact_sheet(rows: List[Dict[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    valid = [row for row in rows if image_exists(row)]
    if not valid:
        img = Image.new("RGB", (1000, 420), "white")
        draw = ImageDraw.Draw(img)
        draw.text((40, 180), "No real_challenge_50 ROI images available yet.", fill=(30, 30, 30))
        img.save(path, quality=92)
        return
    cells = []
    for row in valid[:80]:
        try:
            with Image.open(row["roi_image_path"]) as im:
                thumb = im.convert("RGB")
                thumb.thumbnail((150, 120))
                cells.append((thumb.copy(), row))
        except Exception:
            continue
    cols = 5
    cell_w, cell_h = 230, 170
    sheet = Image.new("RGB", (cols * cell_w, max(1, math.ceil(len(cells) / cols)) * cell_h), "white")
    draw = ImageDraw.Draw(sheet)
    for i, (thumb, row) in enumerate(cells):
        x = (i % cols) * cell_w
        y = (i // cols) * cell_h
        sheet.paste(thumb, (x + 8, y + 8))
        draw.text((x + 8, y + 132), f"{row.get('sample_id','')} {row.get('sample_type','')}"[:34], fill=(0, 0, 0))
        draw.text((x + 8, y + 150), row.get("candidate_cache_name", "")[:34], fill=(80, 80, 80))
    sheet.save(path, quality=92)


def load_entries(cache_dir: Path, index_path: Path):
    if index_path.exists():
        entries = load_similarity_index(index_path)
    else:
        entries = build_similarity_index(cache_dir, output_path=index_path)
    return entries


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
        if candidate_name and any(candidate_name in p or p in candidate_name for p in probes if p):
            return result
        if candidate_path and any(candidate_path in p or p in candidate_path for p in probes if p):
            return result
        if keyword and keyword in probes:
            return result
        if object_category and object_category in probes:
            return result
    return results[0] if results else None


def compute_features(rows: List[Dict[str, str]], cache_dir: Path, index_path: Path) -> Tuple[List[Dict[str, Any]], str, str, str]:
    entries = load_entries(cache_dir, index_path)
    if not entries:
        raise RuntimeError(f"No cache similarity entries available from {cache_dir}")
    if not any(entry.reference_image and Path(entry.reference_image).exists() for entry in entries):
        raise RuntimeError("No cache reference images available; cannot compute real image_score.")

    features: List[Dict[str, Any]] = []
    for row in rows:
        query_image = Path(row.get("roi_image_path", ""))
        if not query_image.exists():
            continue
        query_text = row.get("query_text") or row.get("keyword") or row.get("object_category") or ""
        results = score_cache_entries(
            entries,
            query_text=query_text,
            query_image=query_image,
            text_weight=0.5,
            image_weight=0.5,
            threshold=0.6,
        )
        result = find_candidate_result(results, row)
        if result is None or result.image_score is None:
            raise RuntimeError(f"Missing real image_score for sample {row.get('sample_id')}")
        text_score = float(result.text_score)
        image_score = float(result.image_score)
        fusion_score = round(0.5 * text_score + 0.5 * image_score, 4)
        features.append(
            {
                "sample_id": row.get("sample_id", ""),
                "roi_image_path": row.get("roi_image_path", ""),
                "query_text": query_text,
                "keyword": row.get("keyword", ""),
                "candidate_cache_name": row.get("candidate_cache_name", result.keyword),
                "candidate_glb_path": row.get("candidate_glb_path", result.model_path),
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
                "notes": row.get("notes", ""),
            }
        )
    return (
        features,
        "cache_similarity.text_similarity",
        "cache_similarity.image_similarity histogram+edge+aHash signature cosine",
        "False",
    )


def write_features(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FEATURE_FIELDS)
        writer.writeheader()
        writer.writerows([{k: row.get(k, "") for k in FEATURE_FIELDS} for row in rows])


def feature_check(rows: List[Dict[str, Any]], image_method: str, text_method: str, proxy_used: str) -> Dict[str, Any]:
    counts = Counter(row.get("sample_type", "") for row in rows)
    empty_text = [row.get("sample_id", "") for row in rows if row.get("text_score") in ("", None)]
    empty_image = [row.get("sample_id", "") for row in rows if row.get("image_score") in ("", None)]
    bad_fusion = []
    for row in rows:
        try:
            t = float(row["text_score"])
            i = float(row["image_score"])
            f = float(row["fusion_score"])
            if abs(f - (0.5 * t + 0.5 * i)) > 1e-3:
                bad_fusion.append(row.get("sample_id", ""))
        except Exception:
            bad_fusion.append(row.get("sample_id", ""))
    complete = (
        len(rows) == 50
        and counts.get("near_positive", 0) == 25
        and counts.get("hard_negative", 0) == 25
        and not empty_text
        and not empty_image
        and not bad_fusion
        and proxy_used == "False"
    )
    stats = {
        "feature_rows": len(rows),
        "near_positive": counts.get("near_positive", 0),
        "hard_negative": counts.get("hard_negative", 0),
        "empty_text_score_count": len(empty_text),
        "empty_image_score_count": len(empty_image),
        "bad_fusion_count": len(bad_fusion),
        "image_score_method": image_method,
        "text_score_method": text_method,
        "whether_proxy_score_used": proxy_used,
        "suggest_training": complete,
    }
    text = f"""# real_challenge_50 特征质量检查报告

## 1. 数量检查

- feature_rows: {stats['feature_rows']}
- near_positive: {stats['near_positive']} / 25
- hard_negative: {stats['hard_negative']} / 25

## 2. 分数检查

- empty_text_score_count: {stats['empty_text_score_count']}
- empty_image_score_count: {stats['empty_image_score_count']}
- bad_fusion_count: {stats['bad_fusion_count']}

## 3. 方法检查

- text_score_method: {text_method}
- image_score_method: {image_method}
- whether_proxy_score_used: {proxy_used}

## 4. 结论

- 是否建议进入 real_eval_120 训练: {complete}
"""
    FEATURE_CHECK_MD.write_text(text, encoding="utf-8-sig")
    return stats


def write_blocked_feature_check(reason: str, text_method: str = "", image_method: str = "", proxy_used: str = "False") -> Dict[str, Any]:
    stats = {
        "feature_rows": 0,
        "near_positive": 0,
        "hard_negative": 0,
        "image_score_method": image_method,
        "text_score_method": text_method,
        "whether_proxy_score_used": proxy_used,
        "suggest_training": False,
        "blocked_reason": reason,
    }
    FEATURE_CHECK_MD.write_text(
        f"""# real_challenge_50 特征质量检查报告

## 当前状态

特征未生成。

blocked_reason: {reason}

是否建议进入 real_eval_120 训练: False
""",
        encoding="utf-8-sig",
    )
    return stats


def maybe_train(complete: bool) -> bool:
    if not complete:
        return False
    completed = subprocess.run(
        [sys.executable, "train_real_eval_120_conservative_classifier.py"],
        check=False,
    )
    return completed.returncode == 0


def main() -> None:
    args = parse_args()
    metadata_path = Path(args.metadata_csv)
    output_csv = Path(args.output_csv)
    rows = read_rows(metadata_path)
    collection_stats = validate_collection(rows)
    write_collection_check(collection_stats)
    make_contact_sheet(rows, CONTACT_SHEET)

    features_generated = False
    train_executed = False
    feature_rows = 0
    text_method = ""
    image_method = ""
    proxy_used = "False"

    if collection_stats["total"] < 50:
        reason = f"collection incomplete: {collection_stats['total']} rows found, expected 50"
        feature_stats = write_blocked_feature_check(reason)
    elif (
        collection_stats["near_positive"] != 25
        or collection_stats["hard_negative"] != 25
        or collection_stats["missing_images"]
        or collection_stats["empty_query_text"]
        or collection_stats["empty_candidate_cache_name"]
        or collection_stats["empty_candidate_glb_path"]
        or collection_stats["inconsistent_should_hit"]
    ):
        reason = "collection field validation failed; see real_challenge_50_collection_check.md"
        feature_stats = write_blocked_feature_check(reason)
    else:
        try:
            features, text_method, image_method, proxy_used = compute_features(
                rows, Path(args.cache_dir), Path(args.index_path)
            )
            write_features(output_csv, features)
            features_generated = True
            feature_rows = len(features)
            feature_stats = feature_check(features, image_method, text_method, proxy_used)
            train_executed = maybe_train(bool(feature_stats["suggest_training"]) and args.train_if_complete)
        except Exception as exc:
            reason = f"{type(exc).__name__}: {exc}"
            feature_stats = write_blocked_feature_check(reason, text_method, image_method, proxy_used)

    print("=" * 72)
    print(f"current_real_challenge_count: {collection_stats['total']}")
    print(f"near_positive_count: {collection_stats['near_positive']}")
    print(f"hard_negative_count: {collection_stats['hard_negative']}")
    print(f"real_challenge_50_features.csv generated: {features_generated}")
    print(f"feature_rows: {feature_rows}")
    print(f"image_score_method: {feature_stats.get('image_score_method', image_method)}")
    print(f"text_score_method: {feature_stats.get('text_score_method', text_method)}")
    print(f"whether_proxy_score_used: {feature_stats.get('whether_proxy_score_used', proxy_used)}")
    print(f"whether_training_executed: {train_executed}")
    print("real_eval_120_conservative_classifier_report.md: "
          "paper_repro_outputs/cache_similarity_eval_real_eval_120/conservative_classifier/"
          "real_eval_120_conservative_classifier_report.md")
    print("=" * 72)


if __name__ == "__main__":
    main()
