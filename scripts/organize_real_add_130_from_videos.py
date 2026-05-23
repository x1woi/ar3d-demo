#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Organize real_add_130 ROI samples from named raw videos.

The script parses filenames such as `keyboard_near_positive.mp4`, extracts
frames every 2 seconds for contact-sheet review, and saves a target number of
selected frames as ROI images. It fills metadata from filename conventions only;
it does not call Qwen, TripoSR, Stable Fast 3D, or compute/fabricate similarity
scores.
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
from PIL import Image, ImageDraw


ROOT = Path("paper_repro_outputs/cache_similarity_eval_real_add_130")
RAW_VIDEOS_DIR = ROOT / "raw_videos"
FRAME_CANDIDATE_DIR = ROOT / "frame_candidates"
CONTACT_SHEET_DIR = ROOT / "contact_sheets"
ROI_DIR = ROOT / "roi_images"
METADATA_CSV = ROOT / "metadata" / "real_add_130_metadata.csv"
REPORTS_DIR = ROOT / "reports"
ORGANIZE_REPORT_MD = REPORTS_DIR / "real_add_130_video_organize_report.md"
ORGANIZE_REPORT_JSON = REPORTS_DIR / "real_add_130_video_organize_report.json"

SAMPLE_TYPES = ["positive", "near_positive", "hard_negative", "negative"]
TARGET_BY_TYPE = {"positive": 28, "near_positive": 36, "hard_negative": 36, "negative": 30}
TARGET_BY_OBJECT_AND_TYPE = {
    "keyboard": {"positive": 7, "near_positive": 9, "hard_negative": 9, "negative": 8},
    "cup": {"positive": 7, "near_positive": 9, "hard_negative": 9, "negative": 8},
    "tissue_box": {"positive": 7, "near_positive": 9, "hard_negative": 9, "negative": 7},
    "glasses": {"positive": 7, "near_positive": 9, "hard_negative": 9, "negative": 7},
}

CACHE_INFO = {
    "keyboard": {
        "keyword": "电脑键盘",
        "candidate_cache_name": "keyboard",
        "object_category": "键盘",
        "candidate_glb_path": "runtime_assets/competition_demo_models/keyboard.glb",
        "positive_query": "电脑键盘 用来输入文字的设备",
        "near_query": "相似键盘或按键区域，可以考虑复用键盘模型",
        "negative_query": "无关物体或相似干扰区域，不应复用键盘模型",
    },
    "cup": {
        "keyword": "杯子",
        "candidate_cache_name": "cup",
        "object_category": "杯子",
        "candidate_glb_path": "runtime_assets/competition_demo_models/cup.glb",
        "positive_query": "杯子 用来盛放液体的容器",
        "near_query": "相似杯子或容器，可以考虑复用杯子模型",
        "negative_query": "无关物体或相似干扰区域，不应复用杯子模型",
    },
    "tissue_box": {
        "keyword": "纸巾盒",
        "candidate_cache_name": "tissue_box",
        "object_category": "纸巾盒",
        "candidate_glb_path": "runtime_assets/competition_demo_models/tissue_box.glb",
        "positive_query": "纸巾盒 用来放纸巾的盒子",
        "near_query": "相似盒子或包装盒，可以考虑复用纸巾盒模型",
        "negative_query": "无关物体或相似干扰区域，不应复用纸巾盒模型",
    },
    "glasses": {
        "keyword": "眼镜",
        "candidate_cache_name": "glasses",
        "object_category": "眼镜",
        "candidate_glb_path": "runtime_assets/competition_demo_models/glasses.glb",
        "positive_query": "眼镜 帮助人看清东西的工具",
        "near_query": "相似眼镜图像，可以复用眼镜模型",
        "negative_query": "无关物体或相似干扰区域，不应复用眼镜模型",
    },
}

METADATA_FIELDS = [
    "sample_id",
    "roi_image_path",
    "query_text",
    "keyword",
    "candidate_cache_name",
    "candidate_glb_path",
    "sample_type",
    "should_hit",
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


@dataclass
class CandidateFrame:
    video_path: Path
    object_key: str
    sample_type: str
    frame_index: int
    timestamp_sec: float
    frame_path: Path


def parse_video_name(path: Path) -> Tuple[str, str] | None:
    stem = path.stem.strip(".").lower()
    for object_key in sorted(TARGET_BY_OBJECT_AND_TYPE, key=len, reverse=True):
        prefix = f"{object_key}_"
        if not stem.startswith(prefix):
            continue
        rest = stem[len(prefix) :].strip(".")
        if rest in SAMPLE_TYPES:
            return object_key, rest
    return None


def ensure_dirs() -> None:
    for sample_type in SAMPLE_TYPES:
        (ROI_DIR / sample_type).mkdir(parents=True, exist_ok=True)
    FRAME_CANDIDATE_DIR.mkdir(parents=True, exist_ok=True)
    CONTACT_SHEET_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    METADATA_CSV.parent.mkdir(parents=True, exist_ok=True)


def extract_every_2s(video_path: Path, object_key: str, sample_type: str) -> List[CandidateFrame]:
    out_dir = FRAME_CANDIDATE_DIR / object_key / sample_type
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = frame_count / fps if fps else 0.0
    candidates: List[CandidateFrame] = []
    if fps <= 0 or frame_count <= 0:
        cap.release()
        return candidates

    timestamp = 0.0
    while timestamp <= duration + 1e-6:
        frame_index = min(frame_count - 1, int(round(timestamp * fps)))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = cap.read()
        if ok and frame is not None:
            out_path = out_dir / f"{video_path.stem.strip('.')}_t{timestamp:05.2f}_f{frame_index:06d}.jpg"
            cv2.imencode(".jpg", frame)[1].tofile(str(out_path))
            candidates.append(
                CandidateFrame(
                    video_path=video_path,
                    object_key=object_key,
                    sample_type=sample_type,
                    frame_index=frame_index,
                    timestamp_sec=round(timestamp, 3),
                    frame_path=out_path,
                )
            )
        timestamp += 2.0
    cap.release()
    return candidates


def make_contact_sheet(frames: List[CandidateFrame], output_path: Path, title: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not frames:
        img = Image.new("RGB", (900, 320), "white")
        draw = ImageDraw.Draw(img)
        draw.text((30, 140), f"No frames: {title}", fill=(0, 0, 0))
        img.save(output_path, quality=92)
        return
    cells = []
    for idx, item in enumerate(frames):
        with Image.open(item.frame_path) as im:
            thumb = im.convert("RGB")
            thumb.thumbnail((150, 105))
            cells.append((idx + 1, thumb.copy(), item))
    cols = 5
    cell_w, cell_h = 230, 155
    sheet = Image.new("RGB", (cols * cell_w, math.ceil(len(cells) / cols) * cell_h + 40), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((12, 12), title, fill=(0, 0, 0))
    for idx, thumb, item in cells:
        x = ((idx - 1) % cols) * cell_w
        y = ((idx - 1) // cols) * cell_h + 40
        sheet.paste(thumb, (x + 8, y + 8))
        draw.text((x + 8, y + 116), f"{idx}  t={item.timestamp_sec:.1f}s  f={item.frame_index}", fill=(0, 0, 0))
    sheet.save(output_path, quality=92)


def select_frames(frames: List[CandidateFrame], target: int) -> List[CandidateFrame]:
    if len(frames) <= target:
        return frames
    positions = [round(i * (len(frames) - 1) / (target - 1)) for i in range(target)]
    selected = []
    seen = set()
    for pos in positions:
        if pos not in seen:
            selected.append(frames[pos])
            seen.add(pos)
    idx = 0
    while len(selected) < target and idx < len(frames):
        if idx not in seen:
            selected.append(frames[idx])
            seen.add(idx)
        idx += 1
    selected.sort(key=lambda item: item.timestamp_sec)
    return selected[:target]


def query_for(object_key: str, sample_type: str) -> str:
    info = CACHE_INFO[object_key]
    if sample_type == "positive":
        return info["positive_query"]
    if sample_type == "near_positive":
        return info["near_query"]
    return info["negative_query"]


def should_hit(sample_type: str) -> str:
    return "1" if sample_type in {"positive", "near_positive"} else "0"


def copy_selected_to_roi(selected: Dict[Tuple[str, str], List[CandidateFrame]]) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    sample_index = 1
    for object_key in TARGET_BY_OBJECT_AND_TYPE:
        for sample_type in SAMPLE_TYPES:
            for item in selected.get((object_key, sample_type), []):
                info = CACHE_INFO[object_key]
                sample_id = f"real_add_{sample_index:04d}_{object_key}_{sample_type}"
                roi_path = ROI_DIR / sample_type / f"{sample_id}.jpg"
                with Image.open(item.frame_path) as im:
                    im.convert("RGB").save(roi_path, quality=94)
                rows.append(
                    {
                        "sample_id": sample_id,
                        "roi_image_path": str(roi_path),
                        "query_text": query_for(object_key, sample_type),
                        "keyword": info["keyword"],
                        "candidate_cache_name": info["candidate_cache_name"],
                        "candidate_glb_path": info["candidate_glb_path"],
                        "sample_type": sample_type,
                        "should_hit": should_hit(sample_type),
                        "object_category": info["object_category"],
                        "source_video": str(item.video_path),
                        "frame_index": str(item.frame_index),
                        "timestamp_sec": f"{item.timestamp_sec:.3f}",
                        "viewpoint": "video_sample",
                        "distance": "video_sample",
                        "lighting": "video_sample",
                        "background": "video_sample",
                        "occlusion": "video_sample",
                        "notes": "按视频文件名自动初筛；ROI 为每隔 2 秒抽帧中的均匀选择帧，需人工复核 contact sheet。",
                    }
                )
                sample_index += 1
    return rows


def write_metadata(rows: List[Dict[str, str]]) -> None:
    with METADATA_CSV.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=METADATA_FIELDS)
        writer.writeheader()
        writer.writerows([{field: row.get(field, "") for field in METADATA_FIELDS} for row in rows])


def validate_rows(rows: List[Dict[str, str]]) -> Dict[str, object]:
    counts = {sample_type: 0 for sample_type in SAMPLE_TYPES}
    for row in rows:
        counts[row["sample_type"]] = counts.get(row["sample_type"], 0) + 1
    sample_ids = [row["sample_id"] for row in rows]
    missing_images = [row["sample_id"] for row in rows if not Path(row["roi_image_path"]).exists()]
    missing_glb = [row["sample_id"] for row in rows if not Path(row["candidate_glb_path"]).exists()]
    invalid_label = [
        row["sample_id"]
        for row in rows
        if (row["sample_type"] in {"positive", "near_positive"} and row["should_hit"] != "1")
        or (row["sample_type"] in {"hard_negative", "negative"} and row["should_hit"] != "0")
    ]
    empty_query = [row["sample_id"] for row in rows if not row.get("query_text")]
    empty_keyword = [row["sample_id"] for row in rows if not row.get("keyword")]
    return {
        "total_samples": len(rows),
        "count_by_sample_type": counts,
        "missing_image_count": len(missing_images),
        "missing_glb_path_count": len(missing_glb),
        "missing_glb_samples": missing_glb[:20],
        "invalid_label_count": len(invalid_label),
        "duplicate_sample_id_count": len(sample_ids) - len(set(sample_ids)),
        "empty_query_text_count": len(empty_query),
        "empty_keyword_count": len(empty_keyword),
        "complete_for_feature_compute": (
            len(rows) == 130
            and counts == TARGET_BY_TYPE
            and not missing_images
            and not missing_glb
            and not invalid_label
            and len(sample_ids) == len(set(sample_ids))
            and not empty_query
            and not empty_keyword
        ),
    }


def write_report(
    actual_files: List[str],
    unmatched_files: List[str],
    all_candidates: Dict[Tuple[str, str], List[CandidateFrame]],
    selected: Dict[Tuple[str, str], List[CandidateFrame]],
    validation: Dict[str, object],
) -> None:
    lines = ["| object | sample_type | extracted_frames | selected_roi | target |", "| --- | --- | ---: | ---: | ---: |"]
    for object_key in TARGET_BY_OBJECT_AND_TYPE:
        for sample_type in SAMPLE_TYPES:
            lines.append(
                f"| {object_key} | {sample_type} | {len(all_candidates.get((object_key, sample_type), []))} | "
                f"{len(selected.get((object_key, sample_type), []))} | {TARGET_BY_OBJECT_AND_TYPE[object_key][sample_type]} |"
            )
    text = f"""# real_add_130 视频整理报告

## 1. 输入视频

{chr(10).join(f'- {name}' for name in actual_files)}

## 2. 未识别文件名

{chr(10).join(f'- {name}' for name in unmatched_files) if unmatched_files else '无'}

## 3. 抽帧与选择统计

{chr(10).join(lines)}

## 4. metadata 检查

- total_samples: {validation['total_samples']}
- count_by_sample_type: {validation['count_by_sample_type']}
- missing_image_count: {validation['missing_image_count']}
- missing_glb_path_count: {validation['missing_glb_path_count']}
- invalid_label_count: {validation['invalid_label_count']}
- duplicate_sample_id_count: {validation['duplicate_sample_id_count']}
- empty_query_text_count: {validation['empty_query_text_count']}
- empty_keyword_count: {validation['empty_keyword_count']}
- complete_for_feature_compute: {validation['complete_for_feature_compute']}

## 5. 重要说明

本轮仅按视频文件名进行样本类型和候选缓存模型初筛，不计算、不伪造 text_score / image_score。

如果 `missing_glb_path_count > 0`，说明有候选缓存模型文件缺失。当前 `cup` 在本地没有对应 GLB，因此会阻塞真实特征计算和训练，直到补充 `runtime_assets/competition_demo_models/cup.glb` 或调整候选缓存模型。
"""
    ORGANIZE_REPORT_MD.write_text(text, encoding="utf-8-sig")
    ORGANIZE_REPORT_JSON.write_text(
        json.dumps(
            {
                "actual_files": actual_files,
                "unmatched_files": unmatched_files,
                "validation": validation,
                "selected_counts": {
                    f"{object_key}_{sample_type}": len(selected.get((object_key, sample_type), []))
                    for object_key in TARGET_BY_OBJECT_AND_TYPE
                    for sample_type in SAMPLE_TYPES
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def main() -> None:
    ensure_dirs()
    videos = sorted(RAW_VIDEOS_DIR.glob("*.mp4"))
    actual_files = [path.name for path in videos]
    unmatched_files: List[str] = []
    all_candidates: Dict[Tuple[str, str], List[CandidateFrame]] = {}
    selected: Dict[Tuple[str, str], List[CandidateFrame]] = {}

    for video_path in videos:
        parsed = parse_video_name(video_path)
        if parsed is None:
            unmatched_files.append(video_path.name)
            continue
        object_key, sample_type = parsed
        frames = extract_every_2s(video_path, object_key, sample_type)
        all_candidates[(object_key, sample_type)] = frames
        make_contact_sheet(
            frames,
            CONTACT_SHEET_DIR / f"contact_sheet_{object_key}_{sample_type}.jpg",
            f"{object_key} {sample_type} ({video_path.name})",
        )
        target = TARGET_BY_OBJECT_AND_TYPE[object_key][sample_type]
        selected[(object_key, sample_type)] = select_frames(frames, target)

    selected_rows = copy_selected_to_roi(selected)
    write_metadata(selected_rows)
    validation = validate_rows(selected_rows)
    make_contact_sheet(
        [
            CandidateFrame(
                video_path=Path(row["source_video"]),
                object_key="",
                sample_type=row["sample_type"],
                frame_index=int(row["frame_index"]),
                timestamp_sec=float(row["timestamp_sec"]),
                frame_path=Path(row["roi_image_path"]),
            )
            for row in selected_rows
        ],
        CONTACT_SHEET_DIR / "contact_sheet_selected_roi_all.jpg",
        "Selected real_add_130 ROI samples",
    )
    write_report(actual_files, unmatched_files, all_candidates, selected, validation)

    print("=" * 72)
    print(f"total_samples: {validation['total_samples']}")
    print(f"count_by_sample_type: {validation['count_by_sample_type']}")
    print(f"missing_image_count: {validation['missing_image_count']}")
    print(f"missing_glb_path_count: {validation['missing_glb_path_count']}")
    print(f"invalid_label_count: {validation['invalid_label_count']}")
    print(f"metadata_csv: {METADATA_CSV}")
    print(f"selected_contact_sheet: {CONTACT_SHEET_DIR / 'contact_sheet_selected_roi_all.jpg'}")
    print(f"organize_report: {ORGANIZE_REPORT_MD}")
    print("=" * 72)


if __name__ == "__main__":
    main()
