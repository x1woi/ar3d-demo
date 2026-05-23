#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Interactive ROI collector for the v4_real_200 cache similarity dataset.

This tool only captures ROI images and metadata. It does not call Qwen,
TripoSR, Stable Fast 3D, or modify plus.py.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


SAMPLE_TYPES = ["positive", "near_positive", "hard_negative", "negative"]
TARGET_COUNTS = {
    "positive": 40,
    "near_positive": 60,
    "hard_negative": 60,
    "negative": 40,
}
CSV_FIELDS = [
    "sample_id",
    "roi_image_path",
    "query_text",
    "keyword",
    "candidate_cache_name",
    "candidate_glb_path",
    "sample_type",
    "should_hit",
    "object_category",
    "viewpoint",
    "distance",
    "lighting",
    "background",
    "occlusion",
    "source_video",
    "frame_index",
    "notes",
]
DEFAULT_ROOT = Path("paper_repro_outputs/cache_similarity_eval_v4_real_200")


@dataclass
class SelectionState:
    dragging: bool = False
    start: Optional[Tuple[int, int]] = None
    end: Optional[Tuple[int, int]] = None
    rect: Optional[Tuple[int, int, int, int]] = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect v4_real_200 ROI samples with webcam and manual metadata."
    )
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--output-root", default=str(DEFAULT_ROOT))
    parser.add_argument("--image-ext", default=".jpg", choices=[".jpg", ".png"])
    parser.add_argument("--window-name", default="v4_real_200 ROI Collector")
    parser.add_argument("--frame-width", type=int, default=1280)
    parser.add_argument("--frame-height", type=int, default=720)
    return parser.parse_args()


def ensure_dirs(root: Path) -> Dict[str, Path]:
    image_dirs = {}
    for sample_type in SAMPLE_TYPES:
        path = root / "roi_images" / sample_type
        path.mkdir(parents=True, exist_ok=True)
        image_dirs[sample_type] = path
    (root / "metadata").mkdir(parents=True, exist_ok=True)
    return image_dirs


def metadata_path(root: Path) -> Path:
    return root / "metadata" / "v4_real_200_metadata.csv"


def read_existing_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def existing_counts(rows: List[Dict[str, str]]) -> Dict[str, int]:
    counts = {sample_type: 0 for sample_type in SAMPLE_TYPES}
    for row in rows:
        sample_type = row.get("sample_type", "")
        if sample_type in counts:
            counts[sample_type] += 1
    return counts


def next_sample_number(rows: List[Dict[str, str]]) -> int:
    max_num = 0
    pattern = re.compile(r"v4_(\d+)")
    for row in rows:
        match = pattern.search(row.get("sample_id", ""))
        if match:
            max_num = max(max_num, int(match.group(1)))
    return max_num + 1


def print_counts(counts: Dict[str, int]) -> None:
    total = sum(counts.values())
    target_total = sum(TARGET_COUNTS.values())
    print("=" * 72)
    print("v4_real_200 current collection progress")
    for sample_type in SAMPLE_TYPES:
        print(f"{sample_type:>14}: {counts.get(sample_type, 0):>3} / {TARGET_COUNTS[sample_type]}")
    print(f"{'total':>14}: {total:>3} / {target_total}")
    print("=" * 72)


def sanitize_token(text: str, default: str = "sample") -> str:
    text = (text or "").strip()
    if not text:
        return default
    safe = []
    for ch in text:
        if ch.isalnum() or ch in ("-", "_"):
            safe.append(ch)
        else:
            safe.append("_")
    value = "".join(safe).strip("_")
    return value[:40] or default


def make_sample_id(number: int) -> str:
    return f"v4_{number:06d}"


def unique_image_path(image_dir: Path, sample_id: str, category: str, image_ext: str) -> Path:
    stem = f"{sample_id}_{sanitize_token(category, 'roi')}"
    candidate = image_dir / f"{stem}{image_ext}"
    index = 1
    while candidate.exists():
        candidate = image_dir / f"{stem}_{index:02d}{image_ext}"
        index += 1
    return candidate


def prompt_until_valid(prompt: str, valid_values: List[str]) -> str:
    valid = set(valid_values)
    while True:
        value = input(prompt).strip()
        if value in valid:
            return value
        print(f"Invalid value. Please enter one of: {', '.join(valid_values)}")


def prompt_yes_no_int(prompt: str) -> str:
    while True:
        value = input(prompt).strip()
        if value in {"0", "1"}:
            return value
        print("Invalid value. Please enter 1 or 0.")


def prompt_default(prompt: str, default: str = "") -> str:
    value = input(prompt).strip()
    return value if value else default


def prompt_metadata(sample_id: str, frame_index: int) -> Dict[str, str]:
    print("-" * 72)
    print(f"Saving sample: {sample_id}")
    sample_type = prompt_until_valid(
        "sample_type [positive / near_positive / hard_negative / negative]: ",
        SAMPLE_TYPES,
    )
    should_hit = prompt_yes_no_int("should_hit [1 / 0]: ")
    object_category = prompt_default("object_category: ")
    query_text = prompt_default("query_text: ")
    candidate_cache_name = prompt_default("candidate_cache_name: ")
    candidate_glb_path = prompt_default("candidate_glb_path: ")
    notes = prompt_default("notes: ")

    print("Optional context fields. Press Enter to keep default/blank.")
    viewpoint = prompt_default("viewpoint [front/side/top/angled/...]: ")
    distance = prompt_default("distance [near/mid/far/...]: ")
    lighting = prompt_default("lighting [normal/dim/bright/backlight/...]: ")
    background = prompt_default("background [clean/cluttered/...]: ")
    occlusion = prompt_default("occlusion [none/partial/heavy/...]: ")
    source_video = prompt_default("source_video [camera]: ", "camera")

    return {
        "sample_id": sample_id,
        "roi_image_path": "",
        "query_text": query_text,
        "keyword": object_category,
        "candidate_cache_name": candidate_cache_name,
        "candidate_glb_path": candidate_glb_path,
        "sample_type": sample_type,
        "should_hit": should_hit,
        "object_category": object_category,
        "viewpoint": viewpoint,
        "distance": distance,
        "lighting": lighting,
        "background": background,
        "occlusion": occlusion,
        "source_video": source_video,
        "frame_index": str(frame_index),
        "notes": notes,
    }


def append_metadata(path: Path, row: Dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = path.exists() and path.stat().st_size > 0
    with path.open("a", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if not file_exists:
            writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in CSV_FIELDS})


def normalize_rect(rect: Tuple[int, int, int, int], width: int, height: int) -> Optional[Tuple[int, int, int, int]]:
    x1, y1, x2, y2 = rect
    left = max(0, min(x1, x2))
    top = max(0, min(y1, y2))
    right = min(width, max(x1, x2))
    bottom = min(height, max(y1, y2))
    if right - left < 8 or bottom - top < 8:
        return None
    return left, top, right, bottom


def make_mouse_callback(state: SelectionState):
    def on_mouse(event, x, y, flags, param) -> None:  # noqa: ANN001
        try:
            import cv2
        except Exception:
            return
        if event == cv2.EVENT_LBUTTONDOWN:
            state.dragging = True
            state.start = (x, y)
            state.end = (x, y)
            state.rect = None
        elif event == cv2.EVENT_MOUSEMOVE and state.dragging:
            state.end = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            state.dragging = False
            state.end = (x, y)
            if state.start and state.end:
                state.rect = (*state.start, *state.end)

    return on_mouse


def draw_overlay(frame, state: SelectionState, counts: Dict[str, int], frame_index: int):  # noqa: ANN001
    import cv2

    display = frame.copy()
    height, width = display.shape[:2]
    active_rect = None
    if state.dragging and state.start and state.end:
        active_rect = (*state.start, *state.end)
    elif state.rect:
        active_rect = state.rect
    if active_rect:
        norm = normalize_rect(active_rect, width, height)
        if norm:
            x1, y1, x2, y2 = norm
            cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 255), 2)

    lines = [
        "Drag mouse to select ROI | s: save | c: clear | q: quit",
        f"frame_index={frame_index}",
        "counts: "
        + " | ".join(
            f"{sample_type} {counts.get(sample_type, 0)}/{TARGET_COUNTS[sample_type]}"
            for sample_type in SAMPLE_TYPES
        ),
    ]
    y = 28
    for line in lines:
        cv2.putText(display, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 255, 0), 2, cv2.LINE_AA)
        y += 26
    return display


def save_selected_roi(
    frame,
    state: SelectionState,
    image_dirs: Dict[str, Path],
    csv_path: Path,
    sample_number: int,
    image_ext: str,
    frame_index: int,
) -> Tuple[bool, int, Optional[str]]:
    import cv2

    if not state.rect:
        print("No ROI selected. Drag a rectangle first.")
        return False, sample_number, None

    height, width = frame.shape[:2]
    rect = normalize_rect(state.rect, width, height)
    if not rect:
        print("ROI too small. Please select a larger region.")
        return False, sample_number, None

    sample_id = make_sample_id(sample_number)
    metadata = prompt_metadata(sample_id, frame_index)
    sample_type = metadata["sample_type"]
    x1, y1, x2, y2 = rect
    roi = frame[y1:y2, x1:x2]
    image_path = unique_image_path(image_dirs[sample_type], sample_id, metadata["object_category"], image_ext)

    ok = cv2.imwrite(str(image_path), roi)
    if not ok:
        print(f"Failed to save ROI image: {image_path}")
        return False, sample_number, None

    metadata["roi_image_path"] = str(image_path)
    append_metadata(csv_path, metadata)
    print(f"Saved ROI: {image_path}")
    return True, sample_number + 1, sample_type


def main() -> None:
    args = parse_args()
    root = Path(args.output_root)
    image_dirs = ensure_dirs(root)
    csv_path = metadata_path(root)
    rows = read_existing_rows(csv_path)
    counts = existing_counts(rows)
    sample_number = next_sample_number(rows)
    print_counts(counts)
    print(f"metadata_csv: {csv_path}")
    print("Controls: drag left mouse to select ROI, press s to save, c to clear, q to quit.")

    try:
        import cv2
    except ModuleNotFoundError:
        print("ERROR: opencv-python/cv2 is not installed in this environment.")
        print("Install OpenCV before running the collector, then retry.")
        sys.exit(1)

    cap = cv2.VideoCapture(args.camera_index)
    if not cap.isOpened():
        print(f"ERROR: failed to open camera index {args.camera_index}")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.frame_height)
    state = SelectionState()
    cv2.namedWindow(args.window_name)
    cv2.setMouseCallback(args.window_name, make_mouse_callback(state))

    frame_index = 0
    last_frame = None
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("WARNING: failed to read camera frame.")
                time.sleep(0.1)
                continue
            frame_index += 1
            last_frame = frame.copy()
            display = draw_overlay(frame, state, counts, frame_index)
            cv2.imshow(args.window_name, display)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("c"):
                state.rect = None
                state.start = None
                state.end = None
                print("ROI selection cleared.")
            if key == ord("s"):
                saved, sample_number, sample_type = save_selected_roi(
                    last_frame,
                    state,
                    image_dirs,
                    csv_path,
                    sample_number,
                    args.image_ext,
                    frame_index,
                )
                if saved and sample_type:
                    counts[sample_type] += 1
                    print_counts(counts)
                    state.rect = None
                    state.start = None
                    state.end = None
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print_counts(counts)
        print(f"metadata_csv: {csv_path}")
        print("Collector closed.")


if __name__ == "__main__":
    main()
