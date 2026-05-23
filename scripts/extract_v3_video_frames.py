from __future__ import annotations

import argparse
import csv
import math
import shutil
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
TARGET_LIMITS = {
    "near_positive": 10,
    "hard_negative": 10,
    "negative": 5,
}


def classify_video(path: Path) -> Optional[str]:
    name = path.name.lower()
    if "head_near" in name or "glasses_near" in name:
        return "near_positive"
    if "hard_negative" in name:
        return "hard_negative"
    if "negative" in name:
        return "negative"
    return None


def safe_stem(path: Path) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in path.stem)[:80]


def list_videos(video_dir: Path) -> List[Path]:
    if not video_dir.exists():
        return []
    videos = [p for p in video_dir.iterdir() if p.is_file() and p.suffix.lower() in VIDEO_EXTS]
    videos.sort(key=lambda p: p.name.lower())
    return videos


def average_hash(image_bgr: np.ndarray, hash_size: int = 8) -> Optional[str]:
    try:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(gray, (hash_size, hash_size), interpolation=cv2.INTER_AREA)
        avg = float(small.mean())
        bits = small > avg
        return "".join("1" if value else "0" for value in bits.flatten())
    except Exception:
        return None


def hamming(a: str, b: str) -> int:
    return sum(ch1 != ch2 for ch1, ch2 in zip(a, b))


def quality_check(frame_bgr: np.ndarray, seen_hashes: List[str]) -> Tuple[bool, str, bool, Dict[str, float | int | str]]:
    metrics: Dict[str, float | int | str] = {}
    try:
        height, width = frame_bgr.shape[:2]
        metrics["width"] = width
        metrics["height"] = height
        if width < 64 or height < 64:
            return False, "size_too_small", False, metrics

        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        brightness = float(gray.mean())
        blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        metrics["brightness"] = round(brightness, 3)
        metrics["blur"] = round(blur, 3)

        if brightness < 35:
            return False, "too_dark", False, metrics
        if blur < 35:
            return False, "too_blurry", False, metrics

        ahash = average_hash(frame_bgr)
        metrics["hash"] = ahash or ""
        if ahash:
            for old_hash in seen_hashes:
                if hamming(ahash, old_hash) <= 5:
                    return False, "duplicate_frame", False, metrics
            seen_hashes.append(ahash)
            return True, "clear_candidate", False, metrics

        return True, "quality_uncertain_hash_failed", True, metrics
    except Exception as exc:
        metrics["error"] = str(exc)
        return True, "quality_uncertain_exception", True, metrics


def next_target_path(target_dir: Path, target_type: str, source_frame: Path) -> Path:
    target_dir.mkdir(parents=True, exist_ok=True)
    original = safe_stem(source_frame)
    suffix = source_frame.suffix.lower() or ".jpg"
    index = 1
    while True:
        candidate = target_dir / f"{target_type}_{index:03d}_{original}{suffix}"
        if not candidate.exists():
            return candidate
        index += 1


def fit_image(path: Path, thumb_size: int) -> Tuple[Image.Image, int, int]:
    with Image.open(path) as img:
        width, height = img.size
        img = img.convert("RGB")
        img.thumbnail((thumb_size, thumb_size), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (thumb_size, thumb_size), "white")
        canvas.paste(img, ((thumb_size - img.width) // 2, (thumb_size - img.height) // 2))
        return canvas, width, height


def truncate(text: str, max_len: int = 28) -> str:
    return text if len(text) <= max_len else text[: max_len - 3] + "..."


def make_contact_sheet(
    rows: List[Dict[str, object]],
    output_path: Path,
    csv_path: Optional[Path] = None,
    thumb_size: int = 150,
    cols: int = 5,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cell_w = 220
    cell_h = 220
    margin = 18
    row_count = max(1, math.ceil(len(rows) / max(1, cols)))
    sheet = Image.new("RGB", (margin * 2 + cols * cell_w, margin * 2 + row_count * cell_h), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    csv_rows = []

    for index, row in enumerate(rows, start=1):
        path = Path(str(row["frame_path"]))
        grid_row = (index - 1) // cols
        grid_col = (index - 1) % cols
        x = margin + grid_col * cell_w
        y = margin + grid_row * cell_h
        width = height = 0
        try:
            fitted, width, height = fit_image(path, thumb_size)
        except Exception:
            fitted = Image.new("RGB", (thumb_size, thumb_size), (235, 235, 235))
        sheet.paste(fitted, (x, y + 48))
        label = f"[{index}] {row.get('target_type', '')} {truncate(path.name, 20)}"
        reason = truncate(str(row.get("reason", "")), 30)
        draw.text((x, y), label, fill="black", font=font)
        draw.text((x, y + 18), f"selected={row.get('selected', '')}", fill="black", font=font)
        draw.text((x, y + 34), reason, fill="black", font=font)
        csv_rows.append(
            {
                "index": index,
                "frame_path": str(path),
                "source_video": row.get("source_video", ""),
                "target_type": row.get("target_type", ""),
                "selected": row.get("selected", ""),
                "reason": row.get("reason", ""),
                "width": width,
                "height": height,
            }
        )

    sheet.save(output_path)
    if csv_path:
        with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
            fieldnames = ["index", "frame_path", "source_video", "target_type", "selected", "reason", "width", "height"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_rows)


def extract_and_sort(
    video_dir: Path,
    output_root: Path,
    target_root: Path,
    fps: float = 1.0,
    max_frames_per_video: int = 30,
) -> Dict[str, object]:
    frames_dir = output_root / "frames"
    sheets_dir = output_root / "contact_sheets"
    frames_dir.mkdir(parents=True, exist_ok=True)
    sheets_dir.mkdir(parents=True, exist_ok=True)

    videos = list_videos(video_dir)
    frame_rows: List[Dict[str, object]] = []
    sort_rows: List[Dict[str, object]] = []
    selected_counts = {key: 0 for key in TARGET_LIMITS}
    candidate_counts = {key: 0 for key in TARGET_LIMITS}

    for video in videos:
        target_type = classify_video(video)
        cap = cv2.VideoCapture(str(video))
        if not cap.isOpened():
            continue
        source_fps = cap.get(cv2.CAP_PROP_FPS) or 0
        if source_fps <= 0:
            frame_step = 30
        else:
            frame_step = max(1, int(round(source_fps / max(fps, 0.1))))

        video_hashes: List[str] = []
        saved_count = 0
        frame_index = 0
        while saved_count < max_frames_per_video:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_index % frame_step != 0:
                frame_index += 1
                continue

            stem = safe_stem(video)
            frame_path = frames_dir / f"{stem}_frame_{saved_count + 1:03d}.jpg"
            cv2.imwrite(str(frame_path), frame)
            selected_by_quality, reason, needs_review, metrics = quality_check(frame, video_hashes)

            selected = False
            target_path = ""
            if (
                target_type in TARGET_LIMITS
                and selected_by_quality
                and candidate_counts[target_type] < TARGET_LIMITS[target_type]
            ):
                candidate_counts[target_type] += 1
                target_dir = target_root / target_type
                final_path = next_target_path(target_dir, target_type, frame_path)
                try:
                    shutil.copy2(frame_path, final_path)
                    selected = True
                    target_path = str(final_path)
                    selected_counts[target_type] += 1
                except PermissionError as exc:
                    reason = f"copy_permission_denied: {exc}"
                    needs_review = True
                except OSError as exc:
                    reason = f"copy_failed: {exc}"
                    needs_review = True

            row = {
                "source_video": str(video),
                "frame_path": str(frame_path),
                "target_type": target_type or "",
                "target_path": target_path,
                "selected": str(selected).lower(),
                "reason": reason if target_type else "unknown_video_type",
                "needs_review": str(needs_review).lower(),
                "frame_index": frame_index,
                "timestamp_sec": round(frame_index / source_fps, 3) if source_fps > 0 else "",
                "width": metrics.get("width", ""),
                "height": metrics.get("height", ""),
                "brightness": metrics.get("brightness", ""),
                "blur": metrics.get("blur", ""),
            }
            frame_rows.append(row)
            sort_rows.append(
                {
                    "source_video": row["source_video"],
                    "frame_path": row["frame_path"],
                    "target_type": row["target_type"],
                    "target_path": row["target_path"],
                    "selected": row["selected"],
                    "reason": row["reason"],
                    "needs_review": row["needs_review"],
                }
            )

            saved_count += 1
            frame_index += 1
        cap.release()

    frame_index_path = output_root / "frame_index.csv"
    with frame_index_path.open("w", newline="", encoding="utf-8-sig") as f:
        fieldnames = [
            "source_video",
            "frame_path",
            "target_type",
            "target_path",
            "selected",
            "reason",
            "needs_review",
            "frame_index",
            "timestamp_sec",
            "width",
            "height",
            "brightness",
            "blur",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(frame_rows)

    auto_sort_log_path = output_root / "auto_sort_log.csv"
    with auto_sort_log_path.open("w", newline="", encoding="utf-8-sig") as f:
        fieldnames = ["source_video", "frame_path", "target_type", "target_path", "selected", "reason", "needs_review"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(sort_rows)

    all_sheet_path = sheets_dir / "contact_sheet_all.png"
    all_sheet_csv = sheets_dir / "contact_sheet_all.csv"
    make_contact_sheet(frame_rows, all_sheet_path, all_sheet_csv)

    selected_rows = [
        row
        for row in frame_rows
        if row.get("selected") == "true" or str(row.get("reason", "")).startswith("copy_")
    ]
    selected_sheet_path = sheets_dir / "contact_sheet_selected_review.png"
    selected_sheet_csv = sheets_dir / "contact_sheet_selected_review.csv"
    make_contact_sheet(selected_rows, selected_sheet_path, selected_sheet_csv)

    return {
        "videos": len(videos),
        "extracted_frames": len(frame_rows),
        "selected_counts": selected_counts,
        "candidate_counts": candidate_counts,
        "frame_index": frame_index_path,
        "auto_sort_log": auto_sort_log_path,
        "contact_sheet_all": all_sheet_path,
        "contact_sheet_selected_review": selected_sheet_path,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract v3 self-capture video frames and auto-sort candidates.")
    parser.add_argument("--video-dir", default="v3_self_capture/videos")
    parser.add_argument("--output-root", default="paper_repro_outputs/v3_video_frame_candidates")
    parser.add_argument("--target-root", default="cache_test_v3_real")
    parser.add_argument("--fps", type=float, default=1.0)
    parser.add_argument("--max-frames-per-video", type=int, default=30)
    args = parser.parse_args()

    stats = extract_and_sort(
        video_dir=Path(args.video_dir),
        output_root=Path(args.output_root),
        target_root=Path(args.target_root),
        fps=args.fps,
        max_frames_per_video=args.max_frames_per_video,
    )

    counts: Dict[str, int] = stats["selected_counts"]  # type: ignore[assignment]
    candidate_counts: Dict[str, int] = stats["candidate_counts"]  # type: ignore[assignment]
    print("=" * 72)
    print("v3 video frame extraction and auto-sort completed")
    print(f"videos: {stats['videos']}")
    print(f"extracted_frames: {stats['extracted_frames']}")
    print(f"near_positive_added: {counts['near_positive']}")
    print(f"hard_negative_added: {counts['hard_negative']}")
    print(f"negative_added: {counts['negative']}")
    print(f"near_positive_candidate_count: {candidate_counts['near_positive']}")
    print(f"hard_negative_candidate_count: {candidate_counts['hard_negative']}")
    print(f"negative_candidate_count: {candidate_counts['negative']}")
    print(f"frame_index.csv: {stats['frame_index']}")
    print(f"auto_sort_log.csv: {stats['auto_sort_log']}")
    print(f"contact_sheet_all.png: {stats['contact_sheet_all']}")
    print(f"contact_sheet_selected_review.png: {stats['contact_sheet_selected_review']}")
    print("=" * 72)


if __name__ == "__main__":
    main()
