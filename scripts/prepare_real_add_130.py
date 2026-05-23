#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Prepare the real_add_130 workspace for real_eval_200 collection.

This script only creates directories, metadata templates, sampling guidance,
and a current-count report. It does not call Qwen, TripoSR, Stable Fast 3D, or
modify plus.py.
"""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List


ROOT = Path("paper_repro_outputs/cache_similarity_eval_real_add_130")
RAW_VIDEOS_DIR = ROOT / "raw_videos"
ROI_DIR = ROOT / "roi_images"
METADATA_DIR = ROOT / "metadata"
REPORTS_DIR = ROOT / "reports"
METADATA_CSV = METADATA_DIR / "real_add_130_metadata.csv"
SAMPLING_GUIDE = REPORTS_DIR / "real_add_130_sampling_guide.md"
COUNT_REPORT_MD = REPORTS_DIR / "real_add_130_count_report.md"
COUNT_REPORT_JSON = REPORTS_DIR / "real_add_130_count_report.json"

SAMPLE_TYPES = ["positive", "near_positive", "hard_negative", "negative"]
TARGET_BY_TYPE = {
    "positive": 28,
    "near_positive": 36,
    "hard_negative": 36,
    "negative": 30,
}
TARGET_BY_OBJECT_AND_TYPE = {
    "眼镜": {"positive": 7, "near_positive": 9, "hard_negative": 9, "negative": 8},
    "键盘": {"positive": 7, "near_positive": 9, "hard_negative": 9, "negative": 8},
    "纸巾盒": {"positive": 7, "near_positive": 9, "hard_negative": 9, "negative": 7},
    "网球拍": {"positive": 7, "near_positive": 9, "hard_negative": 9, "negative": 7},
}
CANDIDATE_CACHE = {
    "眼镜": {
        "keyword": "眼镜",
        "candidate_cache_name": "眼镜",
        "candidate_glb_path": "paper_repro_outputs/cache_similarity_model_cache_eval_70/眼镜_db9b92eef7.glb",
        "positive_query": "眼镜 帮助人看清东西的工具",
        "near_query": "相似眼镜图像，可以复用眼镜模型",
        "negative_query": "无关物体或相似干扰区域，不应复用眼镜模型",
    },
    "键盘": {
        "keyword": "电脑键盘",
        "candidate_cache_name": "电脑键盘",
        "candidate_glb_path": "paper_repro_outputs/cache_similarity_model_cache_eval_70/电脑键盘_1e24316d44.glb",
        "positive_query": "电脑键盘 用来输入文字的设备",
        "near_query": "相似键盘或按键区域，可以考虑复用键盘模型",
        "negative_query": "无关物体或相似干扰区域，不应复用键盘模型",
    },
    "纸巾盒": {
        "keyword": "纸巾盒",
        "candidate_cache_name": "纸巾盒",
        "candidate_glb_path": "paper_repro_outputs/cache_similarity_model_cache_eval_70/纸巾盒_ea7bcc88a0.glb",
        "positive_query": "纸巾盒 用来放纸巾的盒子",
        "near_query": "相似盒子或包装盒，可以考虑复用纸巾盒模型",
        "negative_query": "无关物体或相似干扰区域，不应复用纸巾盒模型",
    },
    "网球拍": {
        "keyword": "网球拍",
        "candidate_cache_name": "网球拍",
        "candidate_glb_path": "paper_repro_outputs/cache_similarity_model_cache_eval_70/网球拍_efa3e2b1e3.glb",
        "positive_query": "网球拍 用来击打网球的球拍",
        "near_query": "相似球拍或长条运动器材，可以考虑复用网球拍模型",
        "negative_query": "无关物体或相似干扰区域，不应复用网球拍模型",
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


def ensure_metadata_csv() -> None:
    METADATA_DIR.mkdir(parents=True, exist_ok=True)
    if METADATA_CSV.exists():
        return
    with METADATA_CSV.open("w", encoding="utf-8-sig", newline="") as f:
        csv.DictWriter(f, fieldnames=METADATA_FIELDS).writeheader()


def read_metadata_rows() -> List[Dict[str, str]]:
    if not METADATA_CSV.exists():
        return []
    with METADATA_CSV.open("r", encoding="utf-8-sig", newline="") as f:
        return [row for row in csv.DictReader(f) if any((v or "").strip() for v in row.values())]


def image_exists(row: Dict[str, str]) -> bool:
    path = (row.get("roi_image_path") or "").strip()
    return bool(path) and Path(path).exists()


def should_hit_expected(sample_type: str) -> str:
    return "1" if sample_type in {"positive", "near_positive"} else "0"


def summarize(rows: List[Dict[str, str]]) -> Dict[str, object]:
    by_type = Counter(row.get("sample_type", "") for row in rows)
    by_object_type = defaultdict(Counter)
    missing_images = []
    invalid_labels = []
    empty_required = []
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
        object_category = row.get("object_category", "")
        sample_type = row.get("sample_type", "")
        by_object_type[object_category][sample_type] += 1
        if not image_exists(row):
            missing_images.append(row.get("sample_id") or row.get("roi_image_path") or "")
        if sample_type in SAMPLE_TYPES and str(row.get("should_hit", "")).strip() != should_hit_expected(sample_type):
            invalid_labels.append(row.get("sample_id", ""))
        if any(not (row.get(field) or "").strip() for field in required):
            empty_required.append(row.get("sample_id", ""))

    return {
        "total_samples": len(rows),
        "count_by_sample_type": {key: by_type.get(key, 0) for key in SAMPLE_TYPES},
        "target_by_sample_type": TARGET_BY_TYPE,
        "remaining_by_sample_type": {
            key: max(0, TARGET_BY_TYPE[key] - by_type.get(key, 0)) for key in SAMPLE_TYPES
        },
        "count_by_object_and_type": {
            obj: {typ: by_object_type[obj].get(typ, 0) for typ in SAMPLE_TYPES}
            for obj in TARGET_BY_OBJECT_AND_TYPE
        },
        "target_by_object_and_type": TARGET_BY_OBJECT_AND_TYPE,
        "missing_image_count": len(missing_images),
        "invalid_label_count": len(invalid_labels),
        "empty_required_count": len(empty_required),
        "missing_images": missing_images[:30],
        "invalid_labels": invalid_labels[:30],
        "empty_required": empty_required[:30],
        "complete": (
            len(rows) == 130
            and all(by_type.get(key, 0) == TARGET_BY_TYPE[key] for key in SAMPLE_TYPES)
            and not missing_images
            and not invalid_labels
            and not empty_required
        ),
    }


def write_sampling_guide() -> None:
    rows = []
    for obj, targets in TARGET_BY_OBJECT_AND_TYPE.items():
        cache = CANDIDATE_CACHE[obj]
        rows.append(f"### {obj}")
        rows.append("")
        rows.append(f"- keyword: `{cache['keyword']}`")
        rows.append(f"- candidate_cache_name: `{cache['candidate_cache_name']}`")
        rows.append(f"- candidate_glb_path: `{cache['candidate_glb_path']}`")
        rows.append(f"- positive_query: {cache['positive_query']}")
        rows.append(f"- near_query: {cache['near_query']}")
        rows.append(f"- negative_query: {cache['negative_query']}")
        rows.append("")
        rows.append("| sample_type | target | should_hit |")
        rows.append("| --- | ---: | ---: |")
        for sample_type in SAMPLE_TYPES:
            rows.append(f"| {sample_type} | {targets[sample_type]} | {should_hit_expected(sample_type)} |")
        rows.append("")

    text = f"""# real_add_130 真实 ROI 采集说明

## 1. 目标

基于已有 v3_real_70，新增 130 条真实摄像头 ROI，合并形成 real_eval_200，用于训练和评估保守型学习式图文融合模型。

本工作区只负责采集 ROI 与补齐 metadata，不调用 Qwen、不调用 TripoSR、不运行 Stable Fast 3D、不修改 plus.py。

## 2. 总体数量

| sample_type | target |
| --- | ---: |
| positive | 28 |
| near_positive | 36 |
| hard_negative | 36 |
| negative | 30 |
| total | 130 |

标签规则：

- positive / near_positive: `should_hit = 1`
- hard_negative / negative: `should_hit = 0`

## 3. 分物体采集目标

{chr(10).join(rows)}

## 4. 采集建议

- positive: 同一缓存目标的不同角度、光照、距离，尽量主体清晰。
- near_positive: 同类但不完全相同，或边界相似样本，用于提高召回和 review 判断。
- hard_negative: 外观相似但不应复用的区域，用于压低 false_hit_rate。
- negative: 明显无关物体或背景，用于检验 miss 安全性。

## 5. metadata 填写要求

每条样本至少填写：

`sample_id, roi_image_path, query_text, keyword, candidate_cache_name, candidate_glb_path, sample_type, should_hit, object_category, source_video, frame_index, timestamp_sec, notes`

如果来自摄像头直接截图，`source_video` 可填 `live_camera`，`frame_index` 和 `timestamp_sec` 可按实际情况留空或填 0。

## 6. 后续命令

采集满 130 条后运行：

```powershell
.\\.venv\\Scripts\\python.exe compute_real_add_130_features.py --train-if-complete
```

只有真实特征完整后才会进入 real_eval_200 训练。
"""
    SAMPLING_GUIDE.write_text(text, encoding="utf-8-sig")


def write_count_reports(stats: Dict[str, object]) -> None:
    COUNT_REPORT_JSON.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    table_rows = [
        "| sample_type | current | target | remaining |",
        "| --- | ---: | ---: | ---: |",
    ]
    counts = stats["count_by_sample_type"]  # type: ignore[index]
    remaining = stats["remaining_by_sample_type"]  # type: ignore[index]
    for sample_type in SAMPLE_TYPES:
        table_rows.append(
            f"| {sample_type} | {counts[sample_type]} | {TARGET_BY_TYPE[sample_type]} | {remaining[sample_type]} |"
        )

    object_rows = ["| object_category | positive | near_positive | hard_negative | negative |", "| --- | ---: | ---: | ---: | ---: |"]
    object_counts = stats["count_by_object_and_type"]  # type: ignore[index]
    for obj in TARGET_BY_OBJECT_AND_TYPE:
        item = object_counts[obj]
        object_rows.append(
            f"| {obj} | {item['positive']} / {TARGET_BY_OBJECT_AND_TYPE[obj]['positive']} | "
            f"{item['near_positive']} / {TARGET_BY_OBJECT_AND_TYPE[obj]['near_positive']} | "
            f"{item['hard_negative']} / {TARGET_BY_OBJECT_AND_TYPE[obj]['hard_negative']} | "
            f"{item['negative']} / {TARGET_BY_OBJECT_AND_TYPE[obj]['negative']} |"
        )

    text = f"""# real_add_130 当前采集数量报告

## 1. 总体状态

- total_samples: {stats['total_samples']} / 130
- missing_image_count: {stats['missing_image_count']}
- invalid_label_count: {stats['invalid_label_count']}
- empty_required_count: {stats['empty_required_count']}
- complete: {stats['complete']}

## 2. 按样本类型统计

{chr(10).join(table_rows)}

## 3. 按候选缓存模型统计

{chr(10).join(object_rows)}

## 4. 下一步

如果 `complete=False`，请继续补齐 ROI 图片和 metadata。采满后再运行：

```powershell
.\\.venv\\Scripts\\python.exe compute_real_add_130_features.py --train-if-complete
```
"""
    COUNT_REPORT_MD.write_text(text, encoding="utf-8-sig")


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    RAW_VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    for sample_type in SAMPLE_TYPES:
        (ROI_DIR / sample_type).mkdir(parents=True, exist_ok=True)
    ensure_metadata_csv()
    write_sampling_guide()
    rows = read_metadata_rows()
    stats = summarize(rows)
    write_count_reports(stats)

    print("=" * 72)
    print(f"real_add_130_root: {ROOT}")
    print(f"metadata_csv: {METADATA_CSV}")
    print(f"sampling_guide: {SAMPLING_GUIDE}")
    print(f"count_report: {COUNT_REPORT_MD}")
    print(f"total_samples: {stats['total_samples']} / 130")
    print(f"count_by_sample_type: {stats['count_by_sample_type']}")
    print("=" * 72)


if __name__ == "__main__":
    main()
