#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Prepare directories and metadata templates for real_challenge_50.

This script does not call Qwen, TripoSR, Stable Fast 3D, or modify plus.py.
It only creates a challenge-set workspace for collecting real camera ROI cases.
"""

from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path("paper_repro_outputs/cache_similarity_eval_real_challenge_50")
ROI_DIR = ROOT / "roi_images"
METADATA_DIR = ROOT / "metadata"
METADATA_CSV = METADATA_DIR / "real_challenge_50_metadata.csv"
FEATURE_TEMPLATE_CSV = ROOT / "real_challenge_50_features_template.csv"
GUIDE_MD = ROOT / "real_challenge_50_sampling_guide.md"

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
    "viewpoint",
    "distance",
    "lighting",
    "background",
    "occlusion",
    "notes",
]

FEATURE_FIELDS = [
    "sample_id",
    "image",
    "should_hit",
    "text_score",
    "image_score",
    "fusion_score",
    "score_abs_diff",
    "score_max",
    "score_min",
    "sample_type",
    "candidate_cache_name",
    "notes",
]


def ensure_csv(path: Path, fields: list[str]) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        csv.DictWriter(f, fieldnames=fields).writeheader()


def write_guide() -> None:
    text = """# real_challenge_50 真实挑战集采集说明

## 1. 采集目的

public_train_755 迁移到真实 v3_real_70 后出现较高 false_hit，说明公开数据类别代理分数存在明显 domain gap。本阶段改为补充真实摄像头 ROI 困难样本，构造 real_challenge_50，并与 v3_real_70 合并为 real_eval_120，用于训练保守型学习式融合模型。

## 2. 样本目标

- near_positive: 25
- hard_negative: 25
- total: 50

near_positive 用于提升召回，hard_negative 用于压低 false_hit_rate。两类必须尽量接近当前缓存模型边界，而不是收集明显简单样本。

## 3. 采集建议

near_positive:

- 相似眼镜、不同角度眼镜、遮挡眼镜；
- 相似人头 / 人脸；
- 与已有缓存模型同类但不完全相同的目标；
- 分数可能落在 review 区附近的样本。

hard_negative:

- 像眼镜但不是眼镜的物体；
- 手、遮挡区域、反光区域、纸上图案；
- 像人脸但不是人脸的区域；
- 容易被图像相似度误判的物体。

## 4. 标签规则

- near_positive: should_hit = 1
- hard_negative: should_hit = 0

## 5. 后续特征

训练脚本只使用线上可获得分数特征：

- text_score
- image_score
- fusion_score
- score_abs_diff
- score_max
- score_min

不使用 sample_type / category 作为模型输入。

## 6. 当前限制

本目录只负责采集和整理，不调用 Qwen，不调用 TripoSR，不运行 Stable Fast 3D，不修改 plus.py。模型训练暂不接入工程主流程。
"""
    ROOT.mkdir(parents=True, exist_ok=True)
    GUIDE_MD.write_text(text, encoding="utf-8-sig")


def main() -> None:
    (ROI_DIR / "near_positive").mkdir(parents=True, exist_ok=True)
    (ROI_DIR / "hard_negative").mkdir(parents=True, exist_ok=True)
    METADATA_DIR.mkdir(parents=True, exist_ok=True)
    ensure_csv(METADATA_CSV, METADATA_FIELDS)
    ensure_csv(FEATURE_TEMPLATE_CSV, FEATURE_FIELDS)
    write_guide()

    print("=" * 72)
    print(f"real_challenge_root: {ROOT}")
    print("target_near_positive: 25")
    print("target_hard_negative: 25")
    print(f"metadata_csv: {METADATA_CSV}")
    print(f"feature_template_csv: {FEATURE_TEMPLATE_CSV}")
    print(f"sampling_guide: {GUIDE_MD}")
    print("=" * 72)


if __name__ == "__main__":
    main()
