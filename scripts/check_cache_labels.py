from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple


KNOWN_TYPES = {"positive", "near_positive", "hard_negative", "negative"}
EXPECTED_SHOULD_HIT = {
    "positive": True,
    "near_positive": True,
    "hard_negative": False,
    "negative": False,
}
QUERY_TEXT_RULES = {
    "眼镜": "眼镜 帮助人看清东西的工具",
    "人头": "人头 人的头部或脸部",
    "人脸": "人脸 人的脸部",
    "杯子": "杯子 用来喝水的容器",
    "书": "书 用来阅读的物品",
    "手": "手 人体用于抓握和操作的部位",
    "背景": "背景 画面中的背景区域",
    "桌面": "桌面 用来放置物品的平面",
    "键盘": "键盘 用来输入文字的设备",
    "鼠标": "鼠标 用来控制电脑光标的设备",
}
REQUIRED_FIELDS = ["image", "category", "sample_type", "query_text", "should_hit"]


def read_labels(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError("请先运行 run_cache_v2_experiment.py --rebuild-labels 生成数据集。")
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_labels(path: Path, data: Dict[str, Any]) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def bool_value(value: Any) -> Any:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y", "hit", "命中", "是"}:
        return True
    if text in {"false", "0", "no", "n", "miss", "不命中", "否"}:
        return False
    return value


def fix_empty_query_text(samples: List[Dict[str, Any]]) -> int:
    fixed = 0
    for sample in samples:
        query_text = str(sample.get("query_text") or "").strip()
        category = str(sample.get("category") or "").strip()
        if query_text or not category:
            continue
        if category in QUERY_TEXT_RULES:
            sample["query_text"] = QUERY_TEXT_RULES[category]
            fixed += 1
    return fixed


def check_labels(
    labels_path: Path,
    dataset_dir: Path,
    output_path: Path,
    fix_query_text: bool = False,
) -> Dict[str, Any]:
    data = read_labels(labels_path)
    if "samples" not in data or not isinstance(data["samples"], list):
        raise ValueError("labels.json 必须包含 samples 字段，且 samples 必须是 list。")

    samples = data["samples"]
    fixed_count = 0
    if fix_query_text:
        fixed_count = fix_empty_query_text(samples)
        if fixed_count:
            write_labels(labels_path, data)

    stats = {
        "total_samples": len(samples),
        "positive_count": 0,
        "near_positive_count": 0,
        "hard_negative_count": 0,
        "negative_count": 0,
        "other_sample_type_count": 0,
        "empty_category_count": 0,
        "empty_query_text_count": 0,
        "missing_image_count": 0,
        "inconsistent_should_hit_count": 0,
        "fixed_query_text_count": fixed_count,
    }

    missing_fields: List[Tuple[int, List[str]]] = []
    missing_images: List[Tuple[int, str]] = []
    inconsistent: List[Tuple[int, str, Any, Any]] = []
    unknown_types: List[Tuple[int, str]] = []
    manual_check: List[Tuple[int, str, str]] = []

    for idx, sample in enumerate(samples):
        if not isinstance(sample, dict):
            missing_fields.append((idx, REQUIRED_FIELDS))
            manual_check.append((idx, "", "sample is not an object"))
            continue

        missing = [field for field in REQUIRED_FIELDS if field not in sample]
        if missing:
            missing_fields.append((idx, missing))
            manual_check.append((idx, str(sample.get("image", "")), f"missing fields: {missing}"))

        image = str(sample.get("image") or "").strip()
        category = str(sample.get("category") or "").strip()
        query_text = str(sample.get("query_text") or "").strip()
        sample_type = str(sample.get("sample_type") or "").strip()
        should_hit = bool_value(sample.get("should_hit"))

        if sample_type == "positive":
            stats["positive_count"] += 1
        elif sample_type == "near_positive":
            stats["near_positive_count"] += 1
        elif sample_type == "hard_negative":
            stats["hard_negative_count"] += 1
        elif sample_type == "negative":
            stats["negative_count"] += 1
        else:
            stats["other_sample_type_count"] += 1
            unknown_types.append((idx, sample_type))
            manual_check.append((idx, image, f"unknown sample_type: {sample_type}"))

        if not category:
            stats["empty_category_count"] += 1
            manual_check.append((idx, image, "empty category"))
        if not query_text:
            stats["empty_query_text_count"] += 1
            manual_check.append((idx, image, "empty query_text"))

        image_path = dataset_dir / image
        if not image or not image_path.exists():
            stats["missing_image_count"] += 1
            missing_images.append((idx, image))
            manual_check.append((idx, image, "missing image"))

        if sample_type in EXPECTED_SHOULD_HIT:
            expected = EXPECTED_SHOULD_HIT[sample_type]
            if should_hit is not expected:
                stats["inconsistent_should_hit_count"] += 1
                inconsistent.append((idx, sample_type, should_hit, expected))
                manual_check.append(
                    (idx, image, f"should_hit={should_hit}, expected={expected}")
                )

    can_run = not (
        stats["missing_image_count"] > 0
        or stats["empty_category_count"] > 0
        or stats["empty_query_text_count"] > 0
        or stats["inconsistent_should_hit_count"] > 0
    )
    stats["can_run_official_experiment"] = can_run

    write_report(
        output_path=output_path,
        labels_path=labels_path,
        dataset_dir=dataset_dir,
        stats=stats,
        missing_fields=missing_fields,
        missing_images=missing_images,
        inconsistent=inconsistent,
        unknown_types=unknown_types,
        manual_check=manual_check,
    )
    return stats


def list_lines(title: str, rows: List[Any]) -> List[str]:
    lines = [f"## {title}", ""]
    if not rows:
        lines.append("无")
        lines.append("")
        return lines
    for row in rows:
        lines.append(f"- {row}")
    lines.append("")
    return lines


def write_report(
    output_path: Path,
    labels_path: Path,
    dataset_dir: Path,
    stats: Dict[str, Any],
    missing_fields: List[Tuple[int, List[str]]],
    missing_images: List[Tuple[int, str]],
    inconsistent: List[Tuple[int, str, Any, Any]],
    unknown_types: List[Tuple[int, str]],
    manual_check: List[Tuple[int, str, str]],
) -> None:
    lines = [
        "# labels.json 检查报告",
        "",
        f"- 检查时间: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"- labels.json: {labels_path}",
        f"- dataset_dir: {dataset_dir}",
        "",
        "## 统计信息",
        "",
        f"- total_samples: {stats['total_samples']}",
        f"- positive_count: {stats['positive_count']}",
        f"- near_positive_count: {stats['near_positive_count']}",
        f"- hard_negative_count: {stats['hard_negative_count']}",
        f"- negative_count: {stats['negative_count']}",
        f"- other_sample_type_count: {stats['other_sample_type_count']}",
        f"- empty_category_count: {stats['empty_category_count']}",
        f"- empty_query_text_count: {stats['empty_query_text_count']}",
        f"- missing_image_count: {stats['missing_image_count']}",
        f"- inconsistent_should_hit_count: {stats['inconsistent_should_hit_count']}",
        f"- fixed_query_text_count: {stats['fixed_query_text_count']}",
        "",
        "## 是否建议继续运行实验",
        "",
    ]
    if stats["can_run_official_experiment"]:
        lines.append("labels.json 检查通过，可以继续运行实验。")
    else:
        lines.append("不建议直接运行正式实验，请先人工检查 labels.json。")
    lines.append("")

    lines.extend(list_lines("缺失字段样本列表", missing_fields))
    lines.extend(list_lines("缺失图片列表", missing_images))
    lines.extend(list_lines("should_hit 与 sample_type 不一致的样本列表", inconsistent))
    lines.extend(list_lines("未知 sample_type 样本列表", unknown_types))
    lines.extend(list_lines("需要人工检查的样本列表", manual_check))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Check cache similarity labels.json before official experiment.")
    parser.add_argument(
        "--labels",
        type=str,
        default="paper_repro_outputs/cache_similarity_dataset_v2_hard/labels.json",
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default="paper_repro_outputs/cache_similarity_dataset_v2_hard",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="paper_repro_outputs/cache_similarity_dataset_v2_hard/labels_check_report.md",
    )
    parser.add_argument("--fix-empty-query-text", action="store_true")
    args = parser.parse_args()

    try:
        stats = check_labels(
            labels_path=Path(args.labels),
            dataset_dir=Path(args.dataset_dir),
            output_path=Path(args.output),
            fix_query_text=args.fix_empty_query_text,
        )
    except FileNotFoundError as exc:
        print(str(exc))
        raise SystemExit(1)
    except json.JSONDecodeError as exc:
        print(f"labels.json 解析失败：{exc}")
        raise SystemExit(1)
    except Exception as exc:
        print(f"labels.json 检查失败：{exc}")
        raise SystemExit(1)

    print("=" * 72)
    print(f"labels: {args.labels}")
    print(f"dataset_dir: {args.dataset_dir}")
    print(f"report: {args.output}")
    print(f"total_samples: {stats['total_samples']}")
    print(f"empty_category_count: {stats['empty_category_count']}")
    print(f"empty_query_text_count: {stats['empty_query_text_count']}")
    print(f"missing_image_count: {stats['missing_image_count']}")
    print(f"inconsistent_should_hit_count: {stats['inconsistent_should_hit_count']}")
    print(
        "建议继续运行实验: "
        + ("是" if stats["can_run_official_experiment"] else "否")
    )
    print("=" * 72)


if __name__ == "__main__":
    main()
