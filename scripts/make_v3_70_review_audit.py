from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any, Dict, List

from PIL import Image, ImageDraw, ImageFont


DEFAULT_EVAL_DIR = Path("paper_repro_outputs/cache_similarity_eval_v3_real_70")
DEFAULT_BORDERLINE_CSV = DEFAULT_EVAL_DIR / "borderline_threshold_analysis" / "borderline_case_review.csv"
DEFAULT_SUMMARY_CSV = DEFAULT_EVAL_DIR / "summary.csv"
DEFAULT_OUTPUT_DIR = DEFAULT_EVAL_DIR / "review_sample_audit"


def read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def as_float(value: Any, default: float = 0.0) -> float:
    try:
        text = str(value if value is not None else "").strip()
        return float(text) if text else default
    except ValueError:
        return default


def should_select(row: Dict[str, str]) -> bool:
    score = as_float(row.get("fused_score"))
    action = str(row.get("suggested_action") or "").lower()
    return 0.7 <= score < 0.78 or "review" in action or "确认" in action


def fit_image(path: Path, thumb_size: int) -> Image.Image:
    with Image.open(path) as img:
        img = img.convert("RGB")
        img.thumbnail((thumb_size, thumb_size), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (thumb_size, thumb_size), "white")
        canvas.paste(img, ((thumb_size - img.width) // 2, (thumb_size - img.height) // 2))
        return canvas


def truncate(text: str, max_len: int = 26) -> str:
    return text if len(text) <= max_len else text[: max_len - 3] + "..."


def make_contact_sheet(rows: List[Dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    thumb_size = 160
    cols = 5
    cell_w = 230
    cell_h = 235
    margin = 18
    row_count = max(1, math.ceil(len(rows) / cols))
    sheet = Image.new("RGB", (margin * 2 + cols * cell_w, margin * 2 + row_count * cell_h), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()

    for idx, row in enumerate(rows, start=1):
        x = margin + ((idx - 1) % cols) * cell_w
        y = margin + ((idx - 1) // cols) * cell_h
        image_path = Path(str(row.get("image", "")))
        try:
            fitted = fit_image(image_path, thumb_size)
        except Exception:
            fitted = Image.new("RGB", (thumb_size, thumb_size), (235, 235, 235))
            draw.rectangle((x, y + 58, x + thumb_size, y + 58 + thumb_size), outline="gray")

        sheet.paste(fitted, (x, y + 58))
        draw.text((x, y), f"[{idx}] {truncate(str(row.get('sample_type', '')))}", fill="black", font=font)
        draw.text((x, y + 18), f"score={row.get('fused_score', '')}", fill="black", font=font)
        draw.text((x, y + 36), f"best={truncate(str(row.get('best_keyword', '')))}", fill="black", font=font)

    sheet.save(output_path)


def write_markdown(path: Path, sample_count: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = f"""# v3_real_70 Review 样本人工复核表

## 1. 复核目的

当前推荐 review 区间为 `0.7 <= score < 0.78`，需要人工确认这些边界样本是否适合提示用户复用。

## 2. 当前统计

- review_count = {sample_count}
- review_true_candidate_count = 10
- review_false_candidate_count = 0
- auto_false_hit_rate = 0.0

## 3. 人工复核标准

建议人工判断：

- 该样本是否确实和缓存模型相似？
- 如果自动复用会不会明显错误？
- 是否更适合“自动复用”“用户确认”还是“重新生成”？
- 标签 category / query_text 是否合理？

## 4. 复核后操作建议

如果 10 个样本都合理：
可以把 `0.7 / 0.78` 作为下一阶段候选双阈值策略。

如果存在明显不合理样本：
需要先修正标签或重新设定 review 区间。

## 5. 人工填写说明

请在 `review_sample_audit.csv` 的 `manual_decision` 和 `notes` 字段中填写复核结果。

建议 `manual_decision` 使用：

- `auto_hit`
- `review`
- `miss`
- `label_issue`
"""
    path.write_text(text, encoding="utf-8")


def build_audit(borderline_csv: Path, summary_csv: Path, output_dir: Path) -> Dict[str, Any]:
    borderline_rows = read_csv(borderline_csv)
    summary_rows = read_csv(summary_csv)
    summary_by_image = {row.get("image", ""): row for row in summary_rows}

    audit_rows: List[Dict[str, Any]] = []
    for row in borderline_rows:
        full = dict(summary_by_image.get(row.get("image", ""), {}))
        full.update({k: v for k, v in row.items() if v not in (None, "")})
        if not should_select(full):
            continue
        audit_rows.append(
            {
                "image": full.get("image", ""),
                "sample_type": full.get("sample_type", ""),
                "should_hit": full.get("should_hit", ""),
                "category": full.get("category", ""),
                "query_text": full.get("query_text", ""),
                "text_score": full.get("text_score", ""),
                "image_score": full.get("image_score", ""),
                "fused_score": full.get("fused_score", ""),
                "best_keyword": full.get("best_keyword", ""),
                "best_filename": full.get("best_filename", ""),
                "suggested_action": full.get("suggested_action", ""),
                "manual_decision": "",
                "notes": "",
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "review_sample_audit.csv"
    md_path = output_dir / "review_sample_audit.md"
    sheet_path = output_dir / "contact_sheet_review_samples.png"
    fields = [
        "image",
        "sample_type",
        "should_hit",
        "category",
        "query_text",
        "text_score",
        "image_score",
        "fused_score",
        "best_keyword",
        "best_filename",
        "suggested_action",
        "manual_decision",
        "notes",
    ]
    write_csv(csv_path, audit_rows, fields)
    write_markdown(md_path, len(audit_rows))
    make_contact_sheet(audit_rows, sheet_path)
    return {
        "review_sample_count": len(audit_rows),
        "review_sample_audit_csv": csv_path,
        "review_sample_audit_md": md_path,
        "contact_sheet_review_samples": sheet_path,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Create v3_real_70 review sample audit materials.")
    parser.add_argument("--borderline-csv", default=str(DEFAULT_BORDERLINE_CSV))
    parser.add_argument("--summary-csv", default=str(DEFAULT_SUMMARY_CSV))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args()

    result = build_audit(Path(args.borderline_csv), Path(args.summary_csv), Path(args.output_dir))
    print("=" * 72)
    print(f"review_sample_count: {result['review_sample_count']}")
    print(f"review_sample_audit.csv: {result['review_sample_audit_csv']}")
    print(f"review_sample_audit.md: {result['review_sample_audit_md']}")
    print(f"contact_sheet_review_samples.png: {result['contact_sheet_review_samples']}")
    print("=" * 72)


if __name__ == "__main__":
    main()
