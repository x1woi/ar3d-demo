from __future__ import annotations

import argparse
from pathlib import Path


README_TEXT = {
    "positive": (
        "放应该命中已有缓存的样本，例如同一个目标的不同角度、不同光照、不同裁剪。\n\n"
        "建议数量：12 张。\n"
    ),
    "near_positive": (
        "放同类但不完全相同、可以考虑复用缓存的样本，例如另一副眼镜、相似人头、类似物体。\n\n"
        "建议数量：12 张。\n"
    ),
    "hard_negative": (
        "放外观相似但不应该复用的样本，例如眼镜盒、手拿眼镜但主体是手、圆形物体、纸上图案、"
        "非目标 mask、遮挡严重区域。\n\n"
        "建议数量：12 张。\n"
    ),
    "negative": (
        "放明显不应该命中的样本，例如杯子、书、桌面、键盘、背景等。\n\n"
        "建议数量：14 张。\n"
    ),
}


def prepare_dirs(root_dir: Path) -> None:
    for sample_type, readme in README_TEXT.items():
        target = root_dir / sample_type
        target.mkdir(parents=True, exist_ok=True)
        (target / "README.txt").write_text(readme, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare cache similarity v3_real dataset folders.")
    parser.add_argument(
        "--root-dir",
        default="cache_test_v3_real",
        help="Root directory for positive/near_positive/hard_negative/negative folders.",
    )
    args = parser.parse_args()

    root_dir = Path(args.root_dir)
    prepare_dirs(root_dir)

    dataset_dir = Path("paper_repro_outputs/cache_similarity_dataset_v3_real")
    eval_dir = Path("paper_repro_outputs/cache_similarity_eval_v3_real")

    print("=" * 72)
    print("Cache v3_real sample folders prepared")
    print(f"v3_real 样本目录路径: {root_dir}")
    for sample_type in README_TEXT:
        print(f"- {root_dir / sample_type}")
    print("")
    print("建议样本数量：")
    print("- positive: 12")
    print("- near_positive: 12")
    print("- hard_negative: 12")
    print("- negative: 14")
    print("- total: 约 50")
    print("")
    print(f"v3_real 数据集输出路径: {dataset_dir}")
    print(f"v3_real 实验输出路径: {eval_dir}")
    print("建议下一步：用户向四类目录补充图片。")
    print("=" * 72)


if __name__ == "__main__":
    main()
