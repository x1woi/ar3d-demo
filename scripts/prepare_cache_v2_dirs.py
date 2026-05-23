from __future__ import annotations

import argparse
from pathlib import Path


README_TEXT = {
    "positive": "放应该命中缓存的图片，例如同一副眼镜的不同角度、不同光照、不同裁剪。\n",
    "near_positive": "放同类但不完全相同、可以接受复用缓存的图片，例如另一副眼镜、相似人头图。\n",
    "hard_negative": "放外观相似但不应该命中缓存的图片，例如眼镜盒、手拿眼镜但主体是手、圆形物体、纸上图案等。\n",
    "negative": "放明显不应该命中缓存的图片，例如杯子、书、桌面、键盘、背景、手等。\n",
}


def prepare_dirs(root_dir: Path) -> None:
    root_dir = Path(root_dir)
    for name, readme in README_TEXT.items():
        target = root_dir / name
        target.mkdir(parents=True, exist_ok=True)
        (target / "README.txt").write_text(readme, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare cache similarity v2 hard dataset folders."
    )
    parser.add_argument(
        "--root-dir",
        type=str,
        default="cache_test_v2",
        help="Root directory for positive/near_positive/hard_negative/negative folders.",
    )
    args = parser.parse_args()

    prepare_dirs(Path(args.root_dir))

    print("=" * 72)
    print("Cache v2 hard dataset folders prepared")
    print(f"Root dir: {args.root_dir}")
    for name in README_TEXT:
        print(f"- {Path(args.root_dir) / name}")
    print("=" * 72)


if __name__ == "__main__":
    main()
