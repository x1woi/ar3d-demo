from __future__ import annotations

from pathlib import Path


ROOT = Path("v3_self_capture")
SUBDIRS = ["positive", "near_positive", "hard_negative", "negative", "videos"]


README_TEXT = """# v3_real 自采集样本准备目录

这个目录用于暂存自己采集的真实 ROI / 短视频样本。请先把图片或视频放在这里，人工确认后再分拣到 `cache_test_v3_real/`，不要直接覆盖正式实验目录。

## 1. 采集目标

### positive

同一缓存目标的不同角度、不同距离、不同光照，例如同一个人头、同一副眼镜。

### near_positive

同类但不完全相同的目标，例如不同角度人脸、半遮挡人脸、另一副眼镜、不同角度眼镜。

### hard_negative

看起来相似但不应复用的区域，例如手、遮挡区域、眼镜盒、圆形物体、纸上图案、像人脸但不是人脸的区域、像眼镜但不是眼镜的物体。

### negative

明显无关样本，例如键盘、桌面、杯子、书、背景。

### videos

存放短视频，后续可抽帧生成候选图。建议每类录 10-20 秒，之后人工挑选质量较好的帧。

## 2. 下一轮建议数量

- positive：+5
- near_positive：+10
- hard_negative：+10
- negative：+5

优先补充 near_positive 和 hard_negative，因为它们最能验证 recall、false_hit_rate 和 review 区间是否有效。

## 3. 推荐命名规则

建议按目标和类别命名，方便后续自动补充 category / query_text：

```text
head_positive_001.jpg
head_near_001.jpg
glasses_positive_001.jpg
glasses_near_001.jpg
hand_hard_001.jpg
mask_hard_001.jpg
keyboard_negative_001.jpg
cup_negative_001.jpg
```

## 4. 注意事项

- 不要直接覆盖 `cache_test_v3_real/`。
- 先放入 `v3_self_capture/`。
- 人工确认后再分拣到 `cache_test_v3_real/`。
- 人脸样本建议使用本人或已同意对象。
- 不拍无关路人。
- 不修改 `plus.py`。
- 不调用 Qwen / TripoSR。
- 不训练 MLP。
"""


def prepare_dirs(root: Path = ROOT) -> tuple[Path, Path]:
    root.mkdir(parents=True, exist_ok=True)
    for subdir in SUBDIRS:
        (root / subdir).mkdir(parents=True, exist_ok=True)

    readme_path = root / "README.md"
    readme_path.write_text(README_TEXT, encoding="utf-8")
    return root, readme_path


def main() -> None:
    root, readme_path = prepare_dirs()
    print("=" * 72)
    print("v3_real self-capture directories prepared")
    print(f"v3_self_capture 目录路径: {root.resolve()}")
    print(f"README.md 路径: {readme_path.resolve()}")
    print("下一步提示: 请拍摄图片或短视频后放入 v3_self_capture，再进行抽帧/人工分拣。")
    print("=" * 72)


if __name__ == "__main__":
    main()
