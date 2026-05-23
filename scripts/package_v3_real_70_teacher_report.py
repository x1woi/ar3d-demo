from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import List, Tuple


DEFAULT_EVAL_DIR = Path("paper_repro_outputs/cache_similarity_eval_v3_real_70")
DEFAULT_PACKAGE_DIR = DEFAULT_EVAL_DIR / "teacher_report_package"


FILES_TO_COPY = [
    ("v3_real_70_final_report.md", "v3_real_70_final_report.md"),
    ("integration_plan_v3_real_70.md", "integration_plan_v3_real_70.md"),
    ("plus_integration_patch_plan.md", "plus_integration_patch_plan.md"),
    ("policy_simulation/policy_simulation_report.md", "policy_simulation_report.md"),
    ("policy_dry_run/dry_run_report.md", "dry_run_report.md"),
    ("review_sample_audit/review_sample_audit.md", "review_sample_audit.md"),
    ("review_sample_audit/contact_sheet_review_samples.png", "contact_sheet_review_samples.png"),
    ("borderline_threshold_analysis/borderline_threshold_report.md", "borderline_threshold_report.md"),
    ("policy_simulation/cache_policy_v3_real_70.json", "cache_policy_v3_real_70.json"),
    ("policy_simulation/policy_simulation_summary.json", "policy_simulation_summary.json"),
    ("policy_simulation/policy_simulation_results.csv", "policy_simulation_results.csv"),
]


TEACHER_SUMMARY = """# v3_real_70 阶段汇报摘要

## 1. 当前完成内容

目前已经完成：

- 70 条样本扩展；
- 图文融合缓存复用实验；
- 融合权重分析；
- 边界样本重扫；
- review 样本人工复核；
- policy loader 单元测试；
- dry-run 验证；
- plus.py 接入前方案设计。

## 2. 当前推荐策略

```text
score = 0.5 * text_score + 0.5 * image_score

score >= 0.78：自动复用缓存模型
0.7 <= score < 0.78：进入用户确认区
score < 0.7：不复用，重新生成
```

该策略目前仍为离线验证后的候选策略，尚未正式接入 plus.py。

## 3. 核心实验结果

- total_samples = 70
- recall = 0.6111
- false_hit_rate = 0.0
- false_miss_rate = 0.3889
- review_count = 10
- review_true_candidate_count = 10
- review_false_candidate_count = 0
- auto_false_hit_rate = 0.0
- saved_latency_seconds = 1154.138

70 条实验中系统仍保持误复用率为 0，说明自动复用安全性较好；相比 50 条样本，recall 有提升，说明补充 near_positive 和 hard_negative 自采集样本是有效的。

## 4. Review 区间意义

原 0.7 / 0.75 区间没有覆盖边界样本，review_rate 为 0。

经过边界样本重扫后，将 strong_threshold 调整为 0.78，可以让 10 个 should_hit=true 的边界样本进入 review 区，且没有引入 should_hit=false 的负样本。

## 5. 接入前验证

已经完成：

- cache_policy_loader.py 单元测试；
- cache_policy_dry_run.py dry-run 验证。

dry-run 结果：

- policy_score = 0.75
- policy_decision = review
- best_model_path 指向眼镜缓存模型

当前策略可以独立完成 score 计算和 auto_hit / review / miss 判断。

## 6. 当前不做的事情

- 尚未修改 plus.py；
- 尚未接入前端 review；
- 尚未调用 Qwen；
- 尚未调用 TripoSR；
- 尚未训练 MLP；
- 尚未改变主链路。

## 7. 下一步计划

1. 先给导师确认当前策略；
2. 若认可，再以 enable_policy_cache 开关形式小范围接入；
3. 第一版 review 可先记录日志或按 miss 处理；
4. 后续增加前端确认交互；
5. 继续收集真实运行日志；
6. 样本量继续扩展后再考虑轻量分类器或 MLP。
"""


README = """# v3_real_70 汇报包说明

本目录整理了 v3_real_70 阶段缓存复用实验的关键材料，便于发给导师或制作 PPT。

- v3_real_70_final_report.md：完整实验阶段报告；
- teacher_summary.md：导师简版汇报摘要；
- integration_plan_v3_real_70.md：策略接入前总体方案；
- plus_integration_patch_plan.md：plus.py 接入补丁草案；
- policy_simulation_report.md：策略离线模拟报告；
- dry_run_report.md：接入前 dry-run 验证报告；
- review_sample_audit.md：review 样本人工复核记录；
- contact_sheet_review_samples.png：review 样本图片总览；
- borderline_threshold_report.md：边界样本和阈值重扫报告；
- cache_policy_v3_real_70.json：当前候选策略配置；
- policy_simulation_summary.json：策略模拟统计；
- policy_simulation_results.csv：逐样本策略模拟结果。
"""


INTEGRATION_PLAN = """# v3_real_70 缓存复用策略接入前方案

## 1. 当前策略来源

该策略来自 v3_real_70 离线实验、边界样本重扫、10 个 review 样本人工复核和离线策略模拟。当前策略尚未接入 plus.py，只作为下一阶段小范围、可回退工程接入的候选方案。

- score = 0.5 * text_score + 0.5 * image_score
- strong_threshold = 0.78
- weak_threshold = 0.7
- auto_false_hit_rate = 0.0
- review_false_candidate_count = 0
- review_count = 10

## 2. 策略配置化设计

后续不应把阈值直接硬编码进 plus.py，而应通过配置文件读取策略参数。

当前已有策略配置文件：

```text
paper_repro_outputs/cache_similarity_eval_v3_real_70/policy_simulation/cache_policy_v3_real_70.json
```

建议未来复制一份到运行时配置位置：

```text
runtime_assets/cache_policy.json
```

本阶段不执行复制，也不修改运行时代码。

## 3. 未来接入位置

未来可以在 plus.py 当前融合缓存判断阶段接入策略判断。

```text
关键词缓存命中
→ 融合相似度计算
→ policy decision
→ auto_hit / review / miss
```

## 4. 三种决策行为

- auto_hit：score >= 0.78，直接复用缓存模型。
- review：0.7 <= score < 0.78，前端提示用户确认是否复用。
- miss：score < 0.7，不复用缓存，继续原生成流程。

## 5. 风险控制

1. 策略必须可关闭，例如 enable_policy_cache=false。
2. 异常时必须回退原流程。
3. 找不到 best_model_path 时不能复用。
4. review 区不能自动复用。
5. 日志必须记录 score、decision、threshold、model_path。
6. 不影响原关键词缓存逻辑。

## 6. 后续实施步骤

1. 新增独立 policy loader；
2. 新增离线单元测试；
3. 在 plus.py 中以开关形式轻量接入；
4. review 区先只返回提示，不自动复用；
5. 小样本真实运行测试；
6. 收集日志后再决定是否扩大使用。
"""


REVIEW_AUDIT = """# v3_real_70 Review 样本人工复核表

## 1. 复核目的

当前推荐 review 区间为 0.7 <= score < 0.78，需要人工确认这些边界样本是否适合提示用户复用。

## 2. 当前统计

- review_count = 10
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

如果 10 个样本都合理，可以把 0.7 / 0.78 作为下一阶段候选双阈值策略。

如果存在明显不合理样本，需要先修正标签或重新设定 review 区间。

## 5. 人工复核结论

本次 10 个 review 样本均为 near_positive 眼镜样本。人工复核后认为，这些样本均适合保留在 review 区。前 1～6 张主体较清晰，后 7～10 张存在遮挡、角度变化或局部裁剪，但仍能看出与眼镜缓存模型相关。因此当前 weak=0.7、strong=0.78 的 review 区间具有实际意义，可以作为下一阶段候选策略。
"""


def copy_files(eval_dir: Path, package_dir: Path) -> Tuple[int, List[str]]:
    package_dir.mkdir(parents=True, exist_ok=True)
    copied = 0
    missing: List[str] = []
    for source_rel, target_name in FILES_TO_COPY:
        source = eval_dir / source_rel
        target = package_dir / target_name
        if not source.exists():
            missing.append(str(source))
            continue
        shutil.copy2(source, target)
        copied += 1
    return copied, missing


def has_garbled_text(path: Path) -> bool:
    if not path.exists() or path.suffix.lower() not in {".md", ".txt"}:
        return False
    text = path.read_text(encoding="utf-8", errors="replace")
    return "????" in text or "\ufffd" in text


def package(eval_dir: Path, package_dir: Path) -> dict:
    copied, missing = copy_files(eval_dir, package_dir)
    teacher_summary_path = package_dir / "teacher_summary.md"
    readme_path = package_dir / "README.md"
    teacher_summary_path.write_text(TEACHER_SUMMARY, encoding="utf-8")
    readme_path.write_text(README, encoding="utf-8")
    (package_dir / "integration_plan_v3_real_70.md").write_text(INTEGRATION_PLAN, encoding="utf-8")
    (package_dir / "review_sample_audit.md").write_text(REVIEW_AUDIT, encoding="utf-8")

    garbled_detected = any(has_garbled_text(path) for path in package_dir.iterdir() if path.is_file())
    warning_path = package_dir / "package_warnings.txt"
    if missing:
        warning_path.write_text("\n".join(missing) + "\n", encoding="utf-8")

    return {
        "package_dir": package_dir,
        "copied_file_count": copied,
        "missing_file_count": len(missing),
        "teacher_summary_path": teacher_summary_path,
        "readme_path": readme_path,
        "garbled_detected": garbled_detected,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Package v3_real_70 teacher report materials.")
    parser.add_argument("--eval-dir", default=str(DEFAULT_EVAL_DIR))
    parser.add_argument("--package-dir", default=str(DEFAULT_PACKAGE_DIR))
    args = parser.parse_args()

    result = package(Path(args.eval_dir), Path(args.package_dir))
    print("=" * 72)
    print(f"teacher_report_package: {result['package_dir']}")
    print(f"copied_file_count: {result['copied_file_count']}")
    print(f"missing_file_count: {result['missing_file_count']}")
    print(f"teacher_summary.md: {result['teacher_summary_path']}")
    print(f"README.md: {result['readme_path']}")
    print(f"garbled_detected: {result['garbled_detected']}")
    print("=" * 72)


if __name__ == "__main__":
    main()
