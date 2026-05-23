# real_eval_200 最终阈值 Pareto 调整报告

## 1. 实验目的

前一轮找到了 false_hit_rate=0 的候选阈值，但为了避免只呈现单个过度保守点，本轮进一步扫描不同风险约束下的候选工作点，用于论文中展示阈值选择依据。

## 2. 数据集

- dataset: real_eval_200
- total_samples = 200
- should_hit=True = 100
- should_hit=False = 100

## 3. 决策规则

```text
score = 0.5 * text_score + 0.5 * image_score

score >= strong_threshold:
  高置信自动复用

weak_threshold <= score < strong_threshold:
  低置信候选区

score < weak_threshold:
  不复用
```

## 4. 候选工作点

| name | status | weak_threshold | strong_threshold | recall | false_hit_rate | low_confidence_candidate_rate | auto_hit_count | low_confidence_candidate_count | miss_count | false_hit_count | false_miss_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| conservative_safe | ok | 0.6 | 0.74 | 0.62 | 0.0 | 0.28 | 29 | 56 | 115 | 0 | 38 |
| balanced_low_risk | ok | 0.6 | 0.69 | 0.62 | 0.03 | 0.225 | 40 | 45 | 115 | 3 | 38 |
| balanced_practical | ok | 0.62 | 0.68 | 0.58 | 0.04 | 0.155 | 43 | 31 | 126 | 4 | 42 |
| aggressive_recall | ok | 0.6 | 0.65 | 0.62 | 0.09 | 0.145 | 56 | 29 | 115 | 9 | 38 |
| old_baseline | ok | 0.7 | 0.78 | 0.35 | 0.0 | 0.11 | 16 | 22 | 162 | 0 | 65 |
| current_candidate | ok | 0.6 | 0.74 | 0.62 | 0.0 | 0.28 | 29 | 56 | 115 | 0 | 38 |

## 5. 结果分析

1. false_hit_rate=0 的点并不一定过度保守，但会受到低置信候选区比例影响。`conservative_safe` 在自动误复用为 0 的前提下控制低置信候选区不超过 0.30。

2. 如果允许 false_hit_rate <= 0.03 或 <= 0.05，是否能明显提高 recall 需要看 balanced_low_risk / balanced_practical。若 recall 提升有限，则不值得为了微小召回提升承担误复用风险。

3. low_confidence_candidate_rate 是工程可用性的重要指标。候选区比例过高会增加后续交互或处理负担，因此本报告不只追求 recall。

4. 论文主方法建议展示 conservative_safe，并可附带 balanced_low_risk / balanced_practical 作为风险-召回折中对比。

5. 工程保守默认值应优先选择 false_hit_rate=0 且低置信候选区受控的工作点。

6. aggressive_recall 只能作为消融对照，不建议接入。

## 6. 最终建议

- paper_main_threshold: {'name': 'conservative_safe', 'weak_threshold': 0.6, 'strong_threshold': 0.74, 'reason': '论文主表可展示该折中点，同时报告误复用风险与低置信候选区比例。'}
- runtime_safe_threshold: {'name': 'conservative_safe', 'weak_threshold': 0.6, 'strong_threshold': 0.74, 'reason': '工程默认优先选择自动误复用为 0 且低置信候选区受控的保守点。'}
- ablation_aggressive_threshold: {'name': 'aggressive_recall', 'weak_threshold': 0.6, 'strong_threshold': 0.65, 'reason': '激进设置仅适合作为消融对照，不建议接入。'}
- recommended_for_plus_py: False

如果 balanced_low_risk / balanced_practical 的 false_hit_rate 虽然非 0 但很低，并且 recall 明显高于 conservative_safe，可以在论文主表同时展示二者。若非 0 false_hit 的候选提升不明显，则继续推荐 conservative_safe 或 current_candidate。

## 7. 论文表述建议

不要写“系统不会误复用”。建议写：

在当前 real_eval_200 测试集上，某阈值组合未观察到自动误复用样本；在允许极低误复用风险的设置下，召回率可以进一步变化。本文最终选择某阈值，是在误复用风险、候选区比例和召回率之间的折中。

## 8. 输出图表

- final_threshold_pareto_plot.png generated: True
- final_threshold_bar_comparison.png generated: True
