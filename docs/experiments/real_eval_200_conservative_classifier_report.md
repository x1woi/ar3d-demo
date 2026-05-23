# real_eval_200 保守型学习式融合模型报告

## 1. 实验目的

本实验将已有 v3_real_70 与新增 real_add_130 真实 ROI 特征合并为 real_eval_200，训练保守型学习式图文融合分类器。目标不是单纯提高 recall，而是在 false_hit_rate 接近 0 的前提下，观察学习式模型是否比规则融合 + 双阈值更适合工程接入。

## 2. 数据组成

- total_samples: 200
- source_counts: {'v3_real_70': 70, 'real_add_130': 130}
- should_hit_counts: {1: 100, 0: 100}
- split: train=129, val=42, test=29
- split_note: group-aware split by source_video/group_id; sizes may differ slightly from 140/30/30

## 3. 特征设置

只使用线上可获得的分数特征：

- text_score
- image_score
- fusion_score
- score_abs_diff
- score_max
- score_min

不使用 sample_type / category / object_category 作为模型输入，避免标签泄漏或分布记忆。

## 4. 测试集结果

| method | threshold | recall | false_hit_rate | review_rate | false_miss_rate | f1 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| rule-fusion dual-threshold | weak=0.7, strong=0.78 | 0.7143 | 0.0 | 0.0 | 0.2857 | 0.8333 |
| Logistic Regression | 0.64 | 0.7143 | 0.0909 | 0.1379 | 0.2857 | 0.7143 |
| RandomForest | 0.68 | 0.7143 | 0.0 | 0.2759 | 0.2857 | 0.8333 |
| MLPClassifier | 0.73 | 0.7143 | 0.0909 | 0.1724 | 0.2857 | 0.7143 |

## 5. 必须回答的问题

1. real_eval_200 上学习式模型是否能保持 false_hit_rate 接近 0？
   - 见测试集 `false_hit_rate`。模型选择以 false_hit_rate 为第一优先级。

2. 是否比规则融合 + 双阈值更好？
   - 需要同时满足更低或接近 0 的 false_hit_rate，以及更高 recall。不能只看 recall。

3. 是否建议接入 plus.py？
   - recommended_for_integration = False
   - 当前仍不建议直接接入 plus.py。

4. 是否仍需要保留 review 分支？
   - keep_review = True
   - review 分支仍建议保留，用于处理边界样本并降低误复用风险。

5. 当前最佳工程方案是什么？
   - 当前最佳工程方案仍是规则融合 + 双阈值策略；学习式模型暂不接入 plus.py。
   - 当前候选工程策略仍是：`score = 0.5 * text_score + 0.5 * image_score`，`score >= 0.78` 自动复用，`0.70 <= score < 0.78` 进入 review，`score < 0.70` 不复用。

## 6. 当前结论

本报告用于判断学习式融合是否有工程接入价值。若学习式模型在测试集上出现 false_hit，则不应接入主流程；若保持 false_hit_rate 接近 0 且 recall 明显提升，也仍需更多真实独立测试集验证后再考虑接入。
