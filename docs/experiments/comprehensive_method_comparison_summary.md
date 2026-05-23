# 全方法综合对比摘要

## 1. 为什么做综合对比

本轮把纯文本、纯图像、规则融合、双阈值、真实数据学习式模型和 public-trained 迁移模型统一放到 real_eval_200 上评估，核心目的是判断哪种方法最适合作为工程主流程。

## 2. real_eval_200 数据规模

- total_samples = 200
- v3_real_70 = 70
- real_add_130 = 130
- should_hit=True = 100
- should_hit=False = 100
- 学习式模型只使用线上可获得分数特征，不使用 sample_type / category / object_category。

## 3. 方法对比总表（test split）

| method | method_group | threshold | weak_threshold | strong_threshold | recall | false_hit_rate | review_rate | f1 | recommended_for_integration |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| text-only | single-score | 0.76 |  |  | 0.0 | 0.0 | 0.0 | 0.0 | False |
| image-only | single-score | 0.74 |  |  | 0.7143 | 0.0 | 0.0 | 0.8333 | False |
| rule-fusion 0.5/0.5 fixed@0.60 | rule-fusion | 0.6 |  |  | 0.7143 | 0.2727 | 0.0 | 0.5556 | False |
| rule-fusion 0.5/0.5 best-threshold | rule-fusion | 0.65 |  |  | 0.7143 | 0.0909 | 0.0 | 0.7143 | False |
| dual-threshold fusion | rule-fusion |  | 0.7 | 0.78 | 0.7143 | 0.0 | 0.0 | 0.8333 | Candidate |
| Logistic Regression | real-trained learning | 0.64 |  |  | 0.7143 | 0.0909 | 0.0 | 0.7143 | False |
| RandomForest | real-trained learning | 0.68 |  |  | 0.7143 | 0.0 | 0.0 | 0.8333 | False |
| MLPClassifier | real-trained learning | 0.73 |  |  | 0.7143 | 0.0909 | 0.0 | 0.7143 | False |
| public-trained Logistic Regression transfer | public-trained transfer | 0.5 |  |  | 1.0 | 0.9545 | 0.0 | 0.4 | False |
| public-trained RandomForest transfer | public-trained transfer | 0.5 |  |  | 0.8571 | 0.8636 | 0.0 | 0.375 | False |
| public-trained MLPClassifier transfer | public-trained transfer | 0.5 |  |  | 1.0 | 0.9545 | 0.0 | 0.4 | False |

## 4. 最终结论

当前最佳工程方案仍是规则融合 + 双阈值：

`score = 0.5 * text_score + 0.5 * image_score`

- score >= 0.78: auto_hit，自动复用缓存
- 0.70 <= score < 0.78: review，提示用户确认
- score < 0.70: miss，不复用，回退生成流程

学习式模型在 real_eval_200 上具备探索价值，但当前不建议接入 plus.py。false_hit 的风险高于 false_miss，因此工程主流程优先选择可解释、保守、false_hit_rate 为 0 的方案。
