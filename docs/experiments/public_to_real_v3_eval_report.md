# Public Train → Real v3 迁移评估报告

## 1. 实验目的

public_train_755 在公开数据构造样本上表现很好，但其 image_score 是类别代理分数。本实验把 public 训练模型迁移到真实 v3_real_70 分数上测试，验证是否存在 domain gap。

## 2. 训练来源

- source_dataset = open-images-v7
- total_samples = 755
- positive = 200
- near_positive = 55
- hard_negative = 300
- negative = 200
- image_score 为类别映射代理分数，不是真实图像 embedding 分数。

## 3. 测试来源

- 使用已有 v3_real_70 结果：`paper_repro_outputs\cache_similarity_eval_v3_real_70\learning_fusion_classifier\learning_fusion_dataset.csv`
- real_test_size = 70
- should_hit=True = 36
- should_hit=False = 34
- 未重新调用 Qwen / TripoSR，未重新生成模型。

## 4. 特征设置

只使用：

- text_score
- image_score
- fusion_score
- score_abs_diff
- score_max
- score_min

不使用：

- sample_type
- category
- source_dataset
- object_category

## 5. 实验结果

| method | accuracy | precision | recall | false_hit_rate | false_miss_rate | f1 | auto_hit_count | false_hit_count | false_miss_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| rule-fusion baseline | 0.8 | 1.0 | 0.6111 | 0.0 | 0.3889 | 0.7586 | 22 | 0 | 14 |
| public-trained Logistic Regression | 0.6143 | 0.5738 | 0.9722 | 0.7647 | 0.0278 | 0.7216 | 61 | 26 | 1 |
| public-trained RandomForest | 0.7571 | 0.6939 | 0.9444 | 0.4412 | 0.0556 | 0.8 | 49 | 15 | 2 |
| public-trained MLPClassifier | 0.6143 | 0.5738 | 0.9722 | 0.7647 | 0.0278 | 0.7216 | 61 | 26 | 1 |

## 6. 结果分析

1. public-trained Logistic Regression 在 real_v3 上 false_hit_rate = 0.7647，recall = 0.9722。
2. rule-fusion baseline false_hit_rate = 0.0，recall = 0.6111。
3. 如果 public-trained 模型 recall 下降或 false_hit 上升，说明公开数据类别代理分数与真实摄像头 ROI 分数之间存在 domain gap。
4. public_train 阶段的高分受到类别代理 image_score 影响，不能直接等价为真实图像相似度效果。
5. 工程接入仍应以真实 ROI 测试结果为准。

## 7. 当前结论

公开数据训练模型迁移到真实 v3_real_70 后引入 false_hit 风险，暂不适合接入工程主流程，当前仍推荐规则融合 + 双阈值策略。

recommended_for_integration = False
