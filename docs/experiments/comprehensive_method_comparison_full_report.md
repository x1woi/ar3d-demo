# 全方法综合对比实验报告

## 1. 实验目的

本实验统一比较项目中已经尝试过的所有缓存复用判断方法，包括纯文本、纯图像、规则融合、双阈值融合、学习式融合、公开数据训练模型和真实数据训练模型，目标是判断哪种方法最适合作为工程主流程。

## 2. 数据集说明

real_eval_200 由 v3_real_70 和 real_add_130 合并而成：

- total_samples = 200
- should_hit=True = 100
- should_hit=False = 100
- 使用已有真实 text_score / image_score / fusion_score
- 不使用 sample_type / category / object_category 作为学习式模型输入
- 不调用 Qwen，不调用 TripoSR，不重新生成模型

## 3. 方法列表

- text-only：只使用 text_score，阈值在 val 上扫描。
- image-only：只使用 image_score，阈值在 val 上扫描。
- rule-fusion 0.5 / 0.5：包含 fixed@0.60 和 val 最佳阈值两个结果。
- dual-threshold fusion：weak=0.70，strong=0.78，支持 auto_hit / review / miss。
- Logistic Regression / RandomForest / MLPClassifier：只使用线上分数特征，在 train split 训练，val split 选保守阈值。
- public-trained transfer models：直接加载 public_train_755 训练模型做迁移预测，不在 real_eval_200 上重训。

## 4. real_eval_200 统一结果

| method | method_group | threshold | weak_threshold | strong_threshold | accuracy | precision | recall | false_hit_rate | false_miss_rate | review_rate | f1 | recommended_for_integration |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| text-only | single-score | 0.76 |  |  | 0.7586 | 0.0 | 0.0 | 0.0 | 1.0 | 0.0 | 0.0 | False |
| image-only | single-score | 0.74 |  |  | 0.931 | 1.0 | 0.7143 | 0.0 | 0.2857 | 0.0 | 0.8333 | False |
| rule-fusion 0.5/0.5 fixed@0.60 | rule-fusion | 0.6 |  |  | 0.7241 | 0.4545 | 0.7143 | 0.2727 | 0.2857 | 0.0 | 0.5556 | False |
| rule-fusion 0.5/0.5 best-threshold | rule-fusion | 0.65 |  |  | 0.8621 | 0.7143 | 0.7143 | 0.0909 | 0.2857 | 0.0 | 0.7143 | False |
| dual-threshold fusion | rule-fusion |  | 0.7 | 0.78 | 0.931 | 1.0 | 0.7143 | 0.0 | 0.2857 | 0.0 | 0.8333 | Candidate |
| Logistic Regression | real-trained learning | 0.64 |  |  | 0.8621 | 0.7143 | 0.7143 | 0.0909 | 0.2857 | 0.0 | 0.7143 | False |
| RandomForest | real-trained learning | 0.68 |  |  | 0.931 | 1.0 | 0.7143 | 0.0 | 0.2857 | 0.0 | 0.8333 | False |
| MLPClassifier | real-trained learning | 0.73 |  |  | 0.8621 | 0.7143 | 0.7143 | 0.0909 | 0.2857 | 0.0 | 0.7143 | False |
| public-trained Logistic Regression transfer | public-trained transfer | 0.5 |  |  | 0.2759 | 0.25 | 1.0 | 0.9545 | 0.0 | 0.0 | 0.4 | False |
| public-trained RandomForest transfer | public-trained transfer | 0.5 |  |  | 0.3103 | 0.24 | 0.8571 | 0.8636 | 0.1429 | 0.0 | 0.375 | False |
| public-trained MLPClassifier transfer | public-trained transfer | 0.5 |  |  | 0.2759 | 0.25 | 1.0 | 0.9545 | 0.0 | 0.0 | 0.4 | False |

跳过项：

无

## 5. 历史实验对比

| stage | dataset | method | total_samples | feature_setting | recall | false_hit_rate | false_miss_rate | review_rate | recommended_for_integration | main_conclusion |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A. v3_real_70 规则策略 | v3_real_70 | text-only | 70 | text/image/fusion score comparison | 0.6111 | 0.0 | 0.3889 | 0 | rule baseline only | v3_real_70 showed rule fusion was safer than chasing recall alone. |
| A. v3_real_70 规则策略 | v3_real_70 | image-only | 70 | text/image/fusion score comparison | 0.3333 | 0.0294 | 0.6667 | 0 | rule baseline only | v3_real_70 showed rule fusion was safer than chasing recall alone. |
| A. v3_real_70 规则策略 | v3_real_70 | rule-fusion | 70 | text/image/fusion score comparison | 0.6111 | 0.0 | 0.3889 | 0 | rule baseline only | v3_real_70 showed rule fusion was safer than chasing recall alone. |
| A. v3_real_70 规则策略 | v3_real_70 | dual-threshold fusion | 70 | text/image/fusion score comparison | 0.6111 | 0.0 | 0.3889 | 10 | rule baseline only | v3_real_70 showed rule fusion was safer than chasing recall alone. |
| B. v3_real_70 学习式融合初步实验 | v3_real_70 | Logistic Regression | 70 | included sample_type/category metadata; leakage risk | 1.0 | 0.0 | 0.0 |  | False | High scores may be affected by metadata leakage/distribution memory. |
| B. v3_real_70 学习式融合初步实验 | v3_real_70 | RandomForest | 70 | included sample_type/category metadata; leakage risk | 0.8611 | 0.0 | 0.1389 |  | False | High scores may be affected by metadata leakage/distribution memory. |
| B. v3_real_70 学习式融合初步实验 | v3_real_70 | MLPClassifier | 70 | included sample_type/category metadata; leakage risk | 1.0 | 0.0 | 0.0 |  | False | High scores may be affected by metadata leakage/distribution memory. |
| C. 去元数据消融实验 | v3_real_70 | Logistic Regression | 70 | score-only features, no sample_type/category | 0.6667 | 0.0882 | 0.3333 |  | False | Removing metadata reduced reliability and introduced false_hit risk. |
| C. 去元数据消融实验 | v3_real_70 | RandomForest | 70 | score-only features, no sample_type/category | 0.7778 | 0.1176 | 0.2222 |  | False | Removing metadata reduced reliability and introduced false_hit risk. |
| C. 去元数据消融实验 | v3_real_70 | MLPClassifier | 70 | score-only features, no sample_type/category | 0.7778 | 0.1176 | 0.2222 |  | False | Removing metadata reduced reliability and introduced false_hit risk. |
| D. public_train_755 | public_train_755 | rule-fusion baseline | 755 | text string score + category-proxy image_score | 0.7949 | 0.0 | 0.2051 |  | False | Public constructed samples looked strong, but image_score was a category proxy. |
| D. public_train_755 | public_train_755 | Logistic Regression | 755 | text string score + category-proxy image_score | 1.0 | 0.0 | 0.0 |  | False | Public constructed samples looked strong, but image_score was a category proxy. |
| D. public_train_755 | public_train_755 | RandomForest | 755 | text string score + category-proxy image_score | 1.0 | 0.0 | 0.0 |  | False | Public constructed samples looked strong, but image_score was a category proxy. |
| D. public_train_755 | public_train_755 | MLPClassifier | 755 | text string score + category-proxy image_score | 1.0 | 0.0 | 0.0 |  | False | Public constructed samples looked strong, but image_score was a category proxy. |
| E. public -> real_v3 迁移 | real_v3 | rule-fusion baseline | 70 | public-trained model transferred to real ROI scores | 0.6111 | 0.0 | 0.3889 |  | False | Public-trained models introduced high false_hit on real ROI, showing domain gap. |
| E. public -> real_v3 迁移 | real_v3 | public-trained Logistic Regression | 70 | public-trained model transferred to real ROI scores | 0.9722 | 0.7647 | 0.0278 |  | False | Public-trained models introduced high false_hit on real ROI, showing domain gap. |
| E. public -> real_v3 迁移 | real_v3 | public-trained RandomForest | 70 | public-trained model transferred to real ROI scores | 0.9444 | 0.4412 | 0.0556 |  | False | Public-trained models introduced high false_hit on real ROI, showing domain gap. |
| E. public -> real_v3 迁移 | real_v3 | public-trained MLPClassifier | 70 | public-trained model transferred to real ROI scores | 0.9722 | 0.7647 | 0.0278 |  | False | Public-trained models introduced high false_hit on real ROI, showing domain gap. |
| F. real_eval_200 | real_eval_200 | rule-fusion dual-threshold | 200 | score-only features, group-aware split where possible | 0.7143 | 0.0 | 0.2857 | 0.0 | False | Current best engineering choice remained rule fusion + dual threshold. |
| F. real_eval_200 | real_eval_200 | Logistic Regression | 200 | score-only features, group-aware split where possible | 0.7143 | 0.0909 | 0.2857 | 0.1379 | False | Current best engineering choice remained rule fusion + dual threshold. |
| F. real_eval_200 | real_eval_200 | RandomForest | 200 | score-only features, group-aware split where possible | 0.7143 | 0.0 | 0.2857 | 0.2759 | False | Current best engineering choice remained rule fusion + dual threshold. |
| F. real_eval_200 | real_eval_200 | MLPClassifier | 200 | score-only features, group-aware split where possible | 0.7143 | 0.0909 | 0.2857 | 0.1724 | False | Current best engineering choice remained rule fusion + dual threshold. |
| G. real_eval_200 全方法综合对比 | real_eval_200 | text-only | 200 | single-score | 0.0 | 0.0 | 1.0 | 0.0 | False | Unified comparison on the latest real_eval_200 feature table. |
| G. real_eval_200 全方法综合对比 | real_eval_200 | image-only | 200 | single-score | 0.7143 | 0.0 | 0.2857 | 0.0 | False | Unified comparison on the latest real_eval_200 feature table. |
| G. real_eval_200 全方法综合对比 | real_eval_200 | rule-fusion 0.5/0.5 fixed@0.60 | 200 | rule-fusion | 0.7143 | 0.2727 | 0.2857 | 0.0 | False | Unified comparison on the latest real_eval_200 feature table. |
| G. real_eval_200 全方法综合对比 | real_eval_200 | rule-fusion 0.5/0.5 best-threshold | 200 | rule-fusion | 0.7143 | 0.0909 | 0.2857 | 0.0 | False | Unified comparison on the latest real_eval_200 feature table. |
| G. real_eval_200 全方法综合对比 | real_eval_200 | dual-threshold fusion | 200 | rule-fusion | 0.7143 | 0.0 | 0.2857 | 0.0 | Candidate | Unified comparison on the latest real_eval_200 feature table. |
| G. real_eval_200 全方法综合对比 | real_eval_200 | Logistic Regression | 200 | real-trained learning | 0.7143 | 0.0909 | 0.2857 | 0.0 | False | Unified comparison on the latest real_eval_200 feature table. |
| G. real_eval_200 全方法综合对比 | real_eval_200 | RandomForest | 200 | real-trained learning | 0.7143 | 0.0 | 0.2857 | 0.0 | False | Unified comparison on the latest real_eval_200 feature table. |
| G. real_eval_200 全方法综合对比 | real_eval_200 | MLPClassifier | 200 | real-trained learning | 0.7143 | 0.0909 | 0.2857 | 0.0 | False | Unified comparison on the latest real_eval_200 feature table. |
| G. real_eval_200 全方法综合对比 | real_eval_200 | public-trained Logistic Regression transfer | 200 | public-trained transfer | 1.0 | 0.9545 | 0.0 | 0.0 | False | Unified comparison on the latest real_eval_200 feature table. |
| G. real_eval_200 全方法综合对比 | real_eval_200 | public-trained RandomForest transfer | 200 | public-trained transfer | 0.8571 | 0.8636 | 0.1429 | 0.0 | False | Unified comparison on the latest real_eval_200 feature table. |
| G. real_eval_200 全方法综合对比 | real_eval_200 | public-trained MLPClassifier transfer | 200 | public-trained transfer | 1.0 | 0.9545 | 0.0 | 0.0 | False | Unified comparison on the latest real_eval_200 feature table. |

## 6. 结果分析

1. 不能只看 recall。缓存复用任务中，高 recall 如果伴随 false_hit，会把错误缓存模型自动复用给用户，风险高于漏命中。

2. false_hit 比 false_miss 风险更高。false_miss 最多回退到生成流程，代价是慢；false_hit 会错误复用模型，直接影响系统可信度和 AR 展示结果。

3. public_train 上的高分不能直接说明真实系统有效。public_train_755 的 image_score 是类别映射代理分数，不是真实摄像头 ROI embedding / image signature 分数，因此迁移到 real_v3 后出现高 false_hit，说明存在 domain gap。

4. 去元数据消融很重要。早期学习式实验使用 sample_type / category 后分数过高，可能记住标签分布；去掉元数据后 false_hit 风险暴露出来，更接近真实线上条件。

5. real_eval_200 后仍推荐规则融合 + 双阈值。它可解释、无需训练、false_hit_rate 保守，并且已完成工程接入与三分支验证。相比之下，学习式模型仍需要更大的独立真实测试集。

6. RandomForest 在 real_eval_200 上虽然可以做到较低 false_hit，但仍不直接接入。原因是当前测试集仍较小，切分受视频分组影响；树模型容易受到采集分布影响，后续需要独立 real_test 进一步验证。

7. Review 分支仍需保留。review 可以承接边界样本，在不自动误复用的前提下给用户确认机会，是降低风险与提升体验之间的缓冲区。

## 7. 最终推荐

当前最佳工程方案：

```text
score = 0.5 * text_score + 0.5 * image_score

score >= 0.78:
  auto_hit，自动复用缓存

0.70 <= score < 0.78:
  review，提示用户确认

score < 0.70:
  miss，不复用，回退生成流程
```

- recommended_for_runtime: True / Candidate
- rule-fusion dual-threshold: Candidate
- learning-based models: False

这不表示学习式模型没有价值，而是说明当前阶段还不适合接入工程主流程。

## 8. 后续工作

- 继续扩充真实 ROI 独立测试集；
- 学习式模型暂不接入 plus.py；
- 若继续训练模型，应增加更多 hard_negative / near_positive；
- Review UI 保留为降低误复用风险的重要机制；
- 公开数据可作为补充，但不能替代真实 ROI 评估。

## 9. 附加文件

- chart_created: True
- ranking_file: `comprehensive_method_ranking.csv`
- historical_summary_file: `historical_experiment_summary.csv`
