# ZaowuPi AR 3D Demo

造物π：基于手势姿态估计与图文双相似度缓存复用的 AR 3D 生成系统。

本仓库是比赛展示与复现实验的精简版本，保留可运行 Demo、核心缓存策略代码、演示 GLB 资产和关键实验结果。仓库不包含 Qwen、TripoSR、Stable Fast 3D、Hunyuan3D 等大模型权重，也不包含本地虚拟环境、个人 token 或临时视频文件。

## 在线体验

固定演示地址：

[https://zaowupi-ar3d-demo.serveousercontent.com](https://zaowupi-ar3d-demo.serveousercontent.com)

说明：该链接由本地服务和公网隧道提供。如果页面出现 Serveo 安全提示，请点击 Continue to Site。摄像头体验需要浏览器授权，并建议使用 Chrome / Edge。

## 项目能力

- 摄像头页面与浏览器端 AR 展示
- MediaPipe 手势姿态估计
- ROI 框选与目标触发
- GLB 模型前端加载
- 图文双相似度缓存复用策略
- `auto_hit / low_confidence_candidate / miss` 三分支判断
- 高质量预置 GLB 展示资产
- 多用户缓存复用和速度收益离线实验

## 核心方法

本项目提出的核心方法可以概括为：

**TIFS-Cache：Text-Image Fusion Similarity Cache Reuse**

即通过文本相似度和图像相似度融合判断一个新 ROI 是否可以复用已有 3D 缓存模型，从而减少重复调用 3D 生成后端的等待时间。

当前工程候选策略为：

```text
score = 0.5 * text_score + 0.5 * image_score

score >= strong_threshold:
  auto_hit，高置信自动复用

weak_threshold <= score < strong_threshold:
  low_confidence_candidate，低置信候选区

score < weak_threshold:
  miss，不复用
```

在 `real_eval_200` 的最终阈值扫描中，工程保守默认更倾向选择误复用风险低、候选区可控的工作点。学习式模型目前只作为探索实验，不接入主流程。

## 本地运行

推荐环境：

- Python 3.10 / 3.11 / 3.12
- Windows + Chrome / Edge
- 摄像头

安装依赖：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

轻量比赛 Demo 启动方式：

```powershell
.\.venv\Scripts\python.exe demo_static_server.py
```

浏览器打开：

```text
http://127.0.0.1:5000
```

如果要运行完整本地链路，可启动 `plus.py`：

```powershell
$env:PYTHONIOENCODING="utf-8"
$env:ENABLE_POLICY_CACHE="1"
$env:CACHE_POLICY_PATH="runtime_assets/cache_policy.json"
.\.venv\Scripts\python.exe plus.py
```

浏览器打开：

```text
http://127.0.0.1:5000
```

说明：`demo_static_server.py` 只负责展示摄像头交互和预置 GLB 资产，不调用 Qwen、TripoSR 或任何 3D 生成模型。`plus.py` 是完整链路入口，依赖本机环境更多。

## 目录说明

```text
plus.py
  Flask 主程序，负责页面、摄像头、SSE 状态、GLB 展示和缓存策略接入。

cache_similarity.py
  文本/图像相似度与缓存索引逻辑。

cache_policy_loader.py
  读取策略配置，计算 policy_score，输出 auto_hit / low_confidence_candidate / miss。

cache_policy_dry_run.py
  不启动完整生成流程的策略 dry-run 工具。

runtime_assets/
  运行时配置、浏览器 demo 页面、比赛展示 GLB 资产。

scripts/
  完整实验链路脚本，包括样本整理、标签检查、公开数据构造、真实 ROI 特征计算、
  学习式融合训练、迁移评估、阈值扫描、综合对比和多用户仿真。

docs/experiments/
  关键实验报告、综合对比结果和阈值 Pareto 扫描结果。

docs/competition/
  比赛说明文档 PDF。
```

## 关键实验结果

- `real_eval_200`：共 200 条真实 ROI 样本，正负样本各 100 条。
- 规则融合 + 双阈值仍是当前最佳工程方案。
- `auto_hit` 单样本速度验证：从 baseline 约 51.236 秒降到约 3.403 秒，单次节省约 47.83 秒。
- 公开数据训练模型迁移到真实 ROI 后出现较高 false hit，因此暂不接入工程主流程。
- 学习式模型后续需要更多真实 ROI 独立测试集验证。

详细报告见：

- `docs/experiments/comprehensive_method_comparison_summary.md`
- `docs/experiments/comprehensive_method_comparison_full_report.md`
- `docs/experiments/final_threshold_pareto_report.md`

## 模型与资产说明

本仓库包含比赛展示用的小体积 GLB 资产，位于：

```text
runtime_assets/competition_demo_models_hq/
```

这些资产用于演示前端加载、AR 展示和缓存复用效果。大模型权重不随仓库提交，原因是体积大、授权复杂，也不是本项目核心贡献。详见 `MODEL_ASSETS.md`。

## 复现注意事项

- 不要把 `.venv/`、`.venv_public/`、`.venv_sf3d/` 提交到 GitHub。
- 不要提交 Hugging Face token、API key、SSH 私钥或个人隐私配置。
- TripoSR 和 Stable Fast 3D 只作为后端替换或 baseline 讨论，不是本仓库在线 Demo 的必要依赖。
- 线上固定链接依赖本地服务与公网隧道，如需长期稳定部署，建议后续迁移到 HTTPS 云服务器。

## 完整链路复现入口

常用脚本入口如下：

```text
scripts/run_cache_v3_experiment.py
  v3_real 系列缓存复用实验入口。

scripts/prepare_public_train_1000_dataset.py
scripts/train_public_similarity_fusion_classifier.py
scripts/evaluate_public_model_on_real_v3.py
  公开数据训练与迁移评估链路。

scripts/prepare_real_add_130.py
scripts/compute_real_add_130_features.py
scripts/train_real_eval_200_conservative_classifier.py
  real_eval_200 真实 ROI 训练评估链路。

scripts/run_comprehensive_method_comparison.py
scripts/final_pareto_threshold_sweep.py
  全方法综合对比与最终阈值 Pareto 扫描。
```

这些脚本不会在 GitHub 页面中自动运行。若要复现实验，请先准备对应数据目录，并确认不调用外部大模型。
