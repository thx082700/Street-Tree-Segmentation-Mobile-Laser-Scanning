# SparseTree3D

SparseTree3D 是面向车载激光雷达城市行道树点云的单木实例分割项目。本仓库现已纳入
比赛时使用的真实 `spconv` 稀疏卷积代码、训练与推理配置、数据转换、点标签回投、评测
及 Codabench 提交工具。

**比赛成绩：** 第九届全国激光雷达大会点云智能解析大赛赛道三“城市道路场景单木
分割”**特等奖**。Codabench 提交 `362701` 的 F1 为 **0.829**、WCov 为 **0.903**，
最终公开榜排名第 **2**。2025 年 9 月 2 日保存的早期截图中，该提交曾暂列第 1；后来有
新的提交进入榜单，因此奖项、早期榜单和最终榜单在仓库中分别说明，避免混淆。

![比赛结果可视化](assets/segmentation_examples.png)

## 评测结果

| 指标 | 成绩 |
| --- | ---: |
| Cov | 0.849 |
| WCov | 0.903 |
| Precision | 0.843 |
| Recall | 0.816 |
| **F1** | **0.829** |
| 最终公开榜 | **第 2 名** |
| 正式奖项 | **特等奖** |

- [2025 年 9 月 2 日榜单截图](assets/leaderboard.png)
- [最终榜单截图](assets/leaderboard_final.png)
- [获奖证书隐私处理版](assets/award_certificate_redacted.jpg)（姓名与指导教师姓名已遮挡）
- [成绩与证据说明](docs/results.md)

![隐私处理后的特等奖证书](assets/award_certificate_redacted.jpg)

## 实际方案

1. 将 WHU-STree PLY 转换为 TreeLearn 使用的 LAZ 格式，并处理异常坐标。
2. 以 0.10 m 体素进行采样，生成随机训练裁块和重叠推理瓦片。
3. 使用七层 `spconv` 稀疏 U-Net 提取点云特征。
4. 同时预测树木/非树木语义和指向单木基部/中心的三维偏移。
5. 根据语义置信度、垂直度和偏移幅值过滤点，在偏移后的坐标上执行 HDBSCAN/DBSCAN。
6. 将未分配点归入最近有效实例，再把标签按原 PLY 点序回投。
7. 输出 Codabench 需要的 `[N, 1]`、`int16` NumPy 文件并压缩成 ZIP。

赛时配置中的主要参数为：体素大小 0.10 m、`tau_vert=0.6`、`tau_off=4`、
`tau_min=50`。训练采用 AdamW、初始学习率 0.003、batch size 6、混合精度和最长
3,000 epoch。

## 仓库内容

```text
src/tree_learn/              比赛使用的模型、数据集与推理流水线
src/sparsetree3d/            可独立测试的转换、回投、评测和提交工具
configs/competition/         训练配置和 8 个场景推理配置
tools/competition/           训练、推理、数据生成和评测入口
tools/convert_whu_stree.py   PLY 到 LAZ 转换
tools/run_competition_batch.py
tools/export_submission.py   回投标签并生成提交 ZIP
tests/                       CPU 可运行的回归测试
assets/                      原始汇报结果图、榜单与获奖证明
```

## CPU 快速检查

这条流程用于检查安装、聚类、评测和提交工具，不等同于真实神经网络预测：

```bash
git clone https://github.com/thx082700/SparseTree3D.git
cd SparseTree3D
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e ".[dev]"
python examples/synthetic_demo.py
pytest
```

## GPU 环境

恢复的比赛环境为 Python 3.10、PyTorch 2.0.0、CUDA 11.8 和 `spconv-cu118`：

```bash
conda env create -f environment.yml
conda activate sparsetree3d
```

## 数据准备

通过 [WHU-STree 官方仓库](https://github.com/WHU-USI3DV/WHU-STree)申请或下载数据，
不要把数据集提交到本仓库。

```bash
python tools/convert_whu_stree.py \
  data/raw/WHU-STree-for-competition/train \
  data/competition/train --labeled

python tools/convert_whu_stree.py \
  data/raw/WHU-STree-for-competition/test \
  data/competition/test
```

生成训练裁块和验证瓦片：

```bash
python tools/competition/data_gen/gen_train_data.py \
  --config configs/competition/gen_train_data.yaml
python tools/competition/data_gen/gen_val_data.py \
  --config configs/competition/gen_val_data.yaml
```

## 训练、推理和提交

`checkpoints/hais_ckpt_spconv2.pth` 是 SoftGroup/HAIS 预训练初始化，不是比赛最终模型。
训练命令：

```bash
python tools/train.py --config configs/competition/train.yaml \
  --work_dir competition
```

将选定的微调模型保存为 `checkpoints/competition_epoch_1000.pth` 后，可以推理一个场景
或全部八个场景：

```bash
python tools/infer.py \
  --config configs/competition/pipeline/pipeline_13_2.yaml
python tools/run_competition_batch.py
```

生成提交包：

```bash
python tools/export_submission.py \
  data/raw/WHU-STree-for-competition/test \
  data/competition/test \
  outputs/sparsetree3d_submission.zip
```

## 可复现范围

本次恢复的项目文件包含真实比赛代码、配置和 353 MB 预训练初始化权重，但不包含比赛
训练数据、最终微调权重 `work_dirs/0812/epoch_1000.pth` 和训练日志。因此仓库可以用于
重新训练、推理和制作合规提交，但在找回最终权重前，不能声称下载后即可复现 F1 0.829。
仓库中的分数对应历史提交 `362701`，完整边界见
[docs/reproduction.md](docs/reproduction.md) 和 [docs/provenance.md](docs/provenance.md)。

## 项目归属与引用

比赛适配与提交者：**田浩希，西南石油大学**；指导教师：**熊俊楠、贾宏亮**。

本项目基于开源 [TreeLearn](https://github.com/ecker-lab/TreeLearn)，后者又使用了
SoftGroup 与 spconv。本仓库不把上游方法声明为个人原创；比赛工作主要包括 WHU-STree
适配、训练配置、场景预处理、批量推理、结果回投与提交。许可证与第三方说明见
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
