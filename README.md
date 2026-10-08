# Street-Tree-Segmentation-Mobile-Laser-Scanning

[![CI](https://github.com/thx082700/Street-Tree-Segmentation-Mobile-Laser-Scanning/actions/workflows/ci.yml/badge.svg)](https://github.com/thx082700/Street-Tree-Segmentation-Mobile-Laser-Scanning/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/Python-3.10-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Task](https://img.shields.io/badge/task-3D%20instance%20segmentation-5c6bc0)](http://www.lidar2025wuhan.com/resources1.html)


First place solution for Track 3 (Urban Street-Tree Instance Segmentation) of the
9th National LiDAR Conference Point Cloud Intelligent Processing Competition (2025).

[中文说明](README_CN.md)

![Street-tree instance predictions from the competition report](assets/segmentation_examples.png)

## Prediction gallery

The following images are the original figures embedded in the post-competition
presentation. Colors identify predicted tree instances; gray points are the
surrounding road scene.

| Scene | Challenge |
| --- | --- |
| ![WHU-STree scene 05_3-2](assets/result_05_3-2.png) | Variable tree size and multiple rows |
| ![WHU-STree scene 13_2](assets/result_13_2.png) | Uneven point density |
| ![WHU-STree scene 15_2-1](assets/result_15_2-1.png) | Overlapping broadleaf crowns |
| ![WHU-STree scene 17_3-3](assets/result_17_3-3.png) | Mixed broadleaf and conifer geometry |

## Installation

Use a CUDA-capable Linux workstation with Python 3.10, PyTorch 2.0.0,
CUDA 11.8, and `spconv-cu118`.

```bash
git clone https://github.com/thx082700/Street-Tree-Segmentation-Mobile-Laser-Scanning.git
cd Street-Tree-Segmentation-Mobile-Laser-Scanning
conda create -n Street-Tree-Segmentation-Mobile-Laser-Scanning python=3.10
conda activate Street-Tree-Segmentation-Mobile-Laser-Scanning
conda install pytorch=2.0.0 torchvision=0.15.0 torchaudio=2.0.0 \
  pytorch-cuda=11.8 -c pytorch -c nvidia
python -m pip install spconv-cu118
python -m pip install -e ".[competition]"
```

## Data Preparation

Obtain WHU-STree from its [official repository](https://github.com/WHU-USI3DV/WHU-STree).
Place labeled training PLY files under
`data/raw/WHU-STree-for-competition/train/<road>/PCD/`
and test PLY files under `data/raw/WHU-STree-for-competition/test/<road>/PCD/`.

Convert the training and test point clouds:

```bash
python tools/convert_whu_stree.py \
  data/raw/WHU-STree-for-competition/train \
  data/competition/train --labeled
python tools/convert_whu_stree.py \
  data/raw/WHU-STree-for-competition/test \
  data/competition/test
```

Move a held-out validation cloud from `data/competition/train/forests/` to
`data/competition/val/forest/`. The default configuration uses `12_1.laz`;
set `forest_path` in `configs/competition/gen_val_data.yaml` if using another file.
Then generate training crops and validation tiles:

```bash
python tools/competition/data_gen/gen_train_data.py \
  --config configs/competition/gen_train_data.yaml
python tools/competition/data_gen/gen_val_data.py \
  --config configs/competition/gen_val_data.yaml
```

## Training

Place the SoftGroup/HAIS initialization at `checkpoints/hais_ckpt_spconv2.pth`,
then start training:

```bash
python tools/train.py --config configs/competition/train.yaml \
  --work_dir competition
```

Training settings are in `configs/competition/train.yaml`. Checkpoints are saved
under `work_dirs/competition/`.

## Inference

Place the fine-tuned checkpoint at `checkpoints/competition_epoch_1000.pth`,
then run a scene:

```bash
python tools/infer.py \
  --config configs/competition/pipeline/pipeline_13_2.yaml
```

To run all eight scene configurations:

```bash
python tools/run_competition_batch.py
```

## Evaluation

Evaluate aligned NumPy ground-truth and prediction directories:

```bash
sparsetree-evaluate data/reference_npy outputs/predictions \
  --iou-threshold 0.75
```

The evaluator reports precision, recall, F1, Cov, and WCov.
For forest-level evaluation, set `paths.pred_forest_path` and
`paths.gt_forest_path` in `configs/competition/evaluate.yaml`, then run:

```bash
python tools/competition/evaluation/evaluate.py \
  --config configs/competition/evaluate.yaml
```

## Citation

If this repository supports your work, please cite:

```latex
\bibitem{street_tree_segmentation_mls}
Haoxi Tian.
\textit{Street-Tree-Segmentation-Mobile-Laser-Scanning}.
GitHub repository, 2026.
\url{https://github.com/thx082700/Street-Tree-Segmentation-Mobile-Laser-Scanning}.
```

## License

Street-Tree-Segmentation-Mobile-Laser-Scanning's original code and documentation
are released under the
[MIT License](LICENSE). Upstream-derived TreeLearn files remain subject to the
[upstream MIT license notice](https://github.com/ecker-lab/TreeLearn/blob/main/LICENSE).

Please also cite TreeLearn using the bibliographic information in its
[official repository](https://github.com/ecker-lab/TreeLearn).
