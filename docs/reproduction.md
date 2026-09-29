# Reproduction guide

## 1. Hardware and software

The recovered environment targets Linux with an NVIDIA GPU, CUDA 11.8, Python
3.10, PyTorch 2.0.0, and `spconv-cu118`. Full-scene tiling can require substantial
RAM and approximately 10 GB of GPU memory, depending on point density.

```bash
conda env create -f environment.yml
conda activate sparsetree3d
```

Use a different spconv package when your CUDA runtime differs.

## 2. Dataset

Arrange the organizer-provided split as `<split>/<road>/PCD/*.ply`. Training PLY
files must contain `x`, `y`, `z`, and `tree`. Test files require only XYZ.

```bash
python tools/convert_whu_stree.py DATA/train data/competition/train --labeled
python tools/convert_whu_stree.py DATA/test data/competition/test
```

The converter preserves source point ordering externally and writes TreeLearn
working files internally. The export stage later restores exactly one label per
source PLY point.

## 3. Generate samples

Edit `configs/competition/gen_val_data.yaml` so `forest_path` points to the chosen
labeled validation forest, then run:

```bash
python tools/competition/data_gen/gen_train_data.py \
  --config configs/competition/gen_train_data.yaml
python tools/competition/data_gen/gen_val_data.py \
  --config configs/competition/gen_val_data.yaml
```

The recovered configuration requests 25,000 random crops. The upstream TreeLearn
documentation warns that this can consume hundreds of gigabytes; reduce
`n_samples_total` for a smoke run.

## 4. Train

Place the external initialization at `checkpoints/hais_ckpt_spconv2.pth` or set
`pretrain: null` in `configs/competition/train.yaml`.

```bash
python tools/train.py --config configs/competition/train.yaml \
  --work_dir competition
```

Checkpoints appear in `work_dirs/competition/`. Select a validated checkpoint and
copy or link it to `checkpoints/competition_epoch_1000.pth`, the path used by the
recovered inference configs.

## 5. Inference

```bash
python tools/infer.py \
  --config configs/competition/pipeline/pipeline_13_2.yaml
```

After confirming one scene, run all recovered configs:

```bash
python tools/run_competition_batch.py
```

The configs cover `05_3-2`, `06_2-1`, `09_1-1`, `12_2`, `13_2`, `15_1`,
`15_2-1`, and `17_3-3`. Their `outer_remove=13.5` behavior and scene paths are
preserved from the competition project.

## 6. Export

```bash
python tools/export_submission.py DATA/test data/competition/test \
  outputs/sparsetree3d_submission.zip
```

The exporter uses a KD-tree with a 1 cm tolerance to match processed points back
to the original PLY order. It writes `-1` for unmatched/background points and
positive IDs for tree instances.

## 7. Validate portable components

```bash
python -m pip install -e ".[dev]"
pytest
ruff check src/sparsetree3d tests tools \
  --exclude src/tree_learn --exclude tools/competition
```

## Exact-score limitation

The historical F1 0.829 belongs to Codabench submission `362701`. Exact score
reproduction requires the original competition split and the missing fine-tuned
`epoch_1000.pth`. The repository contains the workflow needed to retrain and
submit, but it does not substitute the pretraining initialization for the final
model.
