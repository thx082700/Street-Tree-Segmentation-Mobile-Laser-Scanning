# WHU-STree data notes

## Sources

- Official dataset repository: <https://github.com/WHU-USI3DV/WHU-STree>
- Competition resource page: <http://www.lidar2025wuhan.com/resources1.html>
- Track 3 Codabench page: <https://www.codabench.org/competitions/8821/>

The dataset is not redistributed in this repository. Follow the official license,
request procedure, and competition terms.

## Competition layout

Recovered scripts expect each split to contain road directories:

```text
train/
├── 05/PCD/*.ply
├── 06/PCD/*.ply
└── ...
```

Training files expose XYZ and a `tree` instance field. Current public WHU-STree
files may additionally provide intensity and species labels; the competition
pipeline does not require species labels.

## Working layout

`tools/convert_whu_stree.py` produces:

```text
data/competition/
├── train/forests/<road>_<cloud>.laz
└── test/pipeline_<road>_<cloud>/forest/<road>_<cloud>.laz
```

Random crops are written under `train/random_crops/npz`; validation tiles under
`val/tiles/npz`; inference outputs remain inside each `pipeline_*` directory.

## Submission layout

Every NumPy file has the original PLY point count and order, shape `[N, 1]`, and
dtype `int16`. The recovered submission code uses `-1` for background/unassigned
points and positive integers for tree instances. The submission ZIP is flat and
contains no parent directory.

## Data leakage

Use road-disjoint training and validation sets. Nearby trajectories can capture
the same trees, so random file-level splitting may leak scene geometry across
training and validation.
