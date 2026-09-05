# WHU-STree data notes

## Official release

WHU-STree combines synchronized mobile LiDAR point clouds and street-view images from Nanjing and
Shenyang. The official repository reports 21,007 annotated tree instances, more than 50 species,
and tree height and diameter-at-breast-height attributes. See:

- Dataset repository: <https://github.com/WHU-USI3DV/WHU-STree>
- Track 3 rules: <https://www.lidar2025wuhan.com/resources1.html>
- Competition page: <https://www.codabench.org/competitions/8821/>

The official repository states that the 2025 competition subset is no longer distributed. Do not
commit requested dataset files or derived labels to this repository. Follow the dataset terms in
the data-request form.

## Point-cloud fields

The current release documents each PLY vertex as:

```text
[x, y, z, intensity, tree, label]
```

`tree` contains the instance ID. `label` contains the species label and is intentionally unused by
the species-agnostic Track 3 pipeline.

## Split policy

The helper groups clouds by road before hashing roads into train and validation sets. This avoids
putting two trajectories from the same road in different sets. For official benchmarking, use the
provided split and reserve `reference_data` for evaluation only.

## Competition output

Each prediction must preserve the input point order and contain one instance ID per point. Files
use a non-negative `int16` vector and the exact name `<road_id>_<point_cloud_id>.npy`. The final ZIP
is flat; it contains no parent directory.

