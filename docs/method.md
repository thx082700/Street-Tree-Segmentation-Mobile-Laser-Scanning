# Method and evaluation

## Problem definition

For every input point in a mobile LiDAR scene, predict `0` for background or a positive integer
identifying one street-tree instance. The competition evaluates both instance detection and point
coverage. A useful model must therefore locate each tree and delineate its full crown and trunk.

## Sparse input representation

The loader subtracts the scene origin and quantizes coordinates at the configured voxel size. A
single representative point supplies the features for each occupied voxel. The sparse tensor uses
four channels:

1. occupancy (`1`),
2. robustly normalized intensity,
3. relative height, and
4. horizontal radius from the scene median.

The inverse voxel map transfers predictions back to every original point without reordering the
scene.

## Network and targets

The encoder downsamples a sparse tensor three times. Residual sparse convolutions increase the
receptive field while keeping computation proportional to occupied voxels. The decoder restores
the original voxel resolution through transposed sparse convolutions and skip connections.

Two `1 x 1 x 1` sparse heads predict:

- logits for background and tree; and
- a three-dimensional vector from each tree point to its instance centroid.

For point `p_i` belonging to tree `k`, the offset target is

```text
o_i = mean({p_j | instance(j) = k}) - p_i
```

Background points have a zero target and do not contribute to the offset loss.

## Objective

```text
L = weighted_cross_entropy(semantic_logits, tree_mask)
    + lambda_offset * mean_squared_error(predicted_offset, target_offset)
```

Class weights compensate for the high proportion of roads, buildings, vehicles, and other
background points. The original report describes fine-tuning from S3DIS pretraining. This release
loads every shape-compatible layer from a supplied checkpoint and safely leaves incompatible task
heads uninitialized.

## Instance generation

1. Retain voxels whose predicted tree probability exceeds the threshold.
2. Shift each retained voxel by its predicted centroid offset.
3. Run DBSCAN in shifted 3D space.
4. Split clusters whose horizontal extent or point count is implausibly large.
5. Merge small fragments and nearby noise into stable clusters.
6. Renumber final trees from `1` and map voxel IDs back to original points.

The fourth and fifth stages encode the two failure cases identified in the competition report:
under-segmentation creates abnormally large clusters, while over-segmentation creates clusters
with very few points.

| Fragmented prediction | After fragment merging |
| --- | --- |
| ![Small over-segmented fragments](../assets/oversegmentation_before.png) | ![Merged tree instance](../assets/oversegmentation_after.png) |

## Metrics

Let `G_i` be a ground-truth instance and `P_j` a predicted instance.

```text
IoU(G_i, P_j) = |G_i intersect P_j| / |G_i union P_j|
```

The evaluator uses one-to-one Hungarian matching. A matched pair counts as a true positive when
its IoU reaches the competition threshold, `0.75`. Precision, Recall, and F1 follow from the total
true positives, false positives, and false negatives.

Coverage measures the best available overlap for each ground-truth tree, whether or not that
overlap reaches the detection threshold:

```text
Cov  = mean_i max_j IoU(G_i, P_j)
WCov = sum_i |G_i| * max_j IoU(G_i, P_j) / sum_i |G_i|
```

WCov gives larger trees proportionally more influence. Dataset-level Cov is weighted by the
number of ground-truth instances in each scene; dataset-level WCov is weighted by the number of
ground-truth tree points.
