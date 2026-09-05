# Resume wording

Use the version that best matches the available space. Keep “Second Prize” as the award and treat
the first-place statement as a time-specific public leaderboard result.

## One-line version

**SparseTree3D — Urban LiDAR Instance Segmentation:** Built a sparse 3D U-Net with joint semantic
classification and centroid-offset regression for WHU-STree street-tree point clouds; added
size-aware clustering to correct over- and under-segmentation, achieving 0.829 F1 and 0.903 WCov
on the 2025 competition public leaderboard (Track 3 Second Prize).

## Two-bullet version

- Engineered a GPU-efficient 3D perception pipeline that voxelizes mobile LiDAR scenes, extracts
  sparse convolutional features, and predicts both tree occupancy and instance-center offsets.
- Designed fragment-merging and oversized-cluster splitting for variable point density and crown
  overlap; achieved 0.829 F1, 0.903 WCov, and first place on the captured public leaderboard,
  receiving Second Prize in Track 3 of the 9th National LiDAR Conference competition.

## Interview talking points

- Explain why sparse convolution fits large outdoor point clouds: compute follows occupied voxels.
- Contrast semantic segmentation with instance segmentation and motivate centroid-offset targets.
- Describe how crown overlap causes under-segmentation and low-density returns cause fragments.
- Discuss road-level validation splits as protection against spatial data leakage.
- State the reproducibility boundary clearly: the public repository is a clean implementation of
  the reported method; retired competition data and the original checkpoint are not redistributed.

