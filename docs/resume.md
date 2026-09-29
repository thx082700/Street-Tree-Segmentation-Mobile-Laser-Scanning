# CV and interview wording

## Recommended one-line CV entry

**SparseTree3D — Urban LiDAR Instance Segmentation:** Adapted and fine-tuned a
`spconv` sparse U-Net for WHU-STree mobile-LiDAR scenes; built preprocessing,
offset-guided clustering, and point-order submission export, achieving 0.829 F1
and 0.903 WCov (2nd final public leaderboard; Grand Prize, 特等奖).

## Two-bullet version

- Adapted the TreeLearn/SoftGroup sparse-convolution pipeline to urban mobile-
  LiDAR, combining tree/background classification with point-to-instance offset
  regression and HDBSCAN grouping for overlapping street-tree crowns.
- Engineered WHU-STree conversion, scene-specific inference, KD-tree label
  back-projection, and Codabench validation; achieved 0.829 F1 and 0.903 WCov,
  placing 2nd on the final public leaderboard and receiving the Grand Prize
  (特等奖) in Track 3 of the 2025 National LiDAR Conference competition.

## Interview talking points

- Sparse convolutions limit computation to occupied voxels in large outdoor scans.
- Semantic logits identify tree points; offset regression makes points from one
  tree easier to cluster even when neighboring crowns overlap.
- HDBSCAN reduces sensitivity to one fixed density radius, while `tau_min`
  rejects fragments caused by sparse returns.
- A KD-tree restores predictions to the original PLY point order after filtering
  and voxelization, which is essential for valid Codabench files.
- The public repository includes the authentic workflow and configs but not the
  competition data or final fine-tuned checkpoint. The score is tied to submission
  `362701`, not claimed as a fresh public rerun.

## Attribution answer

If asked whether the architecture was created from scratch:

> I adapted the open-source TreeLearn pipeline, which builds on SoftGroup and
> spconv. My competition work focused on transferring it to WHU-STree, configuring
> and fine-tuning the model, handling problematic scenes, running batch inference,
> mapping predictions back to the organizer's point order, and producing the
> evaluated submission.
