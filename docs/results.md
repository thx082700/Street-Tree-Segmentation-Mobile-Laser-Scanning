# Competition result and evidence

## Award

Haoxi Tian (田浩希), Southwest Petroleum University, received **特等奖** in
“城市道路场景单木分割” at the 9th National LiDAR Conference Point Cloud
Intelligent Processing Competition. For an English CV, **Grand Prize (特等奖)**
is the clearest wording because it preserves the official Chinese title.

The certificate is dated September 2025. The repository publishes a
[privacy-redacted photograph](../assets/award_certificate_redacted.jpg): opaque
blocks cover the recipient and supervisor names, while the competition title,
award level, organizer seal, and date remain visible. The public image was made
by rotating the supplied photograph and overlaying solid rectangles; it was not
generatively reconstructed.

## Codabench submission

The supplied public-leaderboard records identify participant `thx0827` and
submission ID `362701` with the following metrics:

| Metric | Score |
| --- | ---: |
| Coverage (Cov) | 0.849 |
| Weighted coverage (WCov) | 0.903 |
| Precision | 0.843 |
| Recall | 0.816 |
| F1 | **0.829** |

The [September 2 snapshot](../assets/leaderboard.png) shows the submission in
first place before the leaderboard received later entries. The
[later screenshot](../assets/leaderboard_final.png) shows it in second place
after a September 7 submission. The stable wording is therefore:

> Achieved 0.829 F1 and 0.903 weighted coverage, placing 2nd on the final public
> leaderboard and receiving the competition's Grand Prize (特等奖).

This wording does not equate leaderboard rank with the organizer's award.

## Result figures

The repository's prediction figures are extracted byte-for-byte from the supplied
post-competition PowerPoint. Their SHA-256 values are recorded in
[`provenance.md`](provenance.md). They cover variable tree size, uneven density,
overlapping crowns, and broadleaf/conifer geometry.

## Claims that should not be made

- Do not call the project a first-place final leaderboard entry.
- Do not describe the SoftGroup/HAIS initialization as the competition checkpoint.
- Do not claim that F1 0.829 was rerun from the public repository.
- Do not claim authorship of the upstream TreeLearn method.
