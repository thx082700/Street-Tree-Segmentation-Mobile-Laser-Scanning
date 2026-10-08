# Third-party notices

## TreeLearn

The sparse-convolution model, dataset class, training loop, tiling pipeline, and
several utility modules under `src/tree_learn/` and `tools/competition/` are
adapted from [ecker-lab/TreeLearn](https://github.com/ecker-lab/TreeLearn).
TreeLearn is distributed under the MIT License. A copy is included at
[`third_party/TreeLearn_LICENSE`](third_party/TreeLearn_LICENSE).

This repository adds the WHU-STree competition conversion, scene configurations,
outlier handling, batch execution, prediction back-projection, submission
validation, tests, and project documentation.

## SoftGroup and spconv

TreeLearn builds on [SoftGroup](https://github.com/thangvubk/SoftGroup) and
[spconv](https://github.com/traveller59/spconv). Their respective licenses apply
to those projects. The `hais_ckpt_spconv2.pth` filename referenced by the training
configuration denotes an external pretrained initialization and is not included
in this repository.

## WHU-STree

The dataset belongs to the WHU-USI3DV authors and is not redistributed here.
Download or request it through the
[official WHU-STree repository](https://github.com/WHU-USI3DV/WHU-STree) and
follow its license and competition terms.
