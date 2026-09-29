# Code and artifact provenance

This note documents what was recovered from the competition project and what was
added while preparing the public repository.

## Recovered competition material

The supplied `TreeLearn_Street` directory contained:

- the TreeLearn model, dataset class, training loop, tiling pipeline, and evaluation utilities;
- WHU-STree-specific PLY-to-LAZ conversion scripts;
- outlier handling for known scenes;
- eight scene-specific inference YAML files;
- sequential, pinyin-aware, and KD-tree label back-projection experiments;
- batch inference commands;
- a 353 MB file named `hais_ckpt_spconv2.pth`.

The training YAML loads `hais_ckpt_spconv2.pth` through `pretrain`, while every
scene inference YAML points to `work_dirs/0812/epoch_1000.pth`. This establishes
that the 353 MB file is an initialization checkpoint and the missing
`epoch_1000.pth` is the fine-tuned competition model.

The supplied directory did not contain PLY/LAZ training clouds, generated crops,
the fine-tuned checkpoint, training logs, or Track 3 submission arrays.

## Upstream boundary

Files under `src/tree_learn/` and the research entry points under
`tools/competition/` derive from the MIT-licensed
[ecker-lab/TreeLearn](https://github.com/ecker-lab/TreeLearn) project. TreeLearn
itself acknowledges SoftGroup and spconv. This repository preserves the
TreeLearn license and does not represent the upstream architecture as original
work by the competition participant.

Competition-specific material includes the WHU-STree converters, scene configs,
outlier cases, batch sequence, point-order label export, and competition result
documentation. The public release additionally adds portable paths, validated
CLIs, unit tests, submission checks, and explicit reproducibility notes.

## Artifact checksums

```text
67cacb699d1a6fffb53ccb60e0d35d072495fa479771b6a46e16e2348f77702e  hais_ckpt_spconv2.pth
001ad32d7fe3ed81b1f35b63cacc06f8e6a96d81e9f0abaea67fd505524dad75  oversegmentation_after.png
fc53f66e459b2feb3fd8146b1241f3a3af488786db067082f40e47cedfe0ee44  oversegmentation_before.png
0d082de964dc11b942bee20ee4e47484585d6c6bead7ab75ef604ef02f5ba8b3  segmentation_examples.png
d750d0fe4d5f5216340b29789fb7f2f5418ed597e2c4ea3bad29bd1c46033136  leaderboard.png
1daa9c7b00b0817f606004794a22a84bf8c13fdf271ae30b12114e20a5d245fb  leaderboard_final.png
347870f00ce342f978b053fe2db96e9b4ba517fa9b1bd8221903b7838c33aa66  award_certificate_redacted.jpg
db9894f4667065630ae5820954c5b2b3aaf88be3b0a5faa1da76f9e45a781ed5  result_05_3-2.png
facb8510265c831645d8ad4e06192cc4a835e17de1e98d2e040b24131e834e79  result_13_2.png
b2719ec6a0efd2fb170dd1e75ed5e554acf49d6f115afb5d7271095d1fb96110  result_15_2-1.png
1c0684eb7f2ab8eb8b10fb4423504b330f1ab96aa1deb067d3ee4ed678099c6b  result_17_3-3.png
```

The four result images and the early leaderboard image match the media embedded
in the supplied PowerPoint exactly. `award_certificate_redacted.jpg` is a
deterministic privacy copy of the supplied certificate photograph: it was
rotated upright and two opaque rectangles were placed over the recipient and
supervisor names. No generative image was used for the public evidence file.
