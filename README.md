# var-memorization-analysis

Measuring [UnitMem](https://arxiv.org/abs/2411.03429) memorization on the pretrained [VAR-d16](https://github.com/FoundationVision/VAR) image autoregressive model.

VAR-d16 is a GPT-style causal Transformer (16 blocks, embed dim 1024) that generates
images autoregressively across 10 scales, from a single token up to 16x16. This project
adapts UnitMem — originally designed for self-supervised CNN encoders — to VAR's
architecture and asks two questions:

- **RQ1** — Among the top 10% most memorizing units, where are they located, by layer
  type (fc1 / attention output projection / fc2) and by block depth?
- **RQ2** — Which training images do those units respond to most selectively, and do
  those images share common visual patterns compared with class-matched controls?

A narrative summary of the findings is in [`results/SUMMARY.md`](results/SUMMARY.md), with the
two summary figures under [`results/figures/`](results/figures/).

## Method summary

UnitMem quantifies how selectively a single unit responds to one specific image. For a
unit `u` over a dataset `D'`, the response `a_u(x)` is the mean absolute activation over
10 augmented views of `x`. UnitMem compares the most-activating image against the mean
of all others:

```
UnitMem(u) = (mu_max(u) - mu_-max(u)) / (mu_max(u) + mu_-max(u))
```

VAR complicates this in two ways that the pipeline accounts for:

- **Token counts are wildly unbalanced across scales** (680 tokens total; the two finest
  scales alone hold 62.5% of them), so UnitMem is computed independently *per scale*
  rather than pooling all tokens together — otherwise it would measure almost nothing
  but the finest scales.
- **Scale 0 is a single token, the class-conditioning `[s]` token**, built only from the
  class embedding — it carries no image information (VAR's block-wise causal mask means
  position 0 can only attend to itself) and is excluded from both research questions.

The final RQ1/RQ2 score for a unit averages its per-scale UnitMem values over the nine
image-dependent scales (excluding scale 0).

## Repository layout

```
src/                          core measurement library (model loading, activation
                               hooks, UnitMem accumulators, data augmentation)
01_build_subset.py             build the initial 1,000-image ImageNet subset
02_build_manifest.py           extend to the full 10,000-image manifest
03_compute_activations.py      forward pass: accumulate UnitMem statistics
04_extract_image_features.py   low-level image features used in RQ2
common.py                      shared helpers for the RQ1/RQ2 post-processing stages
05_rq1_localization.py         RQ1: layer-type / block-location analysis
06_rq2_vote_counting.py        RQ2: per-image vote counting from top-10% units
07_pattern_analysis_rq2.py     RQ2: class-matched statistical comparison
08_build_annotation_materials.py  blind annotation materials for semantic patterns
09_semantic_pattern_analysis.py   RQ2: unblinded semantic-pattern statistics
10_semantic_pattern_figure.py     supplementary figure for semantic patterns
11_rq1_summary_figure.py       RQ1 summary figure (layer-type / block heatmap)
12_rq2_feature_figure.py       RQ2 summary figure (class-matched feature differences)
verify_results.py              structural/numerical self-checks on the pipeline output
results/                       result summaries and figures (see below)
```

Scripts are numbered in pipeline order and run from the repository root, e.g.
`python 03_compute_activations.py --stage full10000`.

## What isn't in this repository

- **Model checkpoints** (`checkpoints/`) — download VAR-d16 and its VQVAE from the
  [VAR repository](https://github.com/FoundationVision/VAR) and place them at
  `checkpoints/vae_ch160v4096z32.pth` and `checkpoints/var_d16.pth`.
- **The VAR model source** — clone [FoundationVision/VAR](https://github.com/FoundationVision/VAR)
  into `VAR/` at the repository root; `src/model.py` imports from it directly.
- **ImageNet training images** (`data/images/`, `data_N10000/images/`) — this project
  samples 10,000 images from `ILSVRC/imagenet-1k`'s gated training split on Hugging Face;
  obtaining access requires accepting ImageNet's terms. `01_build_subset.py` and
  `02_build_manifest.py` build this subset locally given an `HF_TOKEN` with access.
- **Any figure containing image thumbnails.** ImageNet's terms of use do not permit
  redistributing the images themselves. `results/figures/fig2_rq2.png` therefore only
  shows the statistical comparison panel; a companion panel of most-voted image
  thumbnails, and a few supplementary figures/contact sheets built purely for visual
  inspection, are omitted from this repository (the scripts that produce them, given
  the data, are still included).
- **Large intermediate arrays** (the raw per-unit activation statistics, tens of MB of
  `.npz` files) — regenerable by running the pipeline, but not committed. `results/`
  contains the small human-readable summaries and the two small `.npz` files needed by
  the later analysis stages.

`results/SUMMARY.md` has the full numeric results (RQ1 and RQ2 tables) if you want the
findings without running anything.

## Setup

```
pip install -r requirements.txt
```

Requires an NVIDIA GPU with a few GB of free VRAM for the forward pass (bfloat16
autocast); everything else runs on CPU.
