# N=10000 paper-method post-processing — final results

Score definition used throughout this document: for unit *u*, `S_u = mean(UnitMem_u,B_1, ..., UnitMem_u,B_9)` — the nine per-scale UnitMem values (each already computed independently per scale) averaged after computation. This is **not** the scale-averaged-activation regime, **not** the token-count-weighted regime, and does not include scale0. A unit is excluded from the candidate pool if it is dead (mu_max < eps) in any of its 9 B_1..B_9 regimes. Pool: fc1 (65,536 units) + attn.proj (16,384) + fc2 (16,384) = 98,304 candidates; 0 dead units found in any group; k = floor(0.10 × 98,304) = **9,830** selected units.

---

## 1. RQ1 — Layer type

| layer type | candidate units | selected (top-10% pool) | selected / candidates | share of the 9,830 selected units |
|---|---|---|---|---|
| fc1 | 65,536 | 9,057 | 13.82% | 92.14% |
| attn.proj | 16,384 | 575 | 3.51% | 5.85% |
| fc2 | 16,384 | 198 | 1.21% | 2.01% |

fc1 units are selected into the pooled top-10% at roughly 4x the rate of attn.proj and 11x the rate of fc2 (13.82% vs 3.51% vs 1.21% of each type's own candidate pool). fc1 accounts for 92.14% of all 9,830 selected units despite being only 66.7% of the candidate pool.

## 2. RQ1 — Block location

Pooled across all three layer types, per block (candidates = 6,144 = 4,096 fc1 + 1,024 attn.proj + 1,024 fc2, per block):

| block | candidates | selected | in-block rate | share of selected |
|---|---|---|---|---|
| 0 | 6,144 | 2,165 | 35.24% | 22.02% |
| 1 | 6,144 | 1,984 | 32.29% | 20.18% |
| 2 | 6,144 | 1,447 | 23.55% | 14.72% |
| 3 | 6,144 | 1,050 | 17.09% | 10.68% |
| 4 | 6,144 | 755 | 12.29% | 7.68% |
| 5 | 6,144 | 837 | 13.62% | 8.51% |
| 6 | 6,144 | 464 | 7.55% | 4.72% |
| 7 | 6,144 | 312 | 5.08% | 3.17% |
| 8 | 6,144 | 80 | 1.30% | 0.81% |
| 9 | 6,144 | 68 | 1.11% | 0.69% |
| 10 | 6,144 | 119 | 1.94% | 1.21% |
| 11 | 6,144 | 78 | 1.27% | 0.79% |
| 12 | 6,144 | 21 | 0.34% | 0.21% |
| 13 | 6,144 | 36 | 0.59% | 0.37% |
| 14 | 6,144 | 40 | 0.65% | 0.41% |
| 15 | 6,144 | 374 | 6.09% | 3.80% |

Blocks 0, 1, and 2 are the top-3 blocks by selected-unit count (2,165 / 1,984 / 1,447), together accounting for 5,596 of the 9,830 selected units (56.93%). Selection rate declines roughly monotonically from block 0 to block 12, with block 15 showing a secondary, smaller rise (374 selected units, 6.09% in-block rate, 3.80% of all selected units) relative to its immediate neighbors (blocks 12–14, 0.34–0.65%).

## 3. RQ1 — Joint layer-type × block result

Selected / candidate (rate), 16 blocks × 3 layer types:

| block | fc1 | attn.proj | fc2 |
|---|---|---|---|
| 0 | 1609/4096 (39.28%) | 374/1024 (36.52%) | 182/1024 (17.77%) |
| 1 | 1861/4096 (45.43%) | 115/1024 (11.23%) | 8/1024 (0.78%) |
| 2 | 1421/4096 (34.69%) | 26/1024 (2.54%) | 0/1024 (0.00%) |
| 3 | 1044/4096 (25.49%) | 6/1024 (0.59%) | 0/1024 (0.00%) |
| 4 | 751/4096 (18.33%) | 3/1024 (0.29%) | 1/1024 (0.10%) |
| 5 | 825/4096 (20.14%) | 12/1024 (1.17%) | 0/1024 (0.00%) |
| 6 | 464/4096 (11.33%) | 0/1024 (0.00%) | 0/1024 (0.00%) |
| 7 | 305/4096 (7.45%) | 7/1024 (0.68%) | 0/1024 (0.00%) |
| 8 | 80/4096 (1.95%) | 0/1024 (0.00%) | 0/1024 (0.00%) |
| 9 | 68/4096 (1.66%) | 0/1024 (0.00%) | 0/1024 (0.00%) |
| 10 | 105/4096 (2.56%) | 13/1024 (1.27%) | 1/1024 (0.10%) |
| 11 | 65/4096 (1.59%) | 13/1024 (1.27%) | 0/1024 (0.00%) |
| 12 | 19/4096 (0.46%) | 2/1024 (0.20%) | 0/1024 (0.00%) |
| 13 | 34/4096 (0.83%) | 2/1024 (0.20%) | 0/1024 (0.00%) |
| 14 | 40/4096 (0.98%) | 0/1024 (0.00%) | 0/1024 (0.00%) |
| 15 | 366/4096 (8.94%) | 2/1024 (0.20%) | 6/1024 (0.59%) |

fc1's block-0/1 selection rates (39.28%, 45.43%) are the two highest cells in the table. attn.proj's block-0 rate (36.52%) is close to fc1's, but attn.proj's selection rate falls off far more sharply after block 0 (11.23% at block 1, ≤2.6% from block 2 onward) than fc1's does. fc2 selection is concentrated almost entirely in block 0 (17.77%), with every other block at or below 0.78% except block 15 (0.59%). See `report/fig1_rq1.png` (panel A: per-type selection ratio with a 10% reference line; panel B: this 16×3 rate heatmap).

## 4. RQ2 — Vote distribution

Each of the 9,830 selected units casts one vote per scale (B_1..B_9) for its argmax image at that scale: 9 × 9,830 = **88,470 total votes** (verified). 6,319 of the 10,000 images (63.19%) received at least one vote.

- Top-1 image (image_id 4187, class_id 418): 2,188 votes (2.4732% of all votes), from 464 distinct selected units spanning 15 of 16 blocks and all 3 layer types.
- Top-10 images: 8,307 combined votes (9.3896% of total).
- Top-50 images: 19,187 combined votes (21.6876% of total).
- Top-10 individual vote counts: 2188, 1048, 943, 881, 624, 570, 561, 542, 522, 428 (image_ids 4187, 8282, 4907, 9046, 2750, 848, 8156, 9049, 228, 9138; classes 418, 828, 490, 904, 275, 84, 815, 904, 22, 913).
- 383 images (3.83% of all 10,000 images; 6.06% of the 6,319 voted-for images) jointly account for ≥50% of the 88,470 total votes; the remaining ~50% is spread across the other 5,936 voted-for images.

Votes are numerically concentrated on a small minority of images: roughly 4% of all images account for half of all unit-scale votes, and the single most-voted image alone receives about 2.5% of all votes cast.

## 5. RQ2 — Class-matched objective image properties

Top-50 selected by vote count, spanning 40 distinct classes. 1:1 within-class matched control (seed=42, without replacement): 50 images, no shortfalls.

| feature | top-50 mean | control mean | diff | Cohen's d | bootstrap 95% CI | perm. p | BH-FDR q | significant? |
|---|---|---|---|---|---|---|---|---|
| grayscale entropy | 6.6912 | 6.9566 | −0.2654 | −0.224 | [−0.7362, +0.1654] | 0.1196 | 0.1595 | no |
| edge density | 0.6477 | 0.5866 | +0.0612 | +0.228 | [−0.0449, +0.1621] | 0.0844 | 0.1363 | no |
| Laplacian sharpness | 0.0771 | 0.0323 | +0.0448 | +0.637 | [+0.0190, +0.0724] | 0.0001 | **0.0008** | **yes** |
| luminance contrast | 0.1842 | 0.2062 | −0.0220 | −0.317 | [−0.0497, +0.0054] | 0.0852 | 0.1363 | no |
| mean saturation | 0.3997 | 0.3265 | +0.0731 | +0.309 | [−0.0171, +0.1641] | 0.0505 | 0.1347 | no |
| colorfulness | 45.1942 | 43.3583 | +1.8359 | +0.064 | [−9.0610, +13.0289] | 0.6920 | 0.6920 | no |
| center-vs-border contrast | 0.0666 | 0.0900 | −0.0234 | −0.320 | [−0.0523, +0.0050] | 0.1631 | 0.1864 | no |
| JPEG bytes/pixel | 0.4054 | 0.3399 | +0.0656 | +0.432 | [+0.0060, +0.1221] | 0.0018 | **0.0072** | **yes** |

Only two of the eight features reach BH-FDR significance (q<0.05): Laplacian sharpness (higher in top-50, d=+0.64) and JPEG bytes-per-pixel (higher in top-50, i.e. less compressible, d=+0.43). The other six features, including grayscale entropy and luminance contrast, are not significant at this N and this vote-based top-50 definition. This is an association between vote frequency and these two low-level statistics; it does not identify a specific visual pattern and does not establish causation.

## 6. Blind semantic-pattern annotation

Blind materials were generated for the vote-based top-50 and its matched control:
- A contact sheet of 100 images (50 top + 50 control), randomly shuffled, with only an anonymous `cell_id` shown, no group/class/vote information visible. (Not included in this repository — see the note on ImageNet redistribution in the top-level README.)
- `results/analysis/annotation_template_N10000.csv` — blank, 7 category columns + notes.
- `results/analysis/rq2_blind_key_N10000.csv` — de-anonymization key, explicitly marked "DO NOT VIEW BEFORE ANNOTATING", not used in any statistic in this document.

Each of the 100 cells was scored against seven fixed binary criteria before unblinding; see `results/analysis/annotation_blind_N10000.csv` for the completed annotation and `results/analysis/annotation_unblinded_statistics_N10000.csv` for the resulting per-pattern statistics.

## 7. Methodological limitations

- **top-K = 50 is a fixed, absolute count**, not scaled with N or with the size of the candidate/vote pool; it is used purely for readability of the images/figures, not as a principled cutoff.
- **Sampling is not uniform over the complete ImageNet-1k train set.** The N=10,000 pool is class-balanced sampling (10 images/class, 1000 classes) from the candidate pool found in 6 of 294 total ILSVRC/imagenet-1k train parquet shards (see `results/dataset_manifest_N10000.json`).
- **attn.proj and fc2 UnitMem is a magnitude-based adaptation** of the original metric (defined for fc1's post-GELU non-negative activations); abs() conflates inhibition and excitation for these two projection layers. Their layer-type and block numbers above are reported as computed, but this caveat applies to their interpretation.
- **High S_u / high vote count indicates instance-level activation selectivity — a memorization proxy — not causally verified storage of a specific training image.** No weight-intervention or data-extraction experiment has been performed in this project at any N.
- The Laplacian-sharpness and JPEG-bytes-per-pixel findings above are associations from a single 50-vs-50 comparison at one top-K and one scoring convention; they have not been cross-validated against other top-K values or scoring conventions.
