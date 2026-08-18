"""
Build the RQ2 feature-comparison figure (class-matched feature differences,
top-50 vs. control).

This reproduces only the statistical panel of low-level image features. A
companion "most-voted images" panel is intentionally not reproduced here
because it would render actual ImageNet training-image thumbnails, which
the dataset's license does not permit redistributing; see README.md.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results" / "analysis"
FEATURES = ROOT / "results" / "image_features.csv"
OUTPUT = ROOT / "results" / "figures" / "fig2_rq2.png"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


feature_rows = read_csv(FEATURES)
features_by_id = {int(row["image_id"]): row for row in feature_rows}
membership = read_csv(RESULTS / "rq2_top50_control_ids_N10000.csv")
top_ids = [int(row["image_id"]) for row in membership if row["group"] == "top"]
control_ids = [int(row["image_id"]) for row in membership if row["group"] == "control"]

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 7,
    }
)

fig, axes = plt.subplots(1, 2, figsize=(4.4, 2.75), constrained_layout=False)
fig.subplots_adjust(left=0.09, right=0.98, bottom=0.15, top=0.82, wspace=0.42)
fig.text(0.53, 0.95, "Class-matched feature differences", ha="center", va="center", fontsize=10)


def values(ids: list[int], column: str) -> np.ndarray:
    return np.asarray([float(features_by_id[image_id][column]) for image_id in ids])


feature_specs = [
    ("laplacian_sharpness", "Laplacian sharpness", "d=0.64, q=0.0008"),
    ("jpeg_bytes_per_pixel", "JPEG bytes / pixel", "d=0.43, q=0.0072"),
]

for ax, (column, label, statistic) in zip(axes, feature_specs):
    top_values = values(top_ids, column)
    control_values = values(control_ids, column)
    parts = ax.boxplot(
        [top_values, control_values],
        labels=["Top-50", "Control"],
        widths=0.58,
        patch_artist=True,
        medianprops={"color": "#333333", "linewidth": 1.1},
        boxprops={"linewidth": 0.8},
        whiskerprops={"linewidth": 0.8},
        capprops={"linewidth": 0.8},
        flierprops={"markersize": 2.5, "markeredgewidth": 0.5},
    )
    parts["boxes"][0].set_facecolor("#dd8452")
    parts["boxes"][1].set_facecolor("#4c72b0")
    ax.set_title(f"{label}\n{statistic}", pad=5)
    ax.grid(axis="y", color="0.88", linewidth=0.55)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_color("0.35")
        spine.set_linewidth(0.6)

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUTPUT, dpi=320, bbox_inches="tight", facecolor="white")
print(OUTPUT)
