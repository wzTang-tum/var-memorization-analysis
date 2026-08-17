"""Build the compact RQ1 localization figure used in the two-page report."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
DATA = np.load(ROOT / "results" / "analysis" / "v4_selection.npz")
OUTPUT = ROOT / "report" / "fig1_rq1.png"

groups = ["fc1", "attn.proj", "fc2"]
candidate_totals = np.array([65_536, 16_384, 16_384])
selected_totals = np.array([(DATA["sel_group"] == name.replace(".", "_")).sum() for name in groups])
# Stored group names use attn_proj; fc1 and fc2 are unchanged.
selected_totals[1] = (DATA["sel_group"] == "attn_proj").sum()
selection_rates = selected_totals / candidate_totals
joint_rates = DATA["joint_selected"] / DATA["joint_candidate"]

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

fig, (ax_bar, ax_heat) = plt.subplots(
    1,
    2,
    figsize=(7.2, 2.4),
    gridspec_kw={"width_ratios": [0.9, 1.55], "wspace": 0.34},
)

colors = ["#dd8452", "#4c72b0", "#55a868"]
bars = ax_bar.bar(groups, selection_rates * 100, color=colors, width=0.68)
ax_bar.axhline(10, color="0.45", linestyle="--", linewidth=0.9, label="Pooled top 10%")
ax_bar.set_title("A. Selection rate by layer type", pad=5)
ax_bar.set_ylabel("Units entering pooled top 10% (%)")
ax_bar.set_ylim(0, 15.5)
ax_bar.grid(axis="y", color="0.88", linewidth=0.55)
ax_bar.set_axisbelow(True)
ax_bar.legend(loc="upper right", frameon=False, fontsize=7)
for bar, value in zip(bars, selection_rates):
    ax_bar.text(bar.get_x() + bar.get_width() / 2, value * 100 + 0.35, f"{value:.1%}", ha="center", va="bottom")

image = ax_heat.imshow(joint_rates, aspect="auto", cmap="viridis", vmin=0, vmax=0.46)
ax_heat.set_title("B. Selection rate by block and layer type", pad=5)
ax_heat.set_xlabel("Layer type")
ax_heat.set_ylabel("Block")
ax_heat.set_xticks(range(3), groups)
ax_heat.set_yticks(range(16), range(16))
colorbar = fig.colorbar(image, ax=ax_heat, fraction=0.045, pad=0.025)
colorbar.set_label("Selected units / candidates")

for axis in (ax_bar, ax_heat):
    for spine in axis.spines.values():
        spine.set_color("0.35")
        spine.set_linewidth(0.6)

fig.subplots_adjust(left=0.065, right=0.965, bottom=0.16, top=0.88)
fig.savefig(OUTPUT, dpi=320, bbox_inches="tight", facecolor="white")
print(OUTPUT)
