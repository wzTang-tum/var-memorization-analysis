"""
Horizontal grouped bar chart of the 7 blind-annotated semantic patterns
(top-50 vs. class-matched control-50), with bootstrap 95% CIs and BH-FDR
significance stars. Style matches report/make_fig2.py (DejaVu Sans,
#dd8452 top-50 / #4c72b0 control, thin 0.35-gray spines).

Supplementary figure only -- not inserted into the 2-page paper body.
"""
import csv
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import common as C

STATS_CSV = os.path.join(C.RESULTS_ANALYSIS, 'annotation_unblinded_statistics_N10000.csv')
OUT_PNG = os.path.join(C.FIG_ANALYSIS, 'rq2_blind_pattern_comparison_N10000.png')

LABELS = {
    'isolated_object_plain_background': 'Isolated object,\nplain background',
    'extreme_closeup_or_unusual_crop': 'Extreme closeup /\nunusual crop',
    'repetitive_texture_or_geometric_pattern': 'Repetitive texture /\ngeometric pattern',
    'unusual_color_or_high_contrast': 'Unusual color /\nhigh contrast',
    'blur_low_resolution_or_compression_artifact': 'Blur / low-res /\ncompression artifact',
    'watermark_or_visible_text': 'Watermark /\nvisible text',
    'unusual_pose_occlusion_or_atypical_composition': 'Unusual pose /\nocclusion / atypical comp.',
}

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 8,
    'axes.titlesize': 10,
    'axes.labelsize': 9,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8.5,
})


def main():
    rows = []
    with open(STATS_CSV, encoding='utf-8') as f:
        for r in csv.DictReader(f):
            rows.append(r)

    # order: most common in top-50 first (top of chart)
    rows.sort(key=lambda r: -float(r['top_proportion']))

    patterns = [r['pattern'] for r in rows]
    n = len(patterns)
    top_p = np.array([float(r['top_proportion']) for r in rows]) * 100
    top_lo = np.array([float(r['top_proportion_ci_lo']) for r in rows]) * 100
    top_hi = np.array([float(r['top_proportion_ci_hi']) for r in rows]) * 100
    ctrl_p = np.array([float(r['control_proportion']) for r in rows]) * 100
    ctrl_lo = np.array([float(r['control_proportion_ci_lo']) for r in rows]) * 100
    ctrl_hi = np.array([float(r['control_proportion_ci_hi']) for r in rows]) * 100
    sig = np.array([int(r['significant_q05']) for r in rows], dtype=bool)
    q = np.array([float(r['q_value_bh_fdr']) for r in rows])

    y = np.arange(n)[::-1]  # top row = first pattern (most common)
    h = 0.34

    fig, ax = plt.subplots(figsize=(7.2, 4.4))

    top_err = np.vstack([top_p - top_lo, top_hi - top_p])
    ctrl_err = np.vstack([ctrl_p - ctrl_lo, ctrl_hi - ctrl_p])

    bars_top = ax.barh(y + h / 2, top_p, height=h, color='#dd8452', label='Top-50',
                        xerr=top_err, error_kw=dict(elinewidth=0.9, capsize=2.5, ecolor='#7a4123'))
    bars_ctrl = ax.barh(y - h / 2, ctrl_p, height=h, color='#4c72b0', label='Class-matched control',
                         xerr=ctrl_err, error_kw=dict(elinewidth=0.9, capsize=2.5, ecolor='#2a4468'))

    ax.set_yticks(y)
    ax.set_yticklabels([LABELS[p] for p in patterns])
    ax.set_xlabel('% of images with pattern present')
    ax.set_xlim(0, 100)
    ax.grid(axis='x', color='0.88', linewidth=0.55)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_color('0.35')
        spine.set_linewidth(0.6)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # significance stars: place above the higher of the two CI uppers for that row
    for i in range(n):
        if sig[i]:
            x_star = max(top_hi[i], ctrl_hi[i]) + 3
            ax.text(x_star, y[i], f'*  q={q[i]:.3f}', va='center', ha='left',
                     fontsize=8, color='#333333', fontweight='bold')

    ax.set_title('Blind-annotated visual patterns: top-50 vs. class-matched controls (N=10000 pipeline)\n'
                  '* = significant after BH-FDR correction (q<0.05, 7 tests)', fontsize=9.5, pad=10)
    ax.legend(loc='lower right', frameon=False, fontsize=8)

    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=300, facecolor='white', bbox_inches='tight')
    plt.close(fig)
    print(f'wrote {OUT_PNG}')


if __name__ == '__main__':
    main()
