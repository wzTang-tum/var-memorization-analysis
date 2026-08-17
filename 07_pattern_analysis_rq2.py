"""
RQ2 pattern analysis: select the top-50 images by vote count
(06_rq2_vote_counting.py's rq2_image_votes_N10000.csv), build a 1:1
within-class matched control (seed=42), and compare on 8 objective
low-level image features -- reusing the already-computed
results/image_features.csv (no re-reading/recomputing image pixels).

Statistical method: class-composition-preserving 1:1-matched permutation
null, add-one Monte Carlo p-value correction, BH-FDR across 8 features.
No causal language.
"""
import csv
import os
import textwrap

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import common as C

SEED = 42
N_PERM = 10000
N_BOOT = 10000
TOP_K = 50
FEATURES = ['grayscale_entropy', 'edge_density', 'laplacian_sharpness', 'luminance_contrast',
            'mean_saturation', 'colorfulness', 'center_vs_border_contrast', 'jpeg_bytes_per_pixel']


def wrap(s, width=95):
    return '\n'.join(textwrap.wrap(s, width=width))


def bh_fdr(pvals):
    pvals = np.asarray(pvals)
    n = len(pvals)
    order = np.argsort(pvals)
    ranked = pvals[order]
    q = ranked * n / (np.arange(n) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0, 1)
    out = np.empty(n)
    out[order] = q
    return out


def cohens_d(a, b):
    na, nb = len(a), len(b)
    pooled_std = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    if pooled_std == 0:
        return 0.0
    return (a.mean() - b.mean()) / pooled_std


def main():
    lines = []

    def log(s=''):
        print(s)
        lines.append(s)

    rng = np.random.default_rng(SEED)

    log('=== RQ2 pattern analysis: top-50 by unit-scale vote count, class-matched control ===')

    # ---- load vote table (rank order already votes desc, image_id asc) ----
    vote_rows = []
    with open(os.path.join(C.RESULTS_ANALYSIS, 'rq2_image_votes_N10000.csv'), encoding='utf-8') as f:
        for row in csv.DictReader(f):
            vote_rows.append({'image_id': int(row['image_id']), 'class_id': int(row['class_id']),
                               'votes': int(row['total_unit_scale_votes'])})
    vote_rows.sort(key=lambda r: (-r['votes'], r['image_id']))

    boundary_votes = vote_rows[TOP_K - 1]['votes']
    n_tied_total = sum(1 for r in vote_rows if r['votes'] == boundary_votes)
    n_tied_selected = sum(1 for r in vote_rows[:TOP_K] if r['votes'] == boundary_votes)
    log(f'top-{TOP_K} boundary vote count = {boundary_votes}; {n_tied_total} images tied at this count, '
        f'{n_tied_selected} of them included in the top-{TOP_K} (tie-break: image_id ascending)')

    top50 = vote_rows[:TOP_K]
    top50_ids = np.array([r['image_id'] for r in top50])
    top50_votes = np.array([r['votes'] for r in top50])
    top8_ids = top50_ids[:8]
    top8_votes = top50_votes[:8]

    manifest = C.load_manifest()
    images = manifest['images']
    id_lookup = {im['image_id']: im for im in images}
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'meta',
                            'imagenet_class_index.json'), encoding='utf-8') as f:
        import json
        class_index = json.load(f)
    class_names = {int(kk): v[1].replace('_', ' ') for kk, v in class_index.items()}

    class_of_image = {im['image_id']: im['class_id'] for im in images}
    images_by_class = {}
    for im in images:
        images_by_class.setdefault(im['class_id'], []).append(im['image_id'])
    for c in images_by_class:
        assert len(images_by_class[c]) == 10

    top50_classes = [class_of_image[i] for i in top50_ids]
    class_k = {}
    for c in top50_classes:
        class_k[c] = class_k.get(c, 0) + 1
    log(f'\ntop-{TOP_K} spans {len(class_k)} distinct classes (of 1000 total)')
    log(f'per-class top-{TOP_K} counts (class_id: k): {class_k}')

    # ---- 1:1 matched control, seed=42, without replacement, report shortfalls ----
    matched_control_ids = []
    shortfall_log = []
    top50_id_set = set(top50_ids.tolist())
    for c, k_c in class_k.items():
        pool = images_by_class[c]
        non_top = [i for i in pool if i not in top50_id_set]
        n_avail = len(non_top)
        n_take = min(k_c, n_avail)
        if n_take < k_c:
            shortfall_log.append((c, k_c, n_avail))
        chosen = rng.choice(non_top, size=n_take, replace=False) if n_take > 0 else np.array([], dtype=int)
        matched_control_ids.extend(int(x) for x in chosen)
    matched_control_ids = np.array(matched_control_ids)
    log(f'\nmatched control (1:1, within-class, without replacement, seed={SEED}): '
        f'{len(matched_control_ids)} images (target {len(top50_ids)})')
    if shortfall_log:
        log(f'INSUFFICIENT CONTROL for {len(shortfall_log)} class(es): {shortfall_log}')
    else:
        log('no shortfalls: every class had enough non-top images for a full 1:1 match.')

    # ---- reuse pre-computed features (do not recompute pixels) ----
    feat_rows = {}
    feat_csv = os.path.join(C.RESULTS_ACTIVATIONS, 'image_features.csv')
    with open(feat_csv, encoding='utf-8') as f:
        reader = csv.DictReader(f)
        missing_cols = [ft for ft in FEATURES if ft not in reader.fieldnames]
        if missing_cols:
            raise RuntimeError(f'{feat_csv} is missing required columns {missing_cols} -- '
                                f'cannot proceed without recomputing (not attempted automatically).')
        for row in reader:
            feat_rows[int(row['image_id'])] = {ft: float(row[ft]) for ft in FEATURES}
    log(f'\nreused {len(feat_rows)} rows of precomputed features from {feat_csv} (no pixels re-read)')

    top_feat = {ft: np.array([feat_rows[i][ft] for i in top50_ids]) for ft in FEATURES}
    ctrl_feat = {ft: np.array([feat_rows[i][ft] for i in matched_control_ids]) for ft in FEATURES}

    # ---- class-composition-preserving matched permutation null ----
    class_list = list(class_k.keys())
    class_pools = {c: np.array(images_by_class[c]) for c in class_list}
    class_ks = {c: class_k[c] for c in class_list}

    perm_diffs = {ft: np.empty(N_PERM) for ft in FEATURES}
    perm_rng = np.random.default_rng(SEED + 1)
    for it in range(N_PERM):
        pseudo_top, pseudo_ctrl = [], []
        for c in class_list:
            pool = class_pools[c]
            k_c = class_ks[c]
            chosen = perm_rng.choice(pool, size=k_c, replace=False)
            chosen_set = set(chosen.tolist())
            rest = np.array([x for x in pool if x not in chosen_set])
            n_take = min(k_c, len(rest))
            matched = perm_rng.choice(rest, size=n_take, replace=False) if n_take > 0 else np.array([], dtype=int)
            pseudo_top.extend(chosen.tolist())
            pseudo_ctrl.extend(matched.tolist())
        pseudo_top = np.array(pseudo_top)
        pseudo_ctrl = np.array(pseudo_ctrl)
        for ft in FEATURES:
            vt = np.array([feat_rows[i][ft] for i in pseudo_top])
            vc = np.array([feat_rows[i][ft] for i in pseudo_ctrl])
            perm_diffs[ft][it] = vt.mean() - vc.mean()

    log('\n' + '=' * 100)
    log(f'{"feature":>25} {"top_mean":>10} {"ctrl_mean":>10} {"diff":>9} {"top_med":>9} {"ctrl_med":>9} '
        f'{"cohens_d":>9} {"CI_lo":>8} {"CI_hi":>8} {"perm_p":>9}')
    perm_p = {}
    for ft in FEATURES:
        vt, vc = top_feat[ft], ctrl_feat[ft]
        obs_diff = vt.mean() - vc.mean()
        d = cohens_d(vt, vc)

        boot_rng = np.random.default_rng(SEED + 2)
        boot_diffs = np.empty(N_BOOT)
        for b in range(N_BOOT):
            bt = boot_rng.choice(vt, size=len(vt), replace=True)
            bc = boot_rng.choice(vc, size=len(vc), replace=True)
            boot_diffs[b] = bt.mean() - bc.mean()
        lo, hi = np.percentile(boot_diffs, [2.5, 97.5])

        n_ge = int(np.sum(np.abs(perm_diffs[ft]) >= abs(obs_diff)))
        p = (n_ge + 1) / (N_PERM + 1)
        perm_p[ft] = p

        log(f'{ft:>25} {vt.mean():>10.4f} {vc.mean():>10.4f} {obs_diff:>+9.4f} {np.median(vt):>9.4f} '
            f'{np.median(vc):>9.4f} {d:>+9.3f} {lo:>+8.4f} {hi:>+8.4f} {p:>9.4f}')

    p_array = np.array([perm_p[ft] for ft in FEATURES])
    q_array = bh_fdr(p_array)
    log('\nBenjamini-Hochberg FDR-corrected q-values (across 8 features):')
    for ft, q in zip(FEATURES, q_array):
        sig = 'SIGNIFICANT (q<0.05)' if q < 0.05 else 'not significant'
        log(f'  {ft:>25}: q = {q:.4f}  -- {sig}')

    log('\n' + wrap(
        'Interpretation: tests whether the 50 images most frequently voted for by pooled top-10% units '
        '(S_u = mean(B_1..B_9)) differ from class-matched non-selected images on 8 objective low-level '
        'pixel statistics. A significant difference is an ASSOCIATION between vote frequency and image '
        'statistics -- it does not identify a specific visual pattern (needs human annotation) and does '
        'not establish that these statistics cause anything about unit selectivity.', width=95))

    # ---- figures ----
    def show_grid(ids, votes_, ncols, path, title, figsize):
        nrows = int(np.ceil(len(ids) / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
        axes = np.array(axes).reshape(nrows, ncols)
        for idx in range(nrows * ncols):
            ax = axes[idx // ncols, idx % ncols]
            if idx < len(ids):
                img_id = int(ids[idx])
                rec = id_lookup[img_id]
                with Image.open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                              rec['local_path'])) as im2:
                    ax.imshow(im2.convert('RGB'))
                cname = class_names[rec['class_id']]
                ax.set_title(f'#{votes_[idx]} votes\n{cname}', fontsize=7)
            ax.axis('off')
        fig.suptitle(title, fontsize=11)
        fig.tight_layout()
        fig.savefig(path, dpi=300)
        plt.close(fig)

    show_grid(top8_ids, top8_votes, 4, os.path.join(C.FIG_ANALYSIS, 'rq2_top8_images_N10000.png'),
              'top-8 most-voted images (S_u=mean(B_1..B_9) pooled top-10% units, N=10000)', (10, 6))
    show_grid(top50_ids, top50_votes, 10, os.path.join(C.FIG_ANALYSIS, 'rq2_top50_supplement_N10000.png'),
              'top-50 most-voted images (supplementary)', (20, 11))
    log(f'\nwrote figures/rq2_top8_images_N10000.png, rq2_top50_supplement_N10000.png '
        f'(not included in the public repository -- see NOTE below)')

    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    for ax, ft in zip(axes.flat, FEATURES):
        vt, vc = top_feat[ft], ctrl_feat[ft]
        bp = ax.boxplot([vt, vc], labels=['top-50', 'control'], patch_artist=True, widths=0.5)
        bp['boxes'][0].set_facecolor('#DD8452')
        bp['boxes'][1].set_facecolor('#4C72B0')
        ax.set_title(f'{ft}\nd={cohens_d(vt,vc):+.2f}, perm p={perm_p[ft]:.3f}', fontsize=8)
    fig.suptitle('RQ2 (N=10000, vote-based top-50): top-50 vs class-matched control, 8 image features', fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(C.FIG_ANALYSIS, 'rq2_top_vs_control_N10000.png'), dpi=300)
    plt.close(fig)
    log(f'wrote figures/rq2_top_vs_control_N10000.png')

    out_path = os.path.join(C.RESULTS_ANALYSIS, 'rq2_pattern_statistics_N10000.txt')
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'\nwrote {out_path}')

    id_out_path = os.path.join(C.RESULTS_ANALYSIS, 'rq2_top50_control_ids_N10000.csv')
    with open(id_out_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['image_id', 'group', 'class_id', 'vote_count'])
        for i, v in zip(top50_ids.tolist(), top50_votes.tolist()):
            w.writerow([i, 'top', class_of_image[i], v])
        for i in matched_control_ids.tolist():
            w.writerow([i, 'control', class_of_image[i], ''])
    print(f'wrote {id_out_path}')

    return top50_ids, top50_votes, matched_control_ids


if __name__ == '__main__':
    main()
