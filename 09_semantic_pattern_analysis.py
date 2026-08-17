"""
RQ2 semantic-pattern unblinding & statistics.

Merges the reviewed blind annotation (annotation_blind_N10000.csv,
cell_id-indexed, 7 binary visual-pattern fields) with the de-anonymization
key (rq2_blind_key_N10000.csv, cell_id -> image_id/group/class_id) and
tests each of the 7 patterns for enrichment in the top-50 vote-getters vs.
the 50 class-matched controls.

Test-selection rule:
  - rq2_blind_key_N10000.csv has no pair_id column (checked below) but does
    carry class_id -> use a class-composition-preserving (stratified)
    permutation test: group labels are shuffled only within each class
    stratum (the set of annotated images sharing that class_id), so the
    number of top/control images per class is preserved in every
    permutation, exactly mirroring how the real top/control split is itself
    class-matched. This uses only the 100 already-annotated images (no
    pixel/feature recomputation needed, unlike the continuous low-level
    features in 07_pattern_analysis_rq2.py, which could be resampled from
    the full per-class pool).
  - 95% CI on the risk difference (top proportion - control proportion) and
    on Cohen's h is obtained by ordinary bootstrap: resample the 50 top and
    50 control binary labels independently with replacement (same procedure,
    same seed offsets as 07_pattern_analysis_rq2.py's feature CIs).
  - BH-FDR correction is applied across the 7 patterns.
  - Fisher's exact test is not used because class_id (grouping information)
    is available in the blind key.
"""
import csv
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import common as C

SEED = 42
N_PERM = 10000
N_BOOT = 10000

PATTERNS = [
    'isolated_object_plain_background',
    'extreme_closeup_or_unusual_crop',
    'repetitive_texture_or_geometric_pattern',
    'unusual_color_or_high_contrast',
    'blur_low_resolution_or_compression_artifact',
    'watermark_or_visible_text',
    'unusual_pose_occlusion_or_atypical_composition',
]


def bh_fdr(pvals):
    pvals = np.asarray(pvals, dtype=float)
    n = len(pvals)
    order = np.argsort(pvals)
    ranked = pvals[order]
    q = ranked * n / (np.arange(n) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0, 1)
    out = np.empty(n)
    out[order] = q
    return out


def cohens_h(p1, p2):
    return 2 * np.arcsin(np.sqrt(np.clip(p1, 0, 1))) - 2 * np.arcsin(np.sqrt(np.clip(p2, 0, 1)))


def main():
    lines = []

    def log(s=''):
        print(s)
        lines.append(s)

    log('=== RQ2 semantic-pattern unblinding & statistics (N=100 blind-annotated cells) ===')

    ann_path = os.path.join(C.RESULTS_ANALYSIS, 'annotation_blind_N10000.csv')
    key_path = os.path.join(C.RESULTS_ANALYSIS, 'rq2_blind_key_N10000.csv')

    ann = pd.read_csv(ann_path)
    key = pd.read_csv(key_path, skiprows=1)  # first physical line is a "do not view" comment row

    log(f'\nloaded annotation file: {ann_path} ({len(ann)} rows)')
    log(f'loaded blind key: {key_path} ({len(key)} rows)')

    # ---------------------------------------------------------------
    # Section 1: data validation
    # ---------------------------------------------------------------
    log('\n' + '=' * 100)
    log('SECTION 1: DATA VALIDATION')
    log('=' * 100)

    checks = []

    ann_ids = sorted(ann['cell_id'].tolist())
    checks.append(('annotation cell_id count == 100', len(ann) == 100))
    checks.append(('annotation cell_id == 0..99, no duplicates', ann_ids == list(range(100))))

    key_ids = sorted(key['cell_id'].tolist())
    checks.append(('key cell_id count == 100', len(key) == 100))
    checks.append(('key cell_id == 0..99, no duplicates', key_ids == list(range(100))))

    merged = ann.merge(key, on='cell_id', how='inner', validate='one_to_one')
    checks.append(('every cell_id matched 1:1 between annotation and key', len(merged) == 100))

    grp_counts = merged['group'].value_counts().to_dict()
    checks.append((f"top/control counts == 50/50 (got {grp_counts})",
                    grp_counts.get('top') == 50 and grp_counts.get('control') == 50))

    for p in PATTERNS:
        vals = set(merged[p].unique().tolist())
        checks.append((f'{p}: values subset of {{0,1}} (got {sorted(vals)})', vals <= {0, 1}))

    has_pair_id = 'pair_id' in key.columns
    checks.append((f'blind key has pair_id column: {has_pair_id}', True))
    has_class_id = 'class_id' in key.columns
    checks.append((f'blind key has class_id column: {has_class_id}', True))
    other_cols = [c for c in key.columns if c not in ('cell_id', 'image_id', 'group', 'class_id')]
    checks.append((f'blind key extra columns beyond cell_id/image_id/group/class_id: {other_cols}', True))

    all_ok = True
    for desc, ok in checks:
        status = 'OK' if ok else 'FAIL'
        if not ok:
            all_ok = False
        log(f'  [{status}] {desc}')
    if not all_ok:
        raise RuntimeError('one or more validation checks failed -- see log above')

    log(f'\nTest-method decision: blind key has no pair_id and does have class_id '
        f'=> class-composition-preserving (stratified-by-class) permutation test, '
        f'not McNemar (no explicit pairing) and not plain Fisher exact (class info is available).')

    # per-class composition check (informational, not a hard requirement)
    cls_top = merged.loc[merged.group == 'top', 'class_id'].value_counts().sort_index()
    cls_ctrl = merged.loc[merged.group == 'control', 'class_id'].value_counts().sort_index()
    same_support = set(cls_top.index) == set(cls_ctrl.index)
    exact_1to1 = same_support and (cls_top.sort_index() == cls_ctrl.reindex(cls_top.index).sort_index()).all()
    log(f'\nclass strata: {merged["class_id"].nunique()} distinct class_id values among the 100 annotated images')
    log(f'top and control classes identical (same class_id support): {same_support}')
    log(f'per-class top count == per-class control count for every class (fully balanced strata): {exact_1to1}')

    # ---------------------------------------------------------------
    # Section 2: per-pattern statistics
    # ---------------------------------------------------------------
    log('\n' + '=' * 100)
    log('SECTION 2: PER-PATTERN STATISTICS (top-50 vs. class-matched control-50)')
    log('=' * 100)

    top_df = merged[merged.group == 'top'].reset_index(drop=True)
    ctrl_df = merged[merged.group == 'control'].reset_index(drop=True)
    n_top, n_ctrl = len(top_df), len(ctrl_df)

    # class -> list of row-indices (within merged) for stratified permutation
    merged_reset = merged.reset_index(drop=True)
    class_to_rows = {}
    for idx, row in merged_reset.iterrows():
        class_to_rows.setdefault(row['class_id'], []).append(idx)
    group_arr = merged_reset['group'].values.copy()

    perm_rng = np.random.default_rng(SEED + 1)
    # pre-generate the 10000 permuted group-label arrays once (class-stratified shuffle),
    # reused across all 7 patterns (identical null design for every feature, as in
    # 07_pattern_analysis_rq2.py)
    strata = list(class_to_rows.values())
    perm_group_labels = np.empty((N_PERM, len(merged_reset)), dtype=object)
    for it in range(N_PERM):
        new_labels = group_arr.copy()
        for idxs in strata:
            if len(idxs) < 2:
                continue  # a singleton stratum has only one possible arrangement
            idxs = np.array(idxs)
            local = group_arr[idxs].copy()
            perm_rng.shuffle(local)
            new_labels[idxs] = local
        perm_group_labels[it] = new_labels

    results = {}
    log(f'\n{"pattern":>48} {"top_n1":>7} {"top_p":>7} {"ctrl_n1":>8} {"ctrl_p":>7} '
        f'{"diff_pp":>8} {"h":>7} {"h_CIlo":>8} {"h_CIhi":>8} {"perm_p":>8}')

    for p in PATTERNS:
        vt = top_df[p].values.astype(int)
        vc = ctrl_df[p].values.astype(int)
        n1_top, n1_ctrl = int(vt.sum()), int(vc.sum())
        p_top, p_ctrl = n1_top / n_top, n1_ctrl / n_ctrl
        diff = p_top - p_ctrl
        h = cohens_h(p_top, p_ctrl)

        # bootstrap CI (independent resampling within each group)
        boot_rng = np.random.default_rng(SEED + 2)
        boot_diff = np.empty(N_BOOT)
        boot_h = np.empty(N_BOOT)
        boot_p_top = np.empty(N_BOOT)
        boot_p_ctrl = np.empty(N_BOOT)
        for b in range(N_BOOT):
            bt = boot_rng.choice(vt, size=n_top, replace=True)
            bc = boot_rng.choice(vc, size=n_ctrl, replace=True)
            pbt, pbc = bt.mean(), bc.mean()
            boot_diff[b] = pbt - pbc
            boot_h[b] = cohens_h(pbt, pbc)
            boot_p_top[b] = pbt
            boot_p_ctrl[b] = pbc
        diff_lo, diff_hi = np.percentile(boot_diff, [2.5, 97.5])
        h_lo, h_hi = np.percentile(boot_h, [2.5, 97.5])
        p_top_lo, p_top_hi = np.percentile(boot_p_top, [2.5, 97.5])
        p_ctrl_lo, p_ctrl_hi = np.percentile(boot_p_ctrl, [2.5, 97.5])

        # class-stratified permutation null for the risk difference
        col = merged_reset[p].values.astype(int)
        perm_diffs = np.empty(N_PERM)
        for it in range(N_PERM):
            lbl = perm_group_labels[it]
            pt = col[lbl == 'top'].mean()
            pc = col[lbl == 'control'].mean()
            perm_diffs[it] = pt - pc
        n_ge = int(np.sum(np.abs(perm_diffs) >= abs(diff) - 1e-12))
        perm_p = (n_ge + 1) / (N_PERM + 1)

        table = np.array([[n1_top, n_top - n1_top], [n1_ctrl, n_ctrl - n1_ctrl]])

        results[p] = dict(n_top=n_top, n1_top=n1_top, p_top=p_top,
                           n_ctrl=n_ctrl, n1_ctrl=n1_ctrl, p_ctrl=p_ctrl,
                           p_top_lo=p_top_lo, p_top_hi=p_top_hi,
                           p_ctrl_lo=p_ctrl_lo, p_ctrl_hi=p_ctrl_hi,
                           diff_pp=diff * 100, diff_lo=diff_lo * 100, diff_hi=diff_hi * 100,
                           h=h, h_lo=h_lo, h_hi=h_hi,
                           perm_p=perm_p, table=table)

        log(f'{p:>48} {n1_top:>7d} {p_top:>7.3f} {n1_ctrl:>8d} {p_ctrl:>7.3f} '
            f'{diff*100:>+8.1f} {h:>+7.3f} {h_lo:>+8.3f} {h_hi:>+8.3f} {perm_p:>8.4f}')

    p_array = np.array([results[p]['perm_p'] for p in PATTERNS])
    q_array = bh_fdr(p_array)
    for p, q in zip(PATTERNS, q_array):
        results[p]['q'] = q

    log('\nBenjamini-Hochberg FDR-corrected q-values (across 7 patterns):')
    for p in PATTERNS:
        r = results[p]
        sig = 'SIGNIFICANT (q<0.05)' if r['q'] < 0.05 else 'not significant'
        log(f'  {p:>48}: q = {r["q"]:.4f}  -- {sig}')

    # ---------------------------------------------------------------
    # Section 3: most-common vs. significantly-enriched
    # ---------------------------------------------------------------
    log('\n' + '=' * 100)
    log('SECTION 3: MOST COMMON IN TOP-50 vs. SIGNIFICANTLY ENRICHED vs. CONTROL')
    log('=' * 100)

    by_top_freq = sorted(PATTERNS, key=lambda p: -results[p]['p_top'])
    log('\nRanked by prevalence within the top-50 (most to least common):')
    for p in by_top_freq:
        r = results[p]
        log(f'  {p:>48}: top={r["p_top"]*100:5.1f}%  control={r["p_ctrl"]*100:5.1f}%  q={r["q"]:.4f}')

    sig_patterns = [p for p in PATTERNS if results[p]['q'] < 0.05]
    log(f'\nPatterns significantly enriched vs. class-matched control after BH-FDR (q<0.05): '
        f'{sig_patterns if sig_patterns else "NONE"}')

    common_not_enriched = [p for p in by_top_freq[:3] if p not in sig_patterns]
    if common_not_enriched:
        log('\nCommon-but-not-enriched (frequent in top-50, but q>=0.05 vs. control -- '
            'i.e. common among frequently-selected images in general, not specific to high UnitMem):')
        for p in common_not_enriched:
            r = results[p]
            log(f'  {p}: top={r["p_top"]*100:.1f}%, control={r["p_ctrl"]*100:.1f}%, '
                f'diff={r["diff_pp"]:+.1f}pp, q={r["q"]:.4f}')

    log('\n' + '\n'.join([
        "Interpretation: this tests whether the 50 images most frequently selected by pooled top-10%",
        "units differ from class-matched non-selected images on 7 predefined binary visual/compositional",
        "patterns, scored blind (group identity and vote counts hidden) and reviewed by the author",
        "before unblinding. A significant q-value is an ASSOCIATION between vote frequency and a visual",
        "pattern's prevalence -- it does not establish that the pattern causes unit selectivity.",
    ]))

    # ---------------------------------------------------------------
    # write outputs
    # ---------------------------------------------------------------
    stats_csv = os.path.join(C.RESULTS_ANALYSIS, 'annotation_unblinded_statistics_N10000.csv')
    with open(stats_csv, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['pattern', 'top_n', 'top_n_positive', 'top_proportion',
                    'top_proportion_ci_lo', 'top_proportion_ci_hi',
                    'control_n', 'control_n_positive', 'control_proportion',
                    'control_proportion_ci_lo', 'control_proportion_ci_hi',
                    'diff_percentage_points', 'diff_ci_lo_pp', 'diff_ci_hi_pp',
                    'cohens_h', 'cohens_h_ci_lo', 'cohens_h_ci_hi',
                    'contingency_top_pos', 'contingency_top_neg',
                    'contingency_ctrl_pos', 'contingency_ctrl_neg',
                    'test_method', 'p_value_raw_two_sided', 'q_value_bh_fdr', 'significant_q05'])
        for p in PATTERNS:
            r = results[p]
            w.writerow([p, r['n_top'], r['n1_top'], f'{r["p_top"]:.6f}',
                        f'{r["p_top_lo"]:.6f}', f'{r["p_top_hi"]:.6f}',
                        r['n_ctrl'], r['n1_ctrl'], f'{r["p_ctrl"]:.6f}',
                        f'{r["p_ctrl_lo"]:.6f}', f'{r["p_ctrl_hi"]:.6f}',
                        f'{r["diff_pp"]:.4f}', f'{r["diff_lo"]:.4f}', f'{r["diff_hi"]:.4f}',
                        f'{r["h"]:.4f}', f'{r["h_lo"]:.4f}', f'{r["h_hi"]:.4f}',
                        r['table'][0, 0], r['table'][0, 1], r['table'][1, 0], r['table'][1, 1],
                        'class-stratified permutation (10000 perms, seed=42); CI: independent bootstrap (10000, seed=44)',
                        f'{r["perm_p"]:.6f}', f'{r["q"]:.6f}', int(r['q'] < 0.05)])
    log(f'\nwrote {stats_csv}')

    out_txt = os.path.join(C.RESULTS_ANALYSIS, 'annotation_unblinded_statistics_log_N10000.txt')
    with open(out_txt, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    log(f'wrote {out_txt}')

    return results, merged_reset, top_df, ctrl_df, by_top_freq, sig_patterns


if __name__ == '__main__':
    main()
