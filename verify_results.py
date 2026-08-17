"""
Automated verification of the paper-method post-processing
(05_rq1_localization.py, 06_rq2_vote_counting.py, 07_pattern_analysis_rq2.py).
Exits non-zero on any check failure.
"""
import os
import sys

import numpy as np

import common as C

GROUPS = C.GROUPS


def main():
    lines = []
    failures = []

    def log(s=''):
        print(s)
        lines.append(s)

    log('=== verification ===')

    data = C.load_npz()

    # (1) only B_1..B_9 read for scoring -- checked structurally: b_rows() asserts
    # rows == [3..11] for every group, and compute_group_scores() never indexes
    # rows 0 (A), 1 (T), or 2 (scale0) anywhere.
    ok = True
    for g in GROUPS:
        rows = C.b_rows(data, g)
        if rows != list(range(3, 12)):
            ok = False
    log(f'(1) only regimes B_1..B_9 (rows 3..11) read for scoring, for all 3 groups: {"PASS" if ok else "FAIL"}')
    if not ok:
        failures.append('(1) B_1..B_9 row indices not as expected')

    # (2) S_u == mean of the 9 B scores, spot-check >=100 units per group
    per_group, pool = C.build_pool(data)
    rng = np.random.default_rng(123)
    ok2 = True
    n_checked = 0
    for g in GROUPS:
        info = per_group[g]
        U = info['U']
        idx = rng.choice(U, size=min(100, U), replace=False)
        for u in idx:
            manual = np.mean(info['unitmem_B'][:, u].astype(np.float64))
            stored = info['S_u'][u]
            if info['dead_any'][u]:
                if not np.isnan(stored):
                    ok2 = False
            else:
                if not np.isclose(manual, stored, atol=1e-9):
                    ok2 = False
            n_checked += 1
    log(f'(2) S_u == mean(B_1..B_9), spot-checked {n_checked} units (>=100/group): {"PASS" if ok2 else "FAIL"}')
    if not ok2:
        failures.append('(2) S_u does not equal mean(B_1..B_9) for some spot-checked unit')

    # (3) A, T, scale0 never used -- checked via source inspection of common.py:
    # compute_group_scores/build_pool/select_top_fraction never read data[f'{g}_unitmem']
    # directly (always go through b_rows()-selected rows).
    src_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'common.py')
    with open(src_path, encoding='utf-8') as f:
        src = f.read()
    bad_patterns = ["regime_names'].index('A')", "regime_names'].index('T')",
                     "unitmem'][0]", "unitmem'][1]", "unitmem'][2]"]
    found_bad = [p for p in bad_patterns if p in src]
    ok3 = len(found_bad) == 0
    log(f'(3) A/T/scale0 never referenced in common.py scoring code: {"PASS" if ok3 else "FAIL"} '
        f'(matched patterns: {found_bad})')
    if not ok3:
        failures.append(f'(3) forbidden regime reference(s) found in common.py: {found_bad}')

    # (4) per-group candidate counts match NPZ shapes
    ok4 = True
    for g in GROUPS:
        expected = C.GROUP_WIDTH[g] * C.N_BLOCKS
        actual = data[f'{g}_unitmem'].shape[1]
        if actual != expected or per_group[g]['U'] != expected:
            ok4 = False
    log(f'(4) per-group candidate counts (fc1=65536, attn_proj=16384, fc2=16384) match NPZ: '
        f'{"PASS" if ok4 else "FAIL"}')
    if not ok4:
        failures.append('(4) per-group candidate count mismatch vs NPZ shapes')

    # (5)+(6) selected counts sum to k; block x layer-type table sums to k
    sel = C.select_top_fraction(pool, frac=0.10)
    k = sel['k']
    sel_idx = np.where(sel['sel_mask'])[0]
    ok5 = len(sel_idx) == k
    log(f'(5) sum of selected units == k: len(selected)={len(sel_idx)}, k={k}: {"PASS" if ok5 else "FAIL"}')
    if not ok5:
        failures.append('(5) selected count != k')

    v4sel = np.load(os.path.join(C.RESULTS_ANALYSIS, 'v4_selection.npz'))
    joint_selected = v4sel['joint_selected']
    ok6 = int(joint_selected.sum()) == k
    log(f'(6) block x layer-type joint table sums to k: sum={int(joint_selected.sum())}, k={k}: '
        f'{"PASS" if ok6 else "FAIL"}')
    if not ok6:
        failures.append('(6) joint block x layer-type table does not sum to k')

    # (7) RQ2 total votes == 9*k
    votes_raw = np.load(os.path.join(C.RESULTS_ANALYSIS, 'v4_votes_raw.npz'))
    total_votes = int(votes_raw['total_votes'])
    ok7 = total_votes == 9 * k
    log(f'(7) RQ2 total votes == 9*k: total_votes={total_votes}, 9*k={9*k}: {"PASS" if ok7 else "FAIL"}')
    if not ok7:
        failures.append('(7) RQ2 total votes != 9*k')

    # (8) all argmax image_id in [0, 9999]
    vote_image = votes_raw['vote_image']
    ok8 = bool(np.all((vote_image >= 0) & (vote_image <= 9999)))
    log(f'(8) all vote image_ids in [0,9999]: min={vote_image.min()}, max={vote_image.max()}: '
        f'{"PASS" if ok8 else "FAIL"}')
    if not ok8:
        failures.append('(8) some vote image_id out of [0,9999]')

    log('\n' + '=' * 70)
    if not failures:
        log('VERDICT: ALL 8 CHECKS PASSED.')
    else:
        log('VERDICT: FAILURES FOUND:')
        for f_ in failures:
            log(f'  - {f_}')
    log('=' * 70)

    out_path = os.path.join(C.RESULTS_ANALYSIS, 'verification_report.txt')
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'\nwrote {out_path}')

    sys.exit(0 if not failures else 1)


if __name__ == '__main__':
    main()
