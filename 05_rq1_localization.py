"""
RQ1 post-processing: "Where are the highest-10% memorizing units located,
by layer type and block location?", using S_u = mean(UnitMem_B_1..B_9)
(the paper-method score), not the scale-averaged-activation or
token-count-weighted regimes.

Reads results/unitmem_N10000_global.npz (read-only) via common.py.
"""
import os
import textwrap

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import common as C

N_BLOCKS = C.N_BLOCKS
GROUPS = C.GROUPS
GROUP_WIDTH = C.GROUP_WIDTH


def wrap(s, width=100):
    return '\n'.join(textwrap.wrap(s, width=width))


def main():
    lines = []

    def log(s=''):
        print(s)
        lines.append(s)

    log('=== RQ1: layer type + block location of the highest-10% units, S_u = mean(B_1..B_9) ===')

    data = C.load_npz()
    for g in GROUPS:
        names = list(data[f'{g}_regime_names'])
        log(f'{g} regime_names = {names} -- only indices 3..11 (B_1..B_9) are read for scoring; '
            f'A (idx 0), T (idx 1), scale0 (idx 2) are never touched.')

    per_group, pool = C.build_pool(data)
    sel = C.select_top_fraction(pool, frac=0.10)
    k = sel['k']
    sel_idx = np.where(sel['sel_mask'])[0]
    assert len(sel_idx) == k

    sel_group = pool['group'][sel_idx]
    sel_block = pool['block'][sel_idx]
    sel_global_id = pool['global_id'][sel_idx]

    log(f'\npooled alive candidates = {sel["n_alive"]} (of raw {C.TOTAL_UNITS} = 65536+16384+16384)')
    for g in GROUPS:
        n_dead_g = int(per_group[g]['dead_any'].sum())
        n_alive_g = per_group[g]['U'] - n_dead_g
        log(f'  {g}: raw={per_group[g]["U"]}, dead-in-any-B_1..B_9={n_dead_g}, alive={n_alive_g}')
    log(f'k = floor(0.10 * {sel["n_alive"]}) = {k}')
    log(f'boundary S_u score = {sel["boundary_score"]:.6f}; units tied exactly at this score: '
        f'{sel["n_tied_at_boundary_total"]} total, {sel["n_tied_at_boundary_selected"]} of them selected '
        f'(tie-break: global fixed unit index, fc1=[0,65535] then attn_proj=[65536,81919] then '
        f'fc2=[81920,98303], ascending, applied only among exactly-tied scores)')

    # ============================================================
    # A. Layer type
    # ============================================================
    log('\n' + '=' * 90)
    log('A. LAYER TYPE')
    log('=' * 90)
    log(f'{"group":>10} {"raw_total":>10} {"dead":>6} {"alive_candidates":>17} {"selected":>9} '
        f'{"selected/alive":>15} {"selected/raw":>13} {"share_of_k":>11}')
    layer_type_rows = {}
    for g in GROUPS:
        raw_total = per_group[g]['U']
        n_dead_g = int(per_group[g]['dead_any'].sum())
        alive_g = raw_total - n_dead_g
        selected_g = int(np.sum(sel_group == g))
        ratio_alive = selected_g / alive_g
        ratio_raw = selected_g / raw_total
        share_k = selected_g / k
        layer_type_rows[g] = dict(raw_total=raw_total, dead=n_dead_g, alive=alive_g,
                                   selected=selected_g, ratio_alive=ratio_alive,
                                   ratio_raw=ratio_raw, share_k=share_k)
        log(f'{g:>10} {raw_total:>10} {n_dead_g:>6} {alive_g:>17} {selected_g:>9} '
            f'{ratio_alive:>15.4%} {ratio_raw:>13.4%} {share_k:>11.4%}')
    assert sum(v['selected'] for v in layer_type_rows.values()) == k

    # ============================================================
    # B. Block location
    # ============================================================
    log('\n' + '=' * 90)
    log('B. BLOCK LOCATION (pooled across all 3 layer types)')
    log('=' * 90)
    alive_block_counts = np.zeros(N_BLOCKS, dtype=int)
    for g in GROUPS:
        info = per_group[g]
        alive_mask = ~info['dead_any']
        for b in range(N_BLOCKS):
            alive_block_counts[b] += int(np.sum(alive_mask & (info['block_idx'] == b)))

    selected_block_counts = np.bincount(sel_block, minlength=N_BLOCKS)
    assert selected_block_counts.sum() == k

    log(f'{"block":>5} {"alive_candidates":>17} {"selected":>9} {"in-block_rate":>14} {"share_of_k":>11}')
    for b in range(N_BLOCKS):
        rate = selected_block_counts[b] / alive_block_counts[b] if alive_block_counts[b] > 0 else float('nan')
        share = selected_block_counts[b] / k
        log(f'{b:>5} {alive_block_counts[b]:>17} {selected_block_counts[b]:>9} {rate:>14.4%} {share:>11.4%}')

    top3_blocks = np.argsort(-selected_block_counts)[:3]
    top3_total = int(selected_block_counts[top3_blocks].sum())
    log(f'\ntop-3 blocks by selected count: {top3_blocks.tolist()} '
        f'(counts {selected_block_counts[top3_blocks].tolist()}), combined = {top3_total} '
        f'({top3_total/k:.4%} of k={k})')

    # ============================================================
    # C. Joint layer-type x block distribution
    # ============================================================
    log('\n' + '=' * 90)
    log('C. JOINT LAYER-TYPE x BLOCK DISTRIBUTION (16 blocks x 3 types)')
    log('=' * 90)
    joint_selected = np.zeros((N_BLOCKS, 3), dtype=int)
    joint_candidate = np.zeros((N_BLOCKS, 3), dtype=int)
    for gi, g in enumerate(GROUPS):
        info = per_group[g]
        alive_mask = ~info['dead_any']
        for b in range(N_BLOCKS):
            joint_candidate[b, gi] = int(np.sum(alive_mask & (info['block_idx'] == b)))
        sel_b_this_g = sel_block[sel_group == g]
        joint_selected[:, gi] = np.bincount(sel_b_this_g, minlength=N_BLOCKS)
    assert joint_selected.sum() == k
    joint_rate = np.divide(joint_selected, joint_candidate, out=np.full_like(joint_selected, np.nan, dtype=float),
                            where=joint_candidate > 0)

    log(f'{"block":>5} ' + ' '.join(f'{g:>28}' for g in GROUPS))
    log(f'{"":>5} ' + ' '.join(f'{"sel/cand(rate)":>28}' for _ in GROUPS))
    for b in range(N_BLOCKS):
        cells = []
        for gi, g in enumerate(GROUPS):
            cells.append(f'{joint_selected[b,gi]}/{joint_candidate[b,gi]}({joint_rate[b,gi]:.2%})')
        log(f'{b:>5} ' + ' '.join(f'{c:>28}' for c in cells))

    # ============================================================
    # fig1
    # ============================================================
    fig = plt.figure(figsize=(13, 6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.3])

    ax0 = fig.add_subplot(gs[0])
    ratios = [layer_type_rows[g]['ratio_alive'] for g in GROUPS]
    bars = ax0.bar(GROUPS, ratios, color=['#DD8452', '#4C72B0', '#55A868'])
    ax0.axhline(0.10, color='gray', linestyle='--', linewidth=1.2, label='10% (pool average)')
    ax0.set_ylabel('fraction of that type\'s alive units selected into pooled top-10%')
    ax0.set_title('A. layer-type selection ratio', fontsize=10)
    for bar, r in zip(bars, ratios):
        ax0.text(bar.get_x() + bar.get_width() / 2, r, f'{r:.1%}', ha='center', va='bottom', fontsize=9)
    ax0.legend(fontsize=8)

    ax1 = fig.add_subplot(gs[1])
    im = ax1.imshow(joint_rate, aspect='auto', cmap='viridis')
    ax1.set_xticks(range(3))
    ax1.set_xticklabels(GROUPS)
    ax1.set_yticks(range(N_BLOCKS))
    ax1.set_ylabel('block index')
    ax1.set_title('B. block x layer-type selection rate', fontsize=10)
    fig.colorbar(im, ax=ax1, label='selection rate (selected/alive candidates)', fraction=0.046)

    fig.suptitle(wrap('RQ1: pooled top-10% units by S_u=mean(UnitMem over scales B_1..B_9), N=10000, '
                       '1000 classes', width=100), fontsize=10)
    fig.tight_layout()
    fig_path = os.path.join(C.FIG_ANALYSIS, 'fig1_rq1_location_N10000.png')
    fig.savefig(fig_path, dpi=300)
    plt.close(fig)
    log(f'\nwrote {fig_path}')

    # persist selection for RQ2 (and for verification)
    out_npz = os.path.join(C.RESULTS_ANALYSIS, 'v4_selection.npz')
    np.savez_compressed(
        out_npz,
        k=k, n_alive=sel['n_alive'], boundary_score=sel['boundary_score'],
        sel_group=sel_group, sel_block=sel_block, sel_global_id=sel_global_id,
        sel_local_idx=pool['local_idx'][sel_idx], sel_S=pool['S'][sel_idx],
        joint_selected=joint_selected, joint_candidate=joint_candidate,
        alive_block_counts=alive_block_counts, selected_block_counts=selected_block_counts,
    )
    log(f'wrote {out_npz}')

    out_txt = os.path.join(C.RESULTS_ANALYSIS, 'rq1_report_N10000.txt')
    with open(out_txt, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'\nwrote {out_txt}')

    return per_group, pool, sel, layer_type_rows


if __name__ == '__main__':
    main()
