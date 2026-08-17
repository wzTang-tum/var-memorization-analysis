"""
RQ2 vote counting: for every unit selected in 05_rq1_localization.py's
pooled top-10% (S_u = mean(B_1..B_9)) and every scale s in B_1..B_9, one
vote goes to image_id = arg_max[B_s row, unit]. Total votes = 9 * k.

Reads results/analysis/v4_selection.npz (from 05_rq1_localization.py) and
results/unitmem_N10000_global.npz (read-only, for arg_max).
"""
import csv
import os

import numpy as np

import common as C

GROUPS = C.GROUPS
N_SCALES = 9


def main():
    lines = []

    def log(s=''):
        print(s)
        lines.append(s)

    log('=== RQ2: unit-scale vote counting (S_u=mean(B_1..B_9) selected units, N=10000) ===')

    data = C.load_npz()
    manifest = C.load_manifest()
    class_of_image = np.array([im['class_id'] for im in manifest['images']])
    assert len(class_of_image) == 10000

    sel = np.load(os.path.join(C.RESULTS_ANALYSIS, 'v4_selection.npz'))
    k = int(sel['k'])
    sel_group = sel['sel_group']
    sel_local_idx = sel['sel_local_idx']
    log(f'k = {k} selected units (from 05_rq1_localization.py); expected total votes = 9 * {k} = {9*k}')

    # rebuild arg_max_B per group (only B_1..B_9 rows read)
    arg_max_B = {}
    for g in GROUPS:
        rows = C.b_rows(data, g)
        arg_max_B[g] = data[f'{g}_arg_max'][rows]  # [9, U_g]

    # per-unit block index, for coverage stats
    block_of = {}
    for g in GROUPS:
        block_of[g] = np.arange(C.GROUP_WIDTH[g] * C.N_BLOCKS) // C.GROUP_WIDTH[g]

    vote_image = np.zeros(9 * k, dtype=np.int64)
    vote_block = np.zeros(9 * k, dtype=np.int64)
    vote_group = np.empty(9 * k, dtype=object)
    vote_scale = np.zeros(9 * k, dtype=np.int64)
    vote_unit_global_id = np.zeros(9 * k, dtype=np.int64)

    pos = 0
    for i in range(k):
        g = sel_group[i]
        local = int(sel_local_idx[i])
        b = int(block_of[g][local])
        gid = int(sel['sel_global_id'][i])
        for s in range(N_SCALES):  # scale index 0..8 -> label 1..9
            img_id = int(arg_max_B[g][s, local])
            vote_image[pos] = img_id
            vote_block[pos] = b
            vote_group[pos] = g
            vote_scale[pos] = s + 1
            vote_unit_global_id[pos] = gid
            pos += 1
    assert pos == 9 * k
    log(f'total votes cast = {pos} (verified == 9*k)')
    assert np.all((vote_image >= 0) & (vote_image <= 9999)), 'vote image_id out of [0,9999]!'
    log('all vote image_ids verified in [0, 9999]')

    # ---- per-image aggregation ----
    per_image = {}
    for idx in range(9 * k):
        img = int(vote_image[idx])
        if img not in per_image:
            per_image[img] = {'total_votes': 0, 'units': set(), 'blocks': set(), 'groups': set(), 'scales': set()}
        rec = per_image[img]
        rec['total_votes'] += 1
        rec['units'].add(int(vote_unit_global_id[idx]))
        rec['blocks'].add(int(vote_block[idx]))
        rec['groups'].add(str(vote_group[idx]))
        rec['scales'].add(int(vote_scale[idx]))

    rows = []
    for img_id, rec in per_image.items():
        rows.append({
            'image_id': img_id,
            'class_id': int(class_of_image[img_id]),
            'total_unit_scale_votes': rec['total_votes'],
            'distinct_selected_units': len(rec['units']),
            'distinct_blocks': len(rec['blocks']),
            'distinct_layer_types': len(rec['groups']),
            'distinct_scales': len(rec['scales']),
        })
    # sort: votes desc, image_id asc (documented, deterministic tie-break)
    rows.sort(key=lambda r: (-r['total_unit_scale_votes'], r['image_id']))

    out_csv = os.path.join(C.RESULTS_ANALYSIS, 'rq2_image_votes_N10000.csv')
    with open(out_csv, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['image_id', 'class_id', 'total_unit_scale_votes',
                                           'distinct_selected_units', 'distinct_blocks',
                                           'distinct_layer_types', 'distinct_scales'])
        w.writeheader()
        w.writerows(rows)
    log(f'\nwrote {out_csv} ({len(rows)} distinct images received >=1 vote)')

    n_with_votes = len(rows)
    log(f'\nimages with >=1 vote: {n_with_votes} / 10000 ({n_with_votes/10000:.4%})')

    total_votes = 9 * k
    top1 = rows[0]
    top10 = rows[:10]
    top50 = rows[:50]
    top10_votes = sum(r['total_unit_scale_votes'] for r in top10)
    top50_votes = sum(r['total_unit_scale_votes'] for r in top50)
    log(f'\ntop-1 image: image_id={top1["image_id"]}, class_id={top1["class_id"]}, '
        f'votes={top1["total_unit_scale_votes"]} ({top1["total_unit_scale_votes"]/total_votes:.4%} of total)')
    log(f'top-10 images: combined votes = {top10_votes} ({top10_votes/total_votes:.4%} of total {total_votes})')
    log(f'top-50 images: combined votes = {top50_votes} ({top50_votes/total_votes:.4%} of total {total_votes})')

    log(f'\n{"rank":>5} {"image_id":>9} {"class_id":>9} {"votes":>7} {"distinct_units":>15} '
        f'{"distinct_blocks":>16} {"distinct_layer_types":>21} {"distinct_scales":>16}')
    for rank, r in enumerate(top10, 1):
        log(f'{rank:>5} {r["image_id"]:>9} {r["class_id"]:>9} {r["total_unit_scale_votes"]:>7} '
            f'{r["distinct_selected_units"]:>15} {r["distinct_blocks"]:>16} '
            f'{r["distinct_layer_types"]:>21} {r["distinct_scales"]:>16}')

    # concentration description (numbers only, no causal language)
    votes_sorted = np.array([r['total_unit_scale_votes'] for r in rows])
    cum = np.cumsum(votes_sorted) / total_votes
    n_images_for_50pct = int(np.searchsorted(cum, 0.5) + 1)
    log(f'\nconcentration: {n_images_for_50pct} images ({n_images_for_50pct/n_with_votes:.4%} of the '
        f'{n_with_votes} voted-for images, {n_images_for_50pct/10000:.4%} of all 10000) account for '
        f'>=50% of total votes ({total_votes}).')
    log(f'the remaining {total_votes - votes_sorted[:n_images_for_50pct].sum()} votes are spread across '
        f'{n_with_votes - n_images_for_50pct} other images.')

    out_txt = os.path.join(C.RESULTS_ANALYSIS, 'rq2_vote_distribution_N10000.txt')
    with open(out_txt, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'\nwrote {out_txt}')

    # persist raw vote arrays for the verification script
    np.savez_compressed(os.path.join(C.RESULTS_ANALYSIS, 'v4_votes_raw.npz'),
                         vote_image=vote_image, k=k, total_votes=total_votes)


if __name__ == '__main__':
    main()
