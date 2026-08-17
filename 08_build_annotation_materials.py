"""
Blind human-annotation materials for the vote-based top-50 / matched-control
set from 07_pattern_analysis_rq2.py: an identity-hiding contact sheet, a
blank annotation template, and a separately-saved answer key explicitly
marked "do not view before annotating".

This script only builds the materials for a human to later annotate; it
does not itself claim any common visual pattern exists.
"""
import csv
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import common as C

SEED = 42
CATEGORIES = [
    'isolated_object_plain_background',
    'extreme_closeup_or_unusual_crop',
    'repetitive_texture_or_geometric_pattern',
    'unusual_color_or_high_contrast',
    'blur_low_resolution_or_compression_artifact',
    'watermark_or_visible_text',
    'unusual_pose_occlusion_or_atypical_composition',
]


def main():
    manifest = C.load_manifest()
    id_lookup = {im['image_id']: im for im in manifest['images']}

    ids_csv = os.path.join(C.RESULTS_ANALYSIS, 'rq2_top50_control_ids_N10000.csv')
    top_ids, ctrl_ids = [], []
    with open(ids_csv, encoding='utf-8') as f:
        for row in csv.DictReader(f):
            (top_ids if row['group'] == 'top' else ctrl_ids).append(int(row['image_id']))
    assert len(top_ids) == 50 and len(ctrl_ids) == 50, f'expected 50+50, got {len(top_ids)}+{len(ctrl_ids)}'

    all_ids = top_ids + ctrl_ids
    all_group = ['top'] * 50 + ['control'] * 50
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(100)
    shuffled_ids = [all_ids[i] for i in perm]
    shuffled_group = [all_group[i] for i in perm]
    cell_ids = list(range(100))  # anonymous sequential cell id, no info about group/class

    # ---- contact sheet: ONLY cell_id shown, no group/class/vote info ----
    ncols = 10
    nrows = 10
    fig, axes = plt.subplots(nrows, ncols, figsize=(20, 21))
    axes = np.array(axes).reshape(nrows, ncols)
    for idx in range(100):
        ax = axes[idx // ncols, idx % ncols]
        img_id = shuffled_ids[idx]
        rec = id_lookup[img_id]
        with Image.open(os.path.join(os.path.dirname(os.path.abspath(__file__)), rec['local_path'])) as im:
            ax.imshow(im.convert('RGB'))
        ax.set_title(f'cell {cell_ids[idx]}', fontsize=8)
        ax.axis('off')
    fig.suptitle('blind contact sheet (N=10000, vote-based top-50 + matched controls, shuffled, '
                 'identity hidden)', fontsize=12)
    fig.tight_layout()
    sheet_path = os.path.join(C.FIG_ANALYSIS, 'rq2_blind_contact_sheet_N10000.png')
    fig.savefig(sheet_path, dpi=300)
    plt.close(fig)
    print(f'wrote {sheet_path} (not included in the public repository -- contains ImageNet thumbnails)')

    # ---- blank annotation template (no labels filled in) ----
    template_path = os.path.join(C.RESULTS_ANALYSIS, 'annotation_template_N10000.csv')
    with open(template_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['cell_id'] + CATEGORIES + ['notes'])
        for cid in cell_ids:
            w.writerow([cid] + [''] * len(CATEGORIES) + [''])
    print(f'wrote {template_path} (blank, for manual completion)')

    # ---- answer key, separately saved, explicitly marked ----
    key_path = os.path.join(C.RESULTS_ANALYSIS, 'rq2_blind_key_N10000.csv')
    with open(key_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['# DO NOT VIEW BEFORE ANNOTATING -- de-anonymization key, not for the annotator'])
        w.writerow(['cell_id', 'image_id', 'group', 'class_id'])
        for cid, img_id, grp in zip(cell_ids, shuffled_ids, shuffled_group):
            w.writerow([cid, img_id, grp, id_lookup[img_id]['class_id']])
    print(f'wrote {key_path} (answer key -- DO NOT VIEW before annotating)')

    print('\nNo pattern labels are pre-filled. Annotation of annotation_template_N10000.csv is still '
          'required before any common-pattern claim can be made for the top-50.')


if __name__ == '__main__':
    main()
