"""
Build the N=10000 (1000 classes x 10 images) dataset manifest.

Design:
  1. The original N=1000 (100 classes x 10) images from 01_build_subset.py
     are included unchanged, reusing their existing local files as
     canonical (never re-extracted).
  2. Every image (old and new) gets a stable `seed_key` used only for
     deterministic view-crop generation, decoupled from its positional
     `image_id` in this manifest. For the 1000 reused images, seed_key
     equals their original positional index (0..999) in
     data/subset_1000.json, so `derive_image_seed(SEED, seed_key)`
     reproduces byte-identical crops to the earlier N=1000 run. New images
     get seed_key = 1000, 1001, ... (a disjoint range, in selection order),
     never colliding with 0..999.
  3. Image validation: PIL .verify() alone only checks file structure, not
     full pixel decode -- every candidate is additionally reopened,
     .convert('RGB'), and .load()'d to confirm true decoding.
  4. SHA256 byte-level dedup across the full pool (old 1000 + newly
     selected). Any duplicate is resolved by keeping the old path as
     canonical (never re-extracting/duplicating a byte-identical image).
  5. Every one of the 6 local parquet shards' per-class label distribution
     is tabulated, and the parquet `label` (0..999) convention is checked
     against data/meta/imagenet_class_index.json's WNID ordering for
     every scanned row (not a sample) -- this is what class-conditioning
     in this project assumes.
  6. This sample is not a uniform draw from the complete ImageNet-1k train
     split -- it is class-balanced sampling from the candidate pool found
     in 6 already-downloaded local parquet shards (out of 294 total). All
     written outputs describe it this way, never as "uniform over the
     full train set".

No network access (shards are already local under checkpoints/imagenet_shards).
"""
import hashlib
import io
import json
import os
import sys

import numpy as np
import pyarrow.parquet as pq
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

ROOT = os.path.dirname(os.path.abspath(__file__))
SHARD_DIR = os.path.join(ROOT, 'checkpoints', 'imagenet_shards', 'data')
SHARD_IDXS = [26, 65, 84, 98, 233, 237]
TOTAL_TRAIN_SHARDS = 294

RESULTS = os.path.join(ROOT, 'results')
DATA_V2 = os.path.join(ROOT, 'data_N10000')
IMAGES_V2 = os.path.join(DATA_V2, 'images')
os.makedirs(RESULTS, exist_ok=True)
os.makedirs(IMAGES_V2, exist_ok=True)

SEED = 42
N_CLASSES_NEW = 1000
N_PER_CLASS = 10
NEW_PICK_SEED = SEED + 3  # distinct from 01_build_subset.py's SEED+2 (old-100-class picks)
MAX_ATTEMPT_POOL = 20     # cap candidates attempted per new class (decode-fallback budget)


def shard_local_path(idx):
    return os.path.join(SHARD_DIR, f'train-{idx:05d}-of-{TOTAL_TRAIN_SHARDS:05d}.parquet')


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def verify_decode(img_bytes):
    """Stronger than PIL .verify() alone: reopen, convert('RGB'), .load()."""
    try:
        with Image.open(io.BytesIO(img_bytes)) as im0:
            im0.verify()
        with Image.open(io.BytesIO(img_bytes)) as im1:
            im1 = im1.convert('RGB')
            im1.load()
        return True
    except Exception:
        return False


def main():
    lines = []

    def log(s=''):
        print(s)
        lines.append(s)

    log('=== building the N=10000 dataset manifest ===')

    with open(os.path.join(ROOT, 'data', 'meta', 'imagenet_class_index.json'), encoding='utf-8') as f:
        class_index = json.load(f)  # {"0": ["n01440764", "tench"], ...}
    wnid_of = {int(k): v[0] for k, v in class_index.items()}
    assert len(wnid_of) == 1000

    with open(os.path.join(ROOT, 'data', 'subset_1000.json'), encoding='utf-8') as f:
        old_manifest = json.load(f)
    old_images = old_manifest['images']
    assert len(old_images) == 1000
    old_classes = set(old_manifest['target_classes'])
    assert len(old_classes) == 100

    # ------------------------------------------------------------
    # Phase 1: full label+path scan of all 6 local shards (no image
    # bytes). Builds per-class candidate pools, per-shard per-class
    # distributions, and the WNID<->label consistency check.
    # ------------------------------------------------------------
    log('\n--- Phase 1: scanning 6 local shards (label + path columns only) ---')
    all_candidates = {c: [] for c in range(1000)}  # class_id -> [(shard_idx, row_idx, path), ...]
    per_shard_class_counts = {s: {} for s in SHARD_IDXS}
    mismatches = []
    total_rows = 0

    for shard_idx in SHARD_IDXS:
        path = shard_local_path(shard_idx)
        table = pq.read_table(path, columns=['image.path', 'label'])
        labels = table.column('label').to_numpy()
        paths = table.column('path').to_pylist()
        for row_idx, (lbl, p) in enumerate(zip(labels, paths)):
            lbl = int(lbl)
            all_candidates[lbl].append((shard_idx, row_idx, p))
            per_shard_class_counts[shard_idx][lbl] = per_shard_class_counts[shard_idx].get(lbl, 0) + 1
            wnid_expected = wnid_of[lbl]
            wnid_in_path = p.split('_')[0]
            if wnid_in_path != wnid_expected:
                mismatches.append((shard_idx, row_idx, lbl, wnid_expected, wnid_in_path, p))
            total_rows += 1
        log(f'  shard {shard_idx}: {len(paths)} rows scanned')

    log(f'\ntotal rows scanned: {total_rows}')
    counts_per_class = {c: len(v) for c, v in all_candidates.items()}
    cvals = list(counts_per_class.values())
    log(f'candidates per class over all 1000 classes: min={min(cvals)}, max={max(cvals)}, '
        f'mean={sum(cvals)/len(cvals):.2f}')
    thin = sorted(c for c, n in counts_per_class.items() if n < N_PER_CLASS)
    log(f'classes with < {N_PER_CLASS} candidates: {thin if thin else "none"}')

    log(f'\nlabel<->WNID consistency check (parquet `label` column vs '
        f'data/meta/imagenet_class_index.json[label][0] vs WNID prefix of `image.path`), '
        f'checked on ALL {total_rows} scanned rows: {len(mismatches)} mismatches')
    if mismatches:
        log('MISMATCHES FOUND (first 20 shown):')
        for m in mismatches[:20]:
            log(f'  {m}')
    else:
        log('PASS: every scanned row\'s image.path WNID prefix matches '
            'imagenet_class_index.json[label][0] for its parquet `label` value. '
            'This confirms the parquet label integer 0..999 uses the same WNID-sorted '
            'convention as data/meta/imagenet_class_index.json across all 1000 classes.')

    # per-shard class distribution report
    dist_path = os.path.join(RESULTS, 'shard_class_distribution.txt')
    with open(dist_path, 'w', encoding='utf-8') as f:
        f.write('Per-shard per-class candidate counts (label+path scan, no image bytes).\n')
        f.write(f'Shards: {SHARD_IDXS} (out of {TOTAL_TRAIN_SHARDS} total train shards)\n\n')
        for shard_idx in SHARD_IDXS:
            d = per_shard_class_counts[shard_idx]
            classes_present = len(d)
            f.write(f'shard {shard_idx}: {classes_present}/1000 classes present, '
                     f'{sum(d.values())} rows total, '
                     f'per-class count min={min(d.values())} max={max(d.values())} '
                     f'mean={sum(d.values())/len(d):.2f}\n')
        f.write('\nfull per-shard per-class histogram (class_id: {shard_idx: count, ...}):\n')
        for c in range(1000):
            row = {s: per_shard_class_counts[s].get(c, 0) for s in SHARD_IDXS if per_shard_class_counts[s].get(c, 0) > 0}
            f.write(f'  {c}: {row}\n')
    log(f'\nwrote {dist_path}')

    assert min(cvals) >= N_PER_CLASS, 'a class has fewer than N_PER_CLASS candidates in the local shard pool!'
    if mismatches:
        raise RuntimeError('label/WNID convention mismatch found -- stopping on an unverified label '
                            'convention. See report above.')

    # cross-check: every old image's (shard,row) location is consistent
    # with its recorded class_id and filename in the fresh scan.
    old_lookup = {(im['source_shard'].split('-')[1], im['row_in_shard']): im for im in old_images}
    # source_shard format: 'data/train-00098-of-00294.parquet' -> shard idx string '00098'
    old_consistency_fail = []
    for im in old_images:
        shard_idx = int(im['source_shard'].split('-')[1])
        row_idx = im['row_in_shard']
        found = None
        for (s, r, p) in all_candidates[im['class_id']]:
            if s == shard_idx and r == row_idx:
                found = p
                break
        if found is None or found != im['image_id']:
            old_consistency_fail.append((im['class_id'], shard_idx, row_idx, im['image_id'], found))
    log(f'\nold-1000-image (shard,row)->(class,filename) consistency check against fresh scan: '
        f'{len(old_consistency_fail)} inconsistencies')
    if old_consistency_fail:
        for x in old_consistency_fail[:20]:
            log(f'  MISMATCH: {x}')
        raise RuntimeError('old N=1000 images inconsistent with fresh parquet scan -- stopping.')
    log('PASS: all 1000 old images\' (shard,row) locations match their recorded class_id and filename.')

    # ------------------------------------------------------------
    # Phase 2: old 1000 images -- mandatory inclusion, canonical paths,
    # SHA256 of existing local files, seed_key = old positional index.
    # ------------------------------------------------------------
    log('\n--- Phase 2: hashing the 1000 old (mandatory, reused) images ---')
    old_records = []  # in OLD manifest order; seed_key = old positional index
    sha_to_record = {}  # sha256 -> record (for dedup)
    for old_pos, im in enumerate(old_images):
        abs_path = os.path.join(ROOT, im['local_path'])
        with open(abs_path, 'rb') as f:
            b = f.read()
        sha = sha256_bytes(b)
        rec = {
            'class_id': im['class_id'],
            'filename': im['image_id'],
            'source_shard': im['source_shard'],
            'row_in_shard': im['row_in_shard'],
            'local_path': im['local_path'],  # reused as-is, not copied into data_N10000
            'sha256': sha,
            'seed_key': old_pos,
            'is_old_subset': True,
        }
        old_records.append(rec)
        if sha in sha_to_record:
            raise RuntimeError(f'unexpected duplicate SHA256 within the old 1000 images themselves: '
                                f'{rec["filename"]} vs {sha_to_record[sha]["filename"]}')
        sha_to_record[sha] = rec
    log(f'hashed {len(old_records)} old images, all distinct SHA256 (no internal duplicates)')

    # ------------------------------------------------------------
    # Phase 3: select 10 images for each of the 900 NEW classes,
    # with strict decode verification and fallback within the pool.
    # ------------------------------------------------------------
    log('\n--- Phase 3: selecting 10 images for each of the 900 new classes ---')
    new_classes = sorted(c for c in range(1000) if c not in old_classes)
    assert len(new_classes) == 900

    # build a bounded, permuted attempt-pool per class
    attempt_pool = {}
    for c in new_classes:
        cand = sorted(all_candidates[c])
        rng = np.random.default_rng(NEW_PICK_SEED + c)
        order = rng.permutation(len(cand))[:MAX_ATTEMPT_POOL]
        attempt_pool[c] = [cand[i] for i in order]

    # group needed rows per shard for a single bytes-read pass per shard
    rows_needed_per_shard = {s: [] for s in SHARD_IDXS}
    for c in new_classes:
        for shard_idx, row_idx, path in attempt_pool[c]:
            rows_needed_per_shard[shard_idx].append((row_idx, c, path))

    extracted_bytes = {}  # (shard_idx, row_idx) -> bytes
    for shard_idx in SHARD_IDXS:
        needed = rows_needed_per_shard[shard_idx]
        if not needed:
            continue
        needed_rows = sorted(set(r for r, _, _ in needed))
        path = shard_local_path(shard_idx)
        table = pq.read_table(path, columns=['image.bytes'])
        bytes_col = table.column('bytes')
        for row_idx in needed_rows:
            extracted_bytes[(shard_idx, row_idx)] = bytes_col[row_idx].as_py()
        log(f'  shard {shard_idx}: extracted bytes for {len(needed_rows)} candidate rows')

    new_records = []
    new_counter = 1000  # seed_key for new images, disjoint from old 0..999
    decode_failures = []
    class_failures = []
    for c in new_classes:
        chosen_for_class = []
        for shard_idx, row_idx, path in attempt_pool[c]:
            if len(chosen_for_class) >= N_PER_CLASS:
                break
            b = extracted_bytes[(shard_idx, row_idx)]
            if not verify_decode(b):
                decode_failures.append((c, shard_idx, row_idx, path))
                continue
            sha = sha256_bytes(b)
            chosen_for_class.append((shard_idx, row_idx, path, b, sha))
        if len(chosen_for_class) < N_PER_CLASS:
            class_failures.append((c, len(chosen_for_class), len(attempt_pool[c])))
            continue
        for shard_idx, row_idx, path, b, sha in chosen_for_class:
            if sha in sha_to_record:
                canon = sha_to_record[sha]
                log(f'  DEDUP: class {c} candidate {path} is byte-identical to already-included '
                    f'{canon["filename"]} (class {canon["class_id"]}) -- reusing canonical entry, '
                    f'not re-extracting.')
                rec = dict(canon)
                rec['seed_key'] = new_counter
                rec['is_old_subset'] = False
                rec['dedup_of'] = canon['filename']
                new_records.append(rec)
                new_counter += 1
                continue
            class_dir = os.path.join(IMAGES_V2, f'{c:03d}')
            os.makedirs(class_dir, exist_ok=True)
            out_path = os.path.join(class_dir, path)
            with open(out_path, 'wb') as f:
                f.write(b)
            rec = {
                'class_id': c,
                'filename': path,
                'source_shard': f'data/train-{shard_idx:05d}-of-{TOTAL_TRAIN_SHARDS:05d}.parquet',
                'row_in_shard': row_idx,
                'local_path': os.path.relpath(out_path, ROOT).replace('\\', '/'),
                'sha256': sha,
                'seed_key': new_counter,
                'is_old_subset': False,
            }
            sha_to_record[sha] = rec
            new_records.append(rec)
            new_counter += 1

    log(f'\nnew-class selection: {len(new_records)}/{900*10} images selected, '
        f'{len(decode_failures)} decode failures encountered (fallback absorbed), '
        f'{len(class_failures)} classes could not reach {N_PER_CLASS} images')
    if decode_failures:
        for d in decode_failures[:20]:
            log(f'  decode failure (absorbed by fallback): {d}')
    if class_failures:
        for cf in class_failures:
            log(f'  CLASS FAILURE: {cf}')
        raise RuntimeError(f'{len(class_failures)} classes could not reach {N_PER_CLASS} successfully-decoded '
                            f'images within the attempt pool (cap={MAX_ATTEMPT_POOL}) -- stopping (no silent '
                            f'workaround, no augmentation-based N-faking). See report above for exactly which '
                            f'classes and why.')

    assert len(new_records) == 900 * 10

    # ------------------------------------------------------------
    # Phase 4: assemble final manifest, image_id = sequential position
    # (class ascending, old-class images keep old within-class order,
    # new-class images in selection order), seed_key already assigned.
    # ------------------------------------------------------------
    log('\n--- Phase 4: assembling final manifest ---')
    by_class = {c: [] for c in range(1000)}
    for rec in old_records:
        by_class[rec['class_id']].append(rec)
    for rec in new_records:
        by_class[rec['class_id']].append(rec)
    for c in range(1000):
        assert len(by_class[c]) == N_PER_CLASS, f'class {c} has {len(by_class[c])} images, expected {N_PER_CLASS}'

    final_images = []
    for c in range(1000):
        for rec in by_class[c]:
            entry = dict(rec)
            entry['image_id'] = len(final_images)  # 0..9999, positional, for arg_max indexing
            final_images.append(entry)
    assert len(final_images) == 10000

    n_old_in_final = sum(1 for e in final_images if e['is_old_subset'])
    assert n_old_in_final == 1000, f'expected exactly 1000 old-subset images in final manifest, got {n_old_in_final}'

    all_shas = [e['sha256'] for e in final_images]
    assert len(set(all_shas)) == len(all_shas), 'unexpected duplicate SHA256 in final manifest after dedup pass!'

    manifest = {
        'seed': SEED,
        'n_classes': 1000,
        'n_per_class': N_PER_CLASS,
        'n_images': 10000,
        'source_dataset': 'ILSVRC/imagenet-1k (local shards only, no re-download)',
        'shards_used': [f'data/train-{s:05d}-of-{TOTAL_TRAIN_SHARDS:05d}.parquet' for s in SHARD_IDXS],
        'sampling_description': (
            'Class-balanced sampling (10 images/class) from the candidate pool found in 6 already-'
            'downloaded local parquet shards (6 of 294 total ILSVRC/imagenet-1k train shards). This '
            'is not a uniform draw from the complete ImageNet-1k train split: within each class, '
            'candidates are restricted to whatever appeared in these 6 (shuffled) shards, typically '
            '11-46 candidates/class (mean 26.1), not the ~1300 true training images/class available '
            'in the full dataset.'
        ),
        'old_subset_note': (
            'The 1000 images from the original N=1000 (100 classes x 10) experiment (data/subset_1000.json) '
            'are included unchanged at their original local paths under data/images/ (not copied into '
            'data_N10000/). Their seed_key equals their original positional index (0..999) in '
            'data/subset_1000.json, reproducing byte-identical augmentation crops. New images (9000) have '
            'seed_key 1000..9999 in selection order, disjoint from the old range.'
        ),
        'label_wnid_consistency_check': f'PASS: 0/{total_rows} mismatches across all 6 shards, all 1000 classes',
        'decode_failures_absorbed_by_fallback': len(decode_failures),
        'images': final_images,
    }

    out_path = os.path.join(RESULTS, 'dataset_manifest_N10000.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2)
    log(f'\nwrote {out_path} ({len(final_images)} images, 1000 classes, {n_old_in_final} reused old + '
        f'{10000-n_old_in_final} new)')

    report_path = os.path.join(RESULTS, 'manifest_build_log.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    log(f'wrote {report_path}')
    log('\n=== manifest build complete ===')


if __name__ == '__main__':
    main()
