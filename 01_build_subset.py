"""
Build the initial 1000-image ImageNet-1k train subset (100 classes x 10
images/class).

Pipeline:
  1. Sample 100 class ids from 0..999 with a fixed seed.
  2. Scan a handful of train parquet shards' label+path columns only
     (no image bytes) until every target class has >= CANDIDATE_MARGIN
     candidates (src/imagenet_index.py). The dataset is fully shuffled,
     so a handful of shards already covers all 1000 classes.
  3. Randomly pick exactly 10 candidates per class (fixed seed).
  4. Download ONLY the shards that were actually scanned/used (a few
     GB, not the full 138GB train split) via hf_hub_download (cached,
     resumable).
  5. Extract exactly those 1000 rows' raw image bytes from the local
     shard files and save them as files under data/images/.
  6. Write data/subset_1000.json recording the seed, target classes,
     shards used, and the exact identifier of every image, in a fixed
     order grouped by class.

Requires HF_TOKEN in the environment (a token that has accepted the
ILSVRC/imagenet-1k gated-dataset terms on huggingface.co).
"""
import io
import json
import os
import sys

import numpy as np
from PIL import Image
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))
from config import SEED, N_CLASSES, N_PER_CLASS, TOTAL_CLASSES, HF_DATASET  # noqa: E402
from imagenet_index import scan_for_candidates, shard_hub_path  # noqa: E402

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, 'data')
IMAGES_DIR = os.path.join(DATA_DIR, 'images')
SHARD_CACHE_DIR = os.path.join(ROOT, 'checkpoints', 'imagenet_shards')


def main():
    token = os.environ.get('HF_TOKEN')
    if not token:
        raise RuntimeError('HF_TOKEN not set in environment')

    os.makedirs(IMAGES_DIR, exist_ok=True)
    os.makedirs(SHARD_CACHE_DIR, exist_ok=True)

    # --- 1. sample target classes ---
    target_classes = np.random.default_rng(SEED).choice(TOTAL_CLASSES, N_CLASSES, replace=False)
    target_classes = sorted(int(c) for c in target_classes)
    print(f'sampled {len(target_classes)} target classes (seed={SEED}): {target_classes}')

    # --- 2. scan for candidates (label+path only, no image bytes) ---
    print('\nscanning shards for candidates (label/path columns only)...')
    candidates, scanned_shards = scan_for_candidates(token, target_classes, seed=SEED)
    counts = [len(v) for v in candidates.values()]
    print(f'\nscanned {len(scanned_shards)} shards: {scanned_shards}')
    print(f'candidates per class: min={min(counts)}, max={max(counts)}, mean={sum(counts)/len(counts):.1f}')
    assert min(counts) >= N_PER_CLASS, 'a target class has fewer than N_PER_CLASS candidates -- scan margin too low'

    # --- 3. pick exactly N_PER_CLASS per class, fixed seed ---
    pick_rng = np.random.default_rng(SEED + 2)
    chosen = {}  # class_id -> list of (shard_idx, row_idx, image_path)
    for c in target_classes:
        cand = sorted(candidates[c])  # sort for a deterministic order before random choice
        idx = pick_rng.choice(len(cand), N_PER_CLASS, replace=False)
        chosen[c] = [cand[i] for i in idx]

    used_shards = sorted(set(shard_idx for c in target_classes for shard_idx, _, _ in chosen[c]))
    print(f'\n{len(used_shards)} distinct shards needed to cover all 1000 chosen images: {used_shards}')

    # --- 4. download only the needed shards ---
    print('\ndownloading needed shards...')
    local_paths = {}
    for shard_idx in used_shards:
        hub_path = shard_hub_path(shard_idx)
        local_path = hf_hub_download(
            repo_id=HF_DATASET, repo_type='dataset', filename=hub_path,
            local_dir=SHARD_CACHE_DIR, token=token,
        )
        local_paths[shard_idx] = local_path
        size_mb = os.path.getsize(local_path) / 1e6
        print(f'  shard {shard_idx}: {local_path} ({size_mb:.1f} MB)')

    # --- 5. extract the 1000 rows' image bytes from the local shard files ---
    print('\nextracting images...')
    # group needed row indices per shard for a single local read per shard
    rows_needed_per_shard = {s: [] for s in used_shards}
    for c in target_classes:
        for shard_idx, row_idx, image_path in chosen[c]:
            rows_needed_per_shard[shard_idx].append((row_idx, c, image_path))

    records = {}  # (shard_idx, row_idx) -> local saved file path
    for shard_idx in used_shards:
        # projecting dotted column names flattens the struct: the returned
        # table has top-level columns 'bytes' / 'path', not a nested 'image'.
        table = pq.read_table(local_paths[shard_idx], columns=['image.bytes', 'image.path'])
        bytes_col = table.column('bytes')
        for row_idx, c, image_path in rows_needed_per_shard[shard_idx]:
            img_bytes = bytes_col[row_idx].as_py()
            # sanity check: bytes actually decode as an image
            with Image.open(io.BytesIO(img_bytes)) as im:
                im.verify()
            class_dir = os.path.join(IMAGES_DIR, f'{c:03d}')
            os.makedirs(class_dir, exist_ok=True)
            out_path = os.path.join(class_dir, image_path)
            with open(out_path, 'wb') as f:
                f.write(img_bytes)
            records[(shard_idx, row_idx)] = os.path.relpath(out_path, ROOT).replace('\\', '/')
    print(f'extracted {len(records)} images to {IMAGES_DIR}')

    # --- 6. write data/subset_1000.json, fixed order grouped by class ---
    images = []
    for c in target_classes:
        for shard_idx, row_idx, image_path in chosen[c]:
            images.append({
                'class_id': c,
                'image_id': image_path,               # original ImageNet filename, e.g. n0144xxxx_1234_...JPEG
                'source_shard': shard_hub_path(shard_idx),
                'row_in_shard': row_idx,
                'local_path': records[(shard_idx, row_idx)],
            })

    manifest = {
        'seed': SEED,
        'n_classes': N_CLASSES,
        'n_per_class': N_PER_CLASS,
        'target_classes': target_classes,
        'shards_scanned_for_candidates': scanned_shards,
        'shards_downloaded': used_shards,
        'candidate_margin': min(counts),
        'source_dataset': HF_DATASET,
        'note': 'ILSVRC/imagenet-1k train parquet is fully shuffled: a single row group already '
                'spans labels 0..999, so class locations were found via a cheap label/path-only '
                'scan rather than assuming class-contiguous shards.',
        'images': images,  # length 1000, fixed order grouped by class_id, never shuffle at runtime
    }
    assert len(images) == N_CLASSES * N_PER_CLASS

    out_path = os.path.join(DATA_DIR, 'subset_1000.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2)
    print(f'\nwrote {out_path} ({len(images)} images, {len(target_classes)} classes)')


if __name__ == '__main__':
    main()
