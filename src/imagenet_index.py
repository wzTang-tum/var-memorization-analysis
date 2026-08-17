"""
Cheap label/path-only scan over ILSVRC/imagenet-1k train parquet shards to
find candidate rows for the target classes, without touching image bytes.

The HF parquet conversion of ILSVRC/imagenet-1k train is fully shuffled --
a single row group of a single shard already spans labels 0..999. This
means any handful of shards, scanned in a fixed random order, gives a
roughly uniform sample across all 1000 classes, so there is no need to
locate "which shard has class X" -- shards are scanned until every target
class has enough candidates.
"""
import time

import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem

from config import HF_DATASET, TOTAL_TRAIN_SHARDS, SEED, CANDIDATE_MARGIN


def shard_repo_path(idx: int) -> str:
    return f'datasets/{HF_DATASET}/data/train-{idx:05d}-of-{TOTAL_TRAIN_SHARDS:05d}.parquet'


def shard_hub_path(idx: int) -> str:
    return f'data/train-{idx:05d}-of-{TOTAL_TRAIN_SHARDS:05d}.parquet'


def scan_for_candidates(token: str, target_classes, seed=SEED, margin=CANDIDATE_MARGIN, max_shards=None, log=print):
    """
    Returns (candidates, shard_scan_order_used) where
    candidates: {class_id: [(shard_idx, row_idx, image_path_str), ...]}
    Scans shards (label+path columns only, no image bytes) in a fixed
    seeded random order until every class in target_classes has >= margin
    candidates.
    """
    fs = HfFileSystem(token=token)
    shard_order = np.random.default_rng(seed + 1).permutation(TOTAL_TRAIN_SHARDS).tolist()

    target_set = set(int(c) for c in target_classes)
    candidates = {c: [] for c in target_set}
    scanned = []

    for shard_idx in shard_order:
        if max_shards is not None and len(scanned) >= max_shards:
            break
        t0 = time.time()
        f = fs.open(shard_repo_path(shard_idx), 'rb')
        pf = pq.ParquetFile(f)
        table = pf.read(columns=['image.path', 'label'])
        labels = table.column('label').to_numpy()
        paths = [row['path'] for row in table.column('image').to_pylist()]
        scanned.append(shard_idx)
        n_added = 0
        for row_idx, (lbl, p) in enumerate(zip(labels, paths)):
            lbl = int(lbl)
            if lbl in target_set and len(candidates[lbl]) < margin * 3:  # cap growth, no need to keep unlimited
                candidates[lbl].append((shard_idx, row_idx, p))
                n_added += 1
        n_short = sum(1 for c in target_set if len(candidates[c]) < margin)
        log(f'  scanned shard {shard_idx} ({len(scanned)} so far, {time.time()-t0:.1f}s): '
            f'+{n_added} candidates, {n_short}/{len(target_set)} classes still below margin={margin}')
        if n_short == 0:
            break

    return candidates, scanned
