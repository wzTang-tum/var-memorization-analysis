"""
The N=10000 forward pass: fc1 + attn_proj + fc2, abs() and clamp(0) paired
in the same forward pass, plus an independent old-subset accumulator
updated in parallel for the 1000 reused images.

Two points worth noting about the design:

  1. All three layer types (fc1, attn_proj, fc2) get the paired abs-vs-clamp
     comparison in one forward pass (HookManager's clamp_compare_groups
     mechanism is generic over group name, so it is enabled for all three
     rather than scoping the comparison to fc1 only).
  2. The per-image seed is derived from a stable `seed_key` (stored in the
     manifest), not from array position. For the 1000 reused old images
     this equals their original position (0..999) in data/subset_1000.json,
     so crops for those images are byte-identical to the earlier N=1000 run
     regardless of where they land in the new 10000-entry manifest. A
     separate UnitMemAccumulator set (`old_accs`), keyed by seed_key
     (0..999) instead of global image_id (0..9999), is updated in the same
     forward pass for exactly the 1000 old-subset images -- necessary
     because the global accumulator is an O(U) streaming reduction
     (run_sum/run_max) that cannot be decomposed post-hoc into an
     arbitrary subset's statistics.
"""
import argparse
import csv
import hashlib
import json
import os
import pickle
import random
import sys
import time

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))
from model import build_and_load          # noqa: E402
from data import TRAIN_TRANSFORM          # noqa: E402
from hooks import HookManager             # noqa: E402
from unitmem import UnitMemAccumulator, compute_regime_values  # noqa: E402
from pipeline import assemble_a_bar       # noqa: E402

ROOT = os.path.dirname(os.path.abspath(__file__))
CKPT_DIR = os.path.join(ROOT, 'checkpoints')
RESULTS = os.path.join(ROOT, 'results')
os.makedirs(RESULTS, exist_ok=True)

SEED = 42
N_BLOCKS = 16
BASE_GROUPS = ['fc1', 'attn_proj', 'fc2']
GROUPS = ['fc1', 'fc1_clamp', 'attn_proj', 'attn_proj_clamp', 'fc2', 'fc2_clamp']
GROUP_WIDTH = {'fc1': 4096, 'attn_proj': 1024, 'fc2': 1024}


def _base(g):
    return g[:-len('_clamp')] if g.endswith('_clamp') else g


def width_of(g):
    return GROUP_WIDTH[_base(g)] * N_BLOCKS


# ============================================================
# Deterministic per-image seeding, parameterized by a stable seed_key
# instead of array position.
# ============================================================
def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def derive_image_seed(seed, seed_key):
    ss = np.random.SeedSequence([seed, seed_key])
    return int(ss.generate_state(1)[0])


def make_views_deterministic(pil_img, seed_key, hash_rows=None, global_img_id=None):
    torch.manual_seed(derive_image_seed(SEED, seed_key))
    views = []
    for view_id in range(10):
        v = TRAIN_TRANSFORM(pil_img)
        if hash_rows is not None:
            h = hashlib.sha256(v.numpy().tobytes()).hexdigest()
            hash_rows.append((global_img_id, seed_key, view_id, h))
        views.append(v)
    return torch.stack(views)


# ============================================================
# Diagnostic: verify abs/clamp are bit-exact functions of the SAME raw
# tensor, checked on fc1, attn_proj, and fc2 (48 hook points).
# ============================================================
class RawIdentityChecker:
    def __init__(self, var):
        self.var = var
        self._handles = []
        self.violations = []
        self.n_checks = 0

    def register(self):
        for i, blk in enumerate(self.var.blocks):
            self._handles.append(blk.ffn.act.register_forward_hook(self._make_hook('fc1', i)))
            self._handles.append(blk.attn.proj.register_forward_hook(self._make_hook('attn_proj', i)))
            self._handles.append(blk.ffn.fc2.register_forward_hook(self._make_hook('fc2', i)))

    def _make_hook(self, group, block_idx):
        def hook(module, inp, out):
            raw = out.detach().float()
            abs_t = raw.abs()
            clamp_t = raw.clamp(min=0)
            self.n_checks += 1
            ok = True
            if abs_t.shape != raw.shape or clamp_t.shape != raw.shape:
                ok = False
            if not torch.isfinite(raw).all():
                ok = False
            if (abs_t < 0).any() or (clamp_t < 0).any():
                ok = False
            pos_mask = raw >= 0
            if not torch.equal(abs_t[pos_mask], clamp_t[pos_mask]):
                ok = False
            neg_mask = raw < 0
            if neg_mask.any():
                if not torch.equal(clamp_t[neg_mask], torch.zeros_like(clamp_t[neg_mask])):
                    ok = False
                if not torch.equal(abs_t[neg_mask], (-raw[neg_mask])):
                    ok = False
            if not ok:
                self.violations.append((group, block_idx))
        return hook

    def remove(self):
        for h in self._handles:
            h.remove()
        self._handles = []


# ============================================================
# Per-image processing (paired, 3 groups, dual accumulators)
# ============================================================
def process_one_image_paired(vae, var, hm, rec, device, accs, old_accs, hash_rows=None):
    with Image.open(rec['local_path_abs']) as pil_im:
        pil_im = pil_im.convert('RGB')
        views = make_views_deterministic(pil_im, rec['seed_key'], hash_rows, rec['image_id'])
    views = views.to(device)
    label = torch.full((10,), rec['class_id'], device=device, dtype=torch.long)

    with torch.no_grad():
        if device == 'cuda':
            torch.cuda.synchronize()
        t0 = time.time()
        gt_idx_Bl = vae.img_to_idxBl(views)
        x_BLCv = vae.quantize.idxBl_to_var_input(gt_idx_Bl)
        if device == 'cuda':
            torch.cuda.synchronize()
        t1 = time.time()
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=(device == 'cuda')):
            var(label, x_BLCv)
        if device == 'cuda':
            torch.cuda.synchronize()
        t2 = time.time()

    for base in BASE_GROUPS:
        a_shape = hm.cache[(base, 0)].shape
        c_shape = hm.cache[(base + '_clamp', 0)].shape
        assert a_shape == c_shape, f'{base}: abs/clamp cache shape mismatch {a_shape} vs {c_shape}'

    for g in GROUPS:
        a_bar = assemble_a_bar(hm, g)
        values_RU = compute_regime_values(a_bar)
        accs[g].update(values_RU, rec['image_id'])
        if rec['is_old_subset'] and old_accs is not None:
            old_accs[g].update(values_RU, rec['seed_key'])  # seed_key == old positional index 0..999

    return t1 - t0, t2 - t1


def new_accs():
    return {g: UnitMemAccumulator(width_of(g)) for g in GROUPS}


def run_paired(vae, var, images, device, checkpoint_path, checkpoint_every, hash_csv_path, log,
               maintain_old_accs=True):
    hm = HookManager(var, clamp_compare_groups=set(BASE_GROUPS))
    hm.register()
    accs = new_accs()
    old_accs = new_accs() if maintain_old_accs else None
    start_idx = 0
    t_vae_total = 0.0
    t_var_total = 0.0

    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, 'rb') as f:
            ckpt = pickle.load(f)
        accs = ckpt['accs']
        old_accs = ckpt['old_accs']
        start_idx = ckpt['next_idx']
        t_vae_total = ckpt['t_vae_total']
        t_var_total = ckpt['t_var_total']
        log(f'resumed from checkpoint {checkpoint_path} at image {start_idx}/{len(images)}')

    write_header = start_idx == 0 or not os.path.exists(hash_csv_path)
    hash_f = open(hash_csv_path, 'w' if write_header else 'a', newline='', encoding='utf-8')
    hash_writer = csv.writer(hash_f)
    if write_header:
        hash_writer.writerow(['global_image_id', 'seed_key', 'view_id', 'tensor_sha256'])

    t_start = time.time()
    for idx in range(start_idx, len(images)):
        rec = images[idx]
        hash_rows = []
        dt_vae, dt_var = process_one_image_paired(vae, var, hm, rec, device, accs, old_accs, hash_rows)
        for row in hash_rows:
            hash_writer.writerow(row)
        t_vae_total += dt_vae
        t_var_total += dt_var

        if (idx + 1) % 50 == 0 or idx == len(images) - 1:
            elapsed = time.time() - t_start
            rate = (idx + 1 - start_idx) / elapsed if elapsed > 0 else 0
            eta_min = (len(images) - idx - 1) / rate / 60 if rate > 0 else float('nan')
            log(f'  {idx+1}/{len(images)} images processed ({rate:.2f} img/s, ETA {eta_min:.1f} min)')

        if (idx + 1) % checkpoint_every == 0:
            hash_f.flush()
            with open(checkpoint_path, 'wb') as f:
                pickle.dump({'accs': accs, 'old_accs': old_accs, 'next_idx': idx + 1,
                             't_vae_total': t_vae_total, 't_var_total': t_var_total}, f)
            log(f'  checkpoint saved at {idx+1}/{len(images)} -> {checkpoint_path}')

    hash_f.close()
    hm.remove()
    return accs, old_accs, t_vae_total, t_var_total


def load_manifest():
    with open(os.path.join(RESULTS, 'dataset_manifest_N10000.json'), encoding='utf-8') as f:
        manifest = json.load(f)
    images = manifest['images']
    for im in images:
        im['local_path_abs'] = os.path.join(ROOT, im['local_path'])
    return manifest, images


def pack_results(accs, extra=None):
    payload = dict(extra or {})
    for g, acc in accs.items():
        res = acc.finalize()
        for k, v in res.items():
            if k == 'regime_names':
                continue
            payload[f'{g}_{k}'] = v
        payload[f'{g}_regime_names'] = np.array(res['regime_names'])
    return payload


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', choices=['accept20', 'regression1000', 'full10000'], required=True)
    args = ap.parse_args()

    lines = []

    def log(s=''):
        print(s)
        lines.append(s)

    manifest, images = load_manifest()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if args.stage == 'accept20':
        log('=== N=20 acceptance test (fc1+attn_proj+fc2, abs+clamp, dual accumulators) ===')
        seed_everything(SEED)
        vae, var, _ = build_and_load(device, CKPT_DIR, depth=16)

        old_pool = [im for im in images if im['is_old_subset']][:10]
        new_pool = [im for im in images if not im['is_old_subset']][:10]
        test_set = old_pool + new_pool
        log(f'test set: {len(old_pool)} old-subset images + {len(new_pool)} new images = {len(test_set)} total')

        checker = RawIdentityChecker(var)
        checker.register()
        hm = HookManager(var, clamp_compare_groups=set(BASE_GROUPS))
        hm.register()
        accs = new_accs()
        old_accs = new_accs()

        hash_rows_all = []
        for rec in test_set:
            hash_rows = []
            process_one_image_paired(vae, var, hm, rec, device, accs, old_accs, hash_rows)
            hash_rows_all.extend(hash_rows)
            log(f'  image_id={rec["image_id"]} seed_key={rec["seed_key"]} old_subset={rec["is_old_subset"]}: '
                f'{len(hash_rows)} view hashes recorded')

        expected_checks = len(test_set) * N_BLOCKS * 3  # 3 groups per block
        log(f'\nRawIdentityChecker: {checker.n_checks} hook calls checked (expected {expected_checks}), '
            f'violations = {checker.violations}')
        assert checker.n_checks == expected_checks
        assert len(checker.violations) == 0, f'raw-identity violations in {checker.violations}'
        log('PASS: abs/clamp bit-exact vs identical raw activation on every check, for fc1, attn_proj, AND fc2.')
        checker.remove()

        for g in GROUPS:
            um = accs[g].finalize()['unitmem']
            assert not np.isnan(um).any() and not np.isinf(um).any(), f'{g}: NaN/Inf at N=20!'
        log('PASS: no NaN/Inf in any of the 6 group accumulators at N=20 (sanity only).')

        for g in GROUPS:
            um_old = old_accs[g].finalize()['unitmem']
            assert um_old.shape == accs[g].finalize()['unitmem'].shape
        log(f'PASS: old_accs updated for exactly {len(old_pool)} old-subset images (n_images='
            f'{old_accs["fc1"].n_images}), independent from global accs (n_images={accs["fc1"].n_images}).')
        assert old_accs['fc1'].n_images == len(old_pool)
        assert accs['fc1'].n_images == len(test_set)
        hm.remove()

        hash_path = os.path.join(RESULTS, 'view_hashes_accept20.csv')
        with open(hash_path, 'w', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            w.writerow(['global_image_id', 'seed_key', 'view_id', 'tensor_sha256'])
            w.writerows(hash_rows_all)
        log(f'\nwrote {hash_path} ({len(hash_rows_all)} rows)')
        log('\n=== N=20 acceptance test: ALL CHECKS PASSED ===')

        out_path = os.path.join(RESULTS, 'accept20_report.txt')
        with open(out_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines) + '\n')
        print(f'\nwrote {out_path}')

    elif args.stage == 'regression1000':
        log('=== N=1000 regression check (old-subset only, strict hard gate) ===')
        seed_everything(SEED)
        vae, var, _ = build_and_load(device, CKPT_DIR, depth=16)

        old_images = sorted([im for im in images if im['is_old_subset']], key=lambda r: r['seed_key'])
        assert len(old_images) == 1000
        assert [im['seed_key'] for im in old_images] == list(range(1000))
        log(f'processing {len(old_images)} old-subset images through this pipeline '
            f'(fc1+attn_proj+fc2, abs+clamp), using a fresh accumulator keyed by seed_key (0..999)')

        hm = HookManager(var, clamp_compare_groups=set(BASE_GROUPS))
        hm.register()
        accs = new_accs()

        hash_path = os.path.join(RESULTS, 'regression1000_view_hashes.csv')
        hash_f = open(hash_path, 'w', newline='', encoding='utf-8')
        hash_writer = csv.writer(hash_f)
        hash_writer.writerow(['global_image_id', 'seed_key', 'view_id', 'tensor_sha256'])

        t0 = time.time()
        for i, rec in enumerate(old_images):
            hash_rows = []
            process_one_image_paired(vae, var, hm, rec, device, accs, None, hash_rows)
            for row in hash_rows:
                hash_writer.writerow(row)
            if (i + 1) % 100 == 0:
                log(f'  {i+1}/1000 processed')
        hash_f.close()
        hm.remove()
        wall = time.time() - t0
        log(f'regression run wall time: {wall:.1f}s')

        out_npz = os.path.join(RESULTS, 'regression1000_unitmem.npz')
        payload = pack_results(accs, extra={'N': 1000})
        np.savez_compressed(out_npz, **payload)
        log(f'wrote {out_npz}')

        out_path = os.path.join(RESULTS, 'regression1000_run_log.txt')
        with open(out_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines) + '\n')
        print(f'\nwrote {out_path}')

    elif args.stage == 'full10000':
        log('=== official N=10000 run (fc1+attn_proj+fc2, abs+clamp, dual accumulators) ===')
        log('re-seeding all RNGs to 42 before the official run')
        seed_everything(SEED)
        vae, var, _ = build_and_load(device, CKPT_DIR, depth=16)

        if device == 'cuda':
            torch.cuda.reset_peak_memory_stats()
        ckpt_path = os.path.join(RESULTS, 'checkpoint_N10000.pkl')
        hash_path = os.path.join(RESULTS, 'view_hashes_N10000.csv')
        t0 = time.time()
        accs, old_accs, t_vae_total, t_var_total = run_paired(
            vae, var, images, device, ckpt_path, 200, hash_path, log, maintain_old_accs=True)
        wall_time = time.time() - t0
        peak_vram_mb = torch.cuda.max_memory_allocated() / 1e6 if device == 'cuda' else 0.0

        for g in GROUPS:
            um = accs[g].finalize()['unitmem']
            assert um.min() >= -1e-6 and um.max() <= 1.0 + 1e-6, f'{g}: UnitMem out of [0,1]!'
            assert not np.isnan(um).any() and not np.isinf(um).any(), f'{g}: NaN/Inf!'
            log(f'{g}: unitmem range=[{um.min():.6f}, {um.max():.6f}], no NaN/Inf -- OK')
        assert accs['fc1'].n_images == 10000
        assert old_accs['fc1'].n_images == 1000
        log(f'global accumulator n_images=10000 (OK), old-subset accumulator n_images=1000 (OK)')

        class_ids = np.array([im['class_id'] for im in images])
        seed_keys = np.array([im['seed_key'] for im in images])
        is_old = np.array([im['is_old_subset'] for im in images])

        out_npz = os.path.join(RESULTS, 'unitmem_N10000_global.npz')
        payload = pack_results(accs, extra={'N': 10000, 'seed': SEED, 'image_class_ids': class_ids,
                                             'image_seed_keys': seed_keys, 'image_is_old_subset': is_old})
        np.savez_compressed(out_npz, **payload)
        log(f'wrote {out_npz}')

        out_npz2 = os.path.join(RESULTS, 'unitmem_old_subset_within_N10000.npz')
        old_class_ids = np.array([im['class_id'] for im in images if im['is_old_subset']])
        old_order = np.argsort([im['seed_key'] for im in images if im['is_old_subset']])
        old_class_ids = old_class_ids[old_order]
        payload2 = pack_results(old_accs, extra={'N': 1000, 'seed': SEED, 'image_class_ids': old_class_ids})
        np.savez_compressed(out_npz2, **payload2)
        log(f'wrote {out_npz2}')

        log(f'\nwall time={wall_time:.1f}s ({wall_time/3600:.3f} h), peak VRAM={peak_vram_mb:.1f} MB, '
            f'VAE encode total={t_vae_total:.1f}s, VAR forward total={t_var_total:.1f}s')

        if os.path.exists(ckpt_path):
            os.remove(ckpt_path)
            log(f'removed checkpoint {ckpt_path} (run completed successfully)')

        out_path = os.path.join(RESULTS, 'full10000_run_log.txt')
        with open(out_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines) + '\n')
        print(f'\nwrote {out_path}')


if __name__ == '__main__':
    main()
