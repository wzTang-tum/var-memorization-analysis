"""
Shared main-loop pipeline: per-image forward pass + regime accumulation,
with checkpointing so an interrupted run can resume.
"""
import os
import pickle
import time

import numpy as np
import torch
from PIL import Image

from data import TRAIN_TRANSFORM
from hooks import HookManager
from unitmem import UnitMemAccumulator, compute_regime_values

GROUP_WIDTH = {'fc1': 4096, 'attn_proj': 1024, 'fc2': 1024}
N_BLOCKS = 16


def _base_group(g: str) -> str:
    return g[:-len('_clamp')] if g.endswith('_clamp') else g


def assemble_a_bar(hm: HookManager, group: str) -> np.ndarray:
    """[10 views, 10 scales, C] per block -> mean over views -> [10 scales, U] numpy (float64)."""
    parts = [hm.cache[(group, i)] for i in range(N_BLOCKS)]
    full = torch.cat(parts, dim=2)
    a_bar = full.mean(dim=0)
    return a_bar.double().cpu().numpy()


def process_one_image(vae, var, hm, im, device, accs, img_id, groups):
    with Image.open(im['local_path_abs']) as pil_im:
        pil_im = pil_im.convert('RGB')
        views = torch.stack([TRAIN_TRANSFORM(pil_im) for _ in range(10)])  # [10, 3, 256, 256]
    views = views.to(device)
    label = torch.full((10,), im['class_id'], device=device, dtype=torch.long)

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

    for g in groups:
        a_bar = assemble_a_bar(hm, g)
        values_RU = compute_regime_values(a_bar)
        accs[g].update(values_RU, img_id)

    return t1 - t0, t2 - t1


def run(vae, var, images, device, groups, checkpoint_path=None, checkpoint_every=200,
        resume=True, log=print):
    """
    groups: e.g. ['fc1', 'attn_proj', 'fc2'] (canonical abs() run) or
    ['fc1_clamp'] (clamp(0) comparison run -- requires 'fc1' in
    clamp_compare_groups when constructing the HookManager below).
    """
    clamp_needed = {_base_group(g) for g in groups if g.endswith('_clamp')}
    hm = HookManager(var, clamp_compare_groups=clamp_needed)
    hm.register()

    widths = {g: GROUP_WIDTH[_base_group(g)] * N_BLOCKS for g in groups}
    accs = {g: UnitMemAccumulator(widths[g]) for g in groups}
    start_idx = 0
    t_vae_total = 0.0
    t_var_total = 0.0

    if checkpoint_path and resume and os.path.exists(checkpoint_path):
        with open(checkpoint_path, 'rb') as f:
            ckpt = pickle.load(f)
        accs = ckpt['accs']
        start_idx = ckpt['next_idx']
        t_vae_total = ckpt['t_vae_total']
        t_var_total = ckpt['t_var_total']
        log(f'resumed from checkpoint {checkpoint_path} at image {start_idx}/{len(images)}')

    for img_id in range(start_idx, len(images)):
        dt_vae, dt_var = process_one_image(vae, var, hm, images[img_id], device, accs, img_id, groups)
        t_vae_total += dt_vae
        t_var_total += dt_var

        if (img_id + 1) % 20 == 0 or img_id == len(images) - 1:
            log(f'  {img_id+1}/{len(images)} images processed')

        if checkpoint_path and (img_id + 1) % checkpoint_every == 0:
            with open(checkpoint_path, 'wb') as f:
                pickle.dump({'accs': accs, 'next_idx': img_id + 1,
                             't_vae_total': t_vae_total, 't_var_total': t_var_total}, f)
            log(f'  checkpoint saved at {img_id+1}/{len(images)} -> {checkpoint_path}')

    hm.remove()
    return accs, t_vae_total, t_var_total


def finalize_and_pack(accs: dict) -> dict:
    payload = {}
    dead_report = {}
    for g, acc in accs.items():
        res = acc.finalize()
        for k, v in res.items():
            if k == 'regime_names':
                continue
            payload[f'{g}_{k}'] = v
        payload[f'{g}_regime_names'] = np.array(res['regime_names'])
        dead_report[g] = dict(zip(res['regime_names'], res['n_dead_per_regime'].tolist()))
    return payload, dead_report
