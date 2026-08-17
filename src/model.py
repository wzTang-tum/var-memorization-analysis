"""
Model loading for the UnitMem-on-VAR-d16 measurement pipeline.

Loads VQVAE + VAR-d16 and forces the settings required for activation
measurement:
  - eval() on both models
  - cond_drop_rate = 0 (VAR.forward's label dropout is not guarded by
    self.training, so it must be disabled explicitly even under .eval())
  - blk.ffn.fused_mlp_func = None on every block (defensive: without this,
    a flash-attn-enabled environment would silently bypass the hooked
    fc1/fc2 modules)
"""
import os
import sys

import torch

VAR_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'VAR')
if VAR_ROOT not in sys.path:
    sys.path.insert(0, VAR_ROOT)

from models import build_vae_var  # noqa: E402


def build_and_load(device: str, checkpoints_dir: str, depth: int = 16):
    patch_nums = (1, 2, 3, 4, 5, 6, 8, 10, 13, 16)

    vae, var = build_vae_var(
        V=4096, Cvae=32, ch=160, share_quant_resi=4,
        device=device, patch_nums=patch_nums,
        num_classes=1000, depth=depth, shared_aln=False,
    )

    vae_ckpt = os.path.join(checkpoints_dir, 'vae_ch160v4096z32.pth')
    var_ckpt = os.path.join(checkpoints_dir, f'var_d{depth}.pth')
    vae.load_state_dict(torch.load(vae_ckpt, map_location='cpu'), strict=True)
    var.load_state_dict(torch.load(var_ckpt, map_location='cpu'), strict=True)

    vae.eval()
    var.eval()
    for p in vae.parameters():
        p.requires_grad_(False)
    for p in var.parameters():
        p.requires_grad_(False)

    # label dropout in VAR.forward is unconditional, not guarded by
    # self.training -- must zero it explicitly.
    var.cond_drop_rate = 0.0

    # guard against the silent fused-MLP hook bypass.
    n_fused_before = sum(b.ffn.fused_mlp_func is not None for b in var.blocks)
    for b in var.blocks:
        b.ffn.fused_mlp_func = None

    return vae, var, n_fused_before
