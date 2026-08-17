"""
Forward-hook registration and per-scale activation segmentation.

Reduces dimensionality inside the hook itself: raw activations
[B, 680, C] are never retained past the hook call. Only the
per-scale-averaged [B, 10, C] tensor is cached.
"""
import torch

from config import BOUNDS

GROUPS = ('fc1', 'attn_proj', 'fc2')


class HookManager:
    def __init__(self, var, clamp_compare_groups=()):
        """
        clamp_compare_groups: group names (e.g. {'fc1'}) for which an
        extra parallel clamp(min=0)-based reduction is cached under
        '<group>_clamp', computed from the same raw activation as the
        normal abs()-based one. Off by default -- the production path only
        computes abs(); this is used for an abs-vs-clamp ranking-sensitivity
        check, not the main run.
        """
        self.var = var
        self.cache = {}        # (group, block_idx) -> [B, 10, C] float tensor, abs() applied
        self.hook_calls = {g: 0 for g in GROUPS}
        self.clamp_compare_groups = set(clamp_compare_groups)
        self._handles = []

    def _make_hook(self, group, block_idx):
        do_clamp = group in self.clamp_compare_groups

        def hook(module, inp, out):
            self.hook_calls[group] += 1
            raw = out.detach().float()
            a = raw.abs()
            self.cache[(group, block_idx)] = torch.stack(
                [a[:, BOUNDS[s]:BOUNDS[s + 1], :].mean(dim=1) for s in range(10)], dim=1
            )  # -> [B, 10, C]
            if do_clamp:
                c = raw.clamp(min=0)
                self.cache[(group + '_clamp', block_idx)] = torch.stack(
                    [c[:, BOUNDS[s]:BOUNDS[s + 1], :].mean(dim=1) for s in range(10)], dim=1
                )
        return hook

    def register(self):
        for i, blk in enumerate(self.var.blocks):
            self._handles.append(blk.ffn.act.register_forward_hook(self._make_hook('fc1', i)))
            self._handles.append(blk.attn.proj.register_forward_hook(self._make_hook('attn_proj', i)))
            self._handles.append(blk.ffn.fc2.register_forward_hook(self._make_hook('fc2', i)))

    def remove(self):
        for h in self._handles:
            h.remove()
        self._handles = []

    def reset(self):
        self.cache = {}
        self.hook_calls = {g: 0 for g in GROUPS}
