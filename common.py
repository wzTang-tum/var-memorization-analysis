"""
Shared post-processing helpers for the RQ1/RQ2 analysis stages, reused
identically by both so the two never diverge on how the pooled top-10%
unit set is defined.

Pure post-processing of the precomputed unitmem_N10000_global.npz
(read-only): no model forward pass here.

Score definition: for each unit u in a group,
    S_u = mean(UnitMem_u,B_1, UnitMem_u,B_2, ..., UnitMem_u,B_9)
i.e. the per-scale UnitMem values (already computed, stored in the B_1..B_9
rows of each group's `unitmem` array) averaged after being computed --
not the scale-averaged-activation regime, not the token-count-weighted
regime, and not scale0 (pure class-condition, architecturally excluded
since position 0 can only attend to itself). Only rows B_1..B_9 are read
from the NPZ anywhere in this module.

A unit is excluded from the candidate pool if it is dead (mu_max < eps) in
any of its 9 B_1..B_9 regimes -- if even one of the 9 terms being averaged
is undefined for a dead unit, the whole mean S_u is contaminated, so the
unit is dropped entirely rather than averaging in a forced-0 value.
"""
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
RESULTS_ACTIVATIONS = os.path.join(ROOT, 'results')
RESULTS_ANALYSIS = os.path.join(ROOT, 'results', 'analysis')
FIG_ANALYSIS = os.path.join(ROOT, 'figures')
os.makedirs(RESULTS_ANALYSIS, exist_ok=True)
os.makedirs(FIG_ANALYSIS, exist_ok=True)

N_BLOCKS = 16
GROUPS = ['fc1', 'attn_proj', 'fc2']
GROUP_WIDTH = {'fc1': 4096, 'attn_proj': 1024, 'fc2': 1024}
# fixed group order for the global tie-break unit index (fc1 first, then
# attn_proj, then fc2), independent of alive/dead status
GROUP_OFFSET = {}
_off = 0
for _g in GROUPS:
    GROUP_OFFSET[_g] = _off
    _off += GROUP_WIDTH[_g] * N_BLOCKS
TOTAL_UNITS = _off  # 98304
assert TOTAL_UNITS == 65536 + 16384 + 16384


def load_npz():
    return np.load(os.path.join(RESULTS_ACTIVATIONS, 'unitmem_N10000_global.npz'))


def load_manifest():
    with open(os.path.join(RESULTS_ACTIVATIONS, 'dataset_manifest_N10000.json'), encoding='utf-8') as f:
        return json.load(f)


def b_rows(data, group):
    """Row indices of B_1..B_9 in this group's regime_names, in scale order 1..9."""
    names = list(data[f'{group}_regime_names'])
    rows = [names.index(f'B_{s}') for s in range(1, 10)]
    assert rows == list(range(3, 12)), f'unexpected B_1..B_9 row layout for {group}: {rows}'
    return rows


def compute_group_scores(data, group):
    """
    Returns dict with, for group `group`, U = GROUP_WIDTH[group]*N_BLOCKS units:
      S_u        [U] float64 -- mean(UnitMem_B_1..B_9), NaN where dead_any
      dead_any   [U] bool    -- dead (mu_max<eps) in >=1 of B_1..B_9
      unitmem_B  [9, U] float32 -- the raw B_1..B_9 UnitMem rows (for spot-check)
      arg_max_B  [9, U] int32   -- the raw B_1..B_9 arg_max rows (for RQ2 votes)
      block_idx  [U] int -- u // GROUP_WIDTH[group] (block-major, matching
                            src/pipeline.py's assemble_a_bar: torch.cat(parts, dim=2)
                            concatenates per-block tensors along the channel axis in
                            block order, so unit u's block is u // channels-per-block)
    """
    rows = b_rows(data, group)
    unitmem_B = data[f'{group}_unitmem'][rows]      # [9, U]
    dead_B = data[f'{group}_dead_mask'][rows]        # [9, U]
    arg_max_B = data[f'{group}_arg_max'][rows]        # [9, U]
    U = unitmem_B.shape[1]
    assert U == GROUP_WIDTH[group] * N_BLOCKS

    dead_any = dead_B.any(axis=0)
    S_u = unitmem_B.astype(np.float64).mean(axis=0)
    S_u = np.where(dead_any, np.nan, S_u)

    block_idx = np.arange(U) // GROUP_WIDTH[group]

    return {
        'S_u': S_u,
        'dead_any': dead_any,
        'unitmem_B': unitmem_B,
        'dead_B': dead_B,
        'arg_max_B': arg_max_B,
        'block_idx': block_idx,
        'U': U,
    }


def build_pool(data):
    """Pool alive units across fc1/attn_proj/fc2 into flat arrays, plus per-group
    per-group-only info needed later (kept separately, not just pooled)."""
    per_group = {g: compute_group_scores(data, g) for g in GROUPS}

    pool_S = []
    pool_group = []
    pool_local_idx = []
    pool_global_id = []
    pool_block = []

    for g in GROUPS:
        info = per_group[g]
        alive_idx = np.where(~info['dead_any'])[0]
        pool_S.append(info['S_u'][alive_idx])
        pool_group.extend([g] * len(alive_idx))
        pool_local_idx.append(alive_idx)
        pool_global_id.append(GROUP_OFFSET[g] + alive_idx)
        pool_block.append(info['block_idx'][alive_idx])

    pool = {
        'S': np.concatenate(pool_S),
        'group': np.array(pool_group),
        'local_idx': np.concatenate(pool_local_idx),
        'global_id': np.concatenate(pool_global_id),
        'block': np.concatenate(pool_block),
    }
    return per_group, pool


def select_top_fraction(pool, frac=0.10):
    """Sort by (-S, global_id) ascending (deterministic tie-break), select
    floor(frac * n_alive). Returns selection mask over the pool arrays plus
    boundary diagnostics."""
    n_alive = len(pool['S'])
    k = int(np.floor(frac * n_alive))

    order = np.lexsort((pool['global_id'], -pool['S']))  # last key primary: -S, then global_id
    selected_order = order[:k]
    sel_mask = np.zeros(n_alive, dtype=bool)
    sel_mask[selected_order] = True

    boundary_score = pool['S'][order[k - 1]] if k > 0 else None
    # count how many alive units (selected or not) share the exact boundary score
    n_tied_at_boundary = int(np.sum(pool['S'] == boundary_score)) if k > 0 else 0
    n_selected_at_boundary = int(np.sum(pool['S'][selected_order] == boundary_score)) if k > 0 else 0

    return {
        'k': k,
        'n_alive': n_alive,
        'sel_mask': sel_mask,
        'order': order,
        'boundary_score': boundary_score,
        'n_tied_at_boundary_total': n_tied_at_boundary,
        'n_tied_at_boundary_selected': n_selected_at_boundary,
    }
