"""
Streaming UnitMem accumulators and regime computation.

Never store all activations: each image contributes exactly one [R, U]
row of scores per layer group, folded into a running sum/max/argmax.
"""
import numpy as np

from config import N_S

REGIME_NAMES = ['A', 'T', 'scale0'] + [f'B_{s}' for s in range(1, 10)]  # 12 regimes
A_IDX = REGIME_NAMES.index('A')
T_WEIGHTS = np.array(N_S[1:], dtype=np.float64) / 680.0  # Regime T token-count weighting


def compute_regime_values(a_bar_scales_U: np.ndarray) -> np.ndarray:
    """
    a_bar_scales_U: [10, U] float64 -- view-averaged, abs() (or clamp())-applied
    per-scale activation for ONE image.
    Returns [12, U]: rows ordered per REGIME_NAMES (A, T, scale0, B_1..B_9).
    """
    scale0 = a_bar_scales_U[0]
    macro_A = a_bar_scales_U[1:].mean(axis=0)                        # Regime A
    global_T = (a_bar_scales_U[1:] * T_WEIGHTS[:, None]).sum(axis=0)  # Regime T
    B_per_scale = [a_bar_scales_U[s] for s in range(1, 10)]           # Regime B (per-scale)
    return np.stack([macro_A, global_T, scale0, *B_per_scale], axis=0)


class UnitMemAccumulator:
    """One instance per layer group (fc1 / attn_proj / fc2)."""

    def __init__(self, U: int, top_k: int = 10, regime_names=None):
        self.U = U
        self.regime_names = list(regime_names) if regime_names is not None else list(REGIME_NAMES)
        self.R = len(self.regime_names)
        self.a_idx = self.regime_names.index('A') if 'A' in self.regime_names else None
        self.run_sum = np.zeros((self.R, U), dtype=np.float64)
        self.run_max = np.full((self.R, U), -np.inf, dtype=np.float32)
        self.arg_max = np.zeros((self.R, U), dtype=np.int32)
        self.n_images = 0
        self.top_k = top_k
        if self.a_idx is not None:
            self.top_v = np.full((U, top_k), -np.inf, dtype=np.float32)
            self.top_i = np.zeros((U, top_k), dtype=np.int32)

    def update(self, values_RU: np.ndarray, img_id: int):
        assert values_RU.shape == (self.R, self.U)
        values_RU = values_RU.astype(np.float64)
        self.run_sum += values_RU
        improved = values_RU > self.run_max
        self.arg_max = np.where(improved, img_id, self.arg_max)
        self.run_max = np.where(improved, values_RU, self.run_max).astype(np.float32)
        self.n_images += 1

        if self.a_idx is not None:
            a_vals = values_RU[self.a_idx].astype(np.float32)
            cat_v = np.concatenate([self.top_v, a_vals[:, None]], axis=1)
            cat_i = np.concatenate([self.top_i, np.full((self.U, 1), img_id, np.int32)], axis=1)
            idx = np.argsort(-cat_v, axis=1)[:, :self.top_k]
            self.top_v = np.take_along_axis(cat_v, idx, axis=1)
            self.top_i = np.take_along_axis(cat_i, idx, axis=1)

    def finalize(self, eps=1e-8):
        N = self.n_images
        mu_max = self.run_max.astype(np.float64)
        mu_nmax = (self.run_sum - self.run_max) / (N - 1)
        denom = mu_max + mu_nmax
        dead = mu_max < eps
        with np.errstate(divide='ignore', invalid='ignore'):
            unitmem = (mu_max - mu_nmax) / np.where(denom == 0, 1.0, denom)
        unitmem = np.where(dead, 0.0, unitmem).astype(np.float32)
        out = {
            'regime_names': self.regime_names,
            'unitmem': unitmem,                      # [R, U]
            'mu_max': mu_max.astype(np.float32),      # [R, U]
            'mu_nmax': mu_nmax.astype(np.float32),    # [R, U]
            'arg_max': self.arg_max,                  # [R, U]
            'dead_mask': dead,                        # [R, U]
            'n_dead_per_regime': dead.sum(axis=1),    # [R]
        }
        if self.a_idx is not None:
            out['top_v'] = self.top_v
            out['top_i'] = self.top_i
        return out
