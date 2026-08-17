"""Shared constants: multi-scale token boundaries and dataset sampling parameters."""

PATCH_NUMS = (1, 2, 3, 4, 5, 6, 8, 10, 13, 16)

BOUNDS = [0, 1, 5, 14, 30, 55, 91, 155, 255, 424, 680]
N_S = [1, 4, 9, 16, 25, 36, 64, 100, 169, 256]

assert [BOUNDS[s + 1] - BOUNDS[s] for s in range(10)] == N_S
assert BOUNDS[-1] == sum(N_S) == 680

# --- data subset ---
SEED = 42
N_CLASSES = 100
N_PER_CLASS = 10
CANDIDATE_MARGIN = 12  # stop scanning a class once it has this many candidates

HF_DATASET = 'ILSVRC/imagenet-1k'
TOTAL_TRAIN_SHARDS = 294
TOTAL_CLASSES = 1000
