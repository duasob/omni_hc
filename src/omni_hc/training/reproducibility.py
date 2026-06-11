from __future__ import annotations

import random

import numpy as np
import torch


DEFAULT_SEED = 42
DATA_SPLIT_SEED_OFFSET = 0
DATA_SHUFFLE_SEED_OFFSET = 1


def training_seed(cfg: dict) -> int:
    return int((cfg.get("training") or {}).get("seed", DEFAULT_SEED))


def seed_everything(seed: int) -> None:
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seeded_generator(seed: int, *, offset: int = 0) -> torch.Generator:
    return torch.Generator(device="cpu").manual_seed(int(seed) + int(offset))
