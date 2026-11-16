"""Seed Python and PyTorch consistently for model and loader execution."""

from __future__ import annotations

import random

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    """Seed supported RNGs and request deterministic cuDNN behavior."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
