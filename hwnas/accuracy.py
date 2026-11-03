"""Accuracy sources.

SyntheticAccuracy is a deterministic, *made-up* function with diminishing returns in model
capacity. It exists so search algorithms can be developed, tested and compared in seconds
without training anything. NEVER report its numbers as real accuracy. Real sources:
short-training proxies, a weight-sharing super-net, or a tabular benchmark.
"""
from __future__ import annotations

import zlib

import numpy as np

from .space import OPS, Arch, SearchSpace

_CAP = {"skip": 0.0, "mb_k3_e1": 1.0, "mb_k3_e3": 2.2, "mb_k3_e6": 3.0, "mb_k5_e3": 2.6, "mb_k5_e6": 3.5}


class SyntheticAccuracy:
    def __init__(self, space: SearchSpace, noise: float = 0.15):
        self.space, self.noise = space, noise
        n = space.n_layers
        self.w = 0.6 + 0.8 * np.arange(n) / (n - 1)  # deeper layers matter more

    def __call__(self, arch: Arch) -> float:
        cap = sum(w * _CAP[OPS[a].name] for w, a in zip(self.w, arch))
        base = 55.0 + 38.0 * (1.0 - np.exp(-cap / 28.0))
        seed = zlib.crc32(np.asarray(arch, dtype=np.int8).tobytes())
        return float(base + np.random.default_rng(seed).normal(0.0, self.noise))
