"""Layer-wise (FBNet-style) search space of MobileNetV2-like inverted-residual blocks.

An architecture is a tuple of operator indices, one per searchable layer. The same
encoding is used by the evolutionary / NSGA-II searchers (hwnas.search) and by the
differentiable super-net (hwnas.supernet), so results are directly comparable.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Op:
    name: str
    kernel: int  # 0 for skip
    expand: int  # 0 for skip


OPS = (
    Op("skip", 0, 0),
    Op("mb_k3_e1", 3, 1),
    Op("mb_k3_e3", 3, 3),
    Op("mb_k3_e6", 3, 6),
    Op("mb_k5_e3", 5, 3),
    Op("mb_k5_e6", 5, 6),
)
SKIP = 0
Arch = tuple  # tuple[int, ...]


@dataclass(frozen=True)
class Layer:
    idx: int
    stage: int
    cin: int
    cout: int
    stride: int
    res_in: int
    res_out: int
    first_in_stage: bool

    @property
    def can_skip(self) -> bool:  # identity is only shape-compatible when nothing changes
        return self.stride == 1 and self.cin == self.cout

    @property
    def residual(self) -> bool:
        return self.can_skip


@dataclass(frozen=True)
class SpaceConfig:
    input_res: int = 32
    in_ch: int = 3
    stem_ch: int = 16
    stem_stride: int = 1
    # (out_channels, n_layers, stride_of_first_layer)
    stages: tuple = ((16, 1, 1), (24, 2, 2), (32, 3, 2), (64, 3, 2), (96, 2, 1))
    head_ch: int = 192
    num_classes: int = 10


class SearchSpace:
    def __init__(self, cfg: SpaceConfig | None = None):
        self.cfg = cfg or SpaceConfig()
        c = self.cfg
        layers, cin, res, idx = [], c.stem_ch, (c.input_res + c.stem_stride - 1) // c.stem_stride, 0
        for s, (cout, n, stride) in enumerate(c.stages):
            for j in range(n):
                st = stride if j == 0 else 1
                rout = (res + st - 1) // st
                layers.append(Layer(idx, s, cin, cout, st, res, rout, j == 0))
                cin, res, idx = cout, rout, idx + 1
        self.layers = tuple(layers)
        self.last_ch, self.final_res = cin, res
        self.valid_ops = tuple(
            tuple(i for i in range(len(OPS)) if i != SKIP or L.can_skip) for L in self.layers
        )

    @property
    def n_layers(self) -> int:
        return len(self.layers)

    def size(self) -> int:
        out = 1
        for v in self.valid_ops:
            out *= len(v)
        return out

    def is_valid(self, arch) -> bool:
        return len(arch) == self.n_layers and all(a in v for a, v in zip(arch, self.valid_ops))

    def random_arch(self, rng: np.random.Generator) -> Arch:
        return tuple(int(rng.choice(v)) for v in self.valid_ops)

    def mutate(self, arch: Arch, rng: np.random.Generator, p: float | None = None) -> Arch:
        p = 2.0 / self.n_layers if p is None else p
        out = list(arch)
        for i, v in enumerate(self.valid_ops):
            if rng.random() < p and len(v) > 1:
                out[i] = int(rng.choice([o for o in v if o != arch[i]]))
        if tuple(out) == tuple(arch):  # guarantee at least one change
            i = int(rng.integers(self.n_layers))
            out[i] = int(rng.choice([o for o in self.valid_ops[i] if o != arch[i]]))
        return tuple(out)

    def crossover(self, a: Arch, b: Arch, rng: np.random.Generator) -> Arch:
        mask = rng.random(self.n_layers) < 0.5
        return tuple(int(x if m else y) for x, y, m in zip(a, b, mask))

    def encode(self, arch: Arch) -> np.ndarray:
        """Flat one-hot encoding (n_layers * len(OPS))."""
        z = np.zeros((self.n_layers, len(OPS)))
        z[np.arange(self.n_layers), list(arch)] = 1.0
        return z.ravel()

    def describe(self, arch: Arch) -> str:
        return " ".join(f"{OPS[a].name.replace('mb_', '')}" for a in arch)
