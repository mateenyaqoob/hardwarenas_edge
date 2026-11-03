"""Latency models: look-up table (LUT), end-to-end simulator, and a learned corrector."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .cost import block_parts, head_parts, stem_parts
from .devices import DeviceProfile, part_latency
from .space import OPS, Arch, SearchSpace


class LUT:
    """latency(arch) = base + sum_l table[l, arch[l]]  (additivity assumption)."""

    def __init__(self, table: np.ndarray, base: float, meta: dict | None = None):
        self.table, self.base, self.meta = np.asarray(table, float), float(base), meta or {}

    def latency(self, arch: Arch) -> float:
        return self.base + float(sum(self.table[l, a] for l, a in enumerate(arch)))

    def save(self, path) -> None:
        Path(path).write_text(json.dumps(dict(table=np.nan_to_num(self.table, nan=-1).tolist(),
                                              base=self.base, meta=self.meta), indent=1))

    @classmethod
    def load(cls, path) -> "LUT":
        d = json.loads(Path(path).read_text())
        t = np.array(d["table"], float)
        t[t < 0] = np.nan
        return cls(t, d["base"], d.get("meta"))


def build_lut(space: SearchSpace, prof: DeviceProfile) -> LUT:
    """Per-(layer, op) latency measured *in isolation*: each block reads its input cold."""
    table = np.full((space.n_layers, len(OPS)), np.nan)
    for L in space.layers:
        for a in space.valid_ops[L.idx]:
            parts = block_parts(L, OPS[a])
            table[L.idx, a] = sum(part_latency(p, prof, cold_input=(i == 0))
                                  for i, p in enumerate(parts))
    base = sum(part_latency(p, prof, cold_input=True) for p in stem_parts(space) + head_parts(space))
    return LUT(table, base, dict(device=prof.name, source="roofline-sim"))


def sim_latency(space: SearchSpace, arch: Arch, prof: DeviceProfile) -> float:
    """Whole-network 'ground truth' with two cross-layer effects the isolated LUT cannot see:
    (1) on-chip activations are not re-read from DRAM; (2) asynchronous runtimes hide part of
    each block's first launch overhead ('overlap'). The LUT therefore slightly over-estimates."""
    total = sum(part_latency(p, prof) for p in stem_parts(space))
    for L, a in zip(space.layers, arch):
        parts = block_parts(L, OPS[a])
        total += sum(part_latency(p, prof) for p in parts)
        if parts:
            total -= prof.overlap * prof.overhead
    return total + sum(part_latency(p, prof) for p in head_parts(space))


def kendall_tau(a, b) -> float:
    a, b = np.asarray(a), np.asarray(b)
    s = np.sign(a[:, None] - a[None, :]) * np.sign(b[:, None] - b[None, :])
    n = len(a)
    return float(s.sum() / (n * (n - 1)))


def mape(pred, true) -> float:
    pred, true = np.asarray(pred), np.asarray(true)
    return float(np.mean(np.abs(pred - true) / true))


class LatencyPredictor:
    """Ridge regression on [LUT sum, one-hot architecture]; corrects systematic LUT bias."""

    def __init__(self, space: SearchSpace, lut: LUT, l2: float = 1e-3):
        self.space, self.lut, self.l2 = space, lut, l2

    def _x(self, archs):
        return np.array([[self.lut.latency(a)] + list(self.space.encode(a)) for a in archs])

    def fit(self, archs, y) -> "LatencyPredictor":
        X = self._x(archs)
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-12
        Z = np.c_[(X - self.mu) / self.sd, np.ones(len(X))]
        reg = self.l2 * np.eye(Z.shape[1]); reg[-1, -1] = 0
        self.w = np.linalg.solve(Z.T @ Z + reg, Z.T @ np.asarray(y))
        return self

    def predict(self, archs) -> np.ndarray:
        X = self._x(archs)
        return np.c_[(X - self.mu) / self.sd, np.ones(len(X))] @ self.w
