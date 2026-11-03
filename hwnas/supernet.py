"""Gumbel-softmax super-net (FBNet-style) with a differentiable, LUT-based latency estimate."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .latency import LUT
from .space import Arch, SearchSpace
from .torch_nets import make_head, make_op, make_stem


class SuperNet(nn.Module):
    def __init__(self, space: SearchSpace, lut: LUT):
        super().__init__()
        self.space = space
        self.stem, self.head = make_stem(space), make_head(space)
        self.layers = nn.ModuleList(
            nn.ModuleList(make_op(L, a) for a in space.valid_ops[L.idx]) for L in space.layers)
        self.thetas = nn.ParameterList(nn.Parameter(1e-3 * torch.randn(len(v))) for v in space.valid_ops)
        for l, v in enumerate(space.valid_ops):
            self.register_buffer(f"lut_{l}", torch.tensor([lut.table[l, a] for a in v], dtype=torch.float32))
        self.base = float(lut.base)

    def arch_parameters(self):
        return list(self.thetas.parameters())

    def weight_parameters(self):
        ids = {id(p) for p in self.arch_parameters()}
        return [p for p in self.parameters() if id(p) not in ids]

    def sample_weights(self, tau: float, hard: bool = False):
        return [F.gumbel_softmax(t, tau=tau, hard=hard) for t in self.thetas]

    def forward(self, x, tau: float = 5.0, ws=None):
        ws = ws if ws is not None else self.sample_weights(tau)
        x = self.stem(x)
        for ops, w in zip(self.layers, ws):
            x = sum(wi * op(x) for wi, op in zip(w, ops))
        return self.head(x), ws

    def expected_latency(self, ws) -> torch.Tensor:
        """E[latency] = base + sum_l sum_i w_{l,i} * LUT[l, i]   (differentiable in theta)."""
        return self.base + sum((w * getattr(self, f"lut_{l}")).sum() for l, w in enumerate(ws))

    @torch.no_grad()
    def derive(self) -> Arch:
        return tuple(int(self.space.valid_ops[l][int(t.argmax())]) for l, t in enumerate(self.thetas))
