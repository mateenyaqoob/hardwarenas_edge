"""Zero-cost proxies: score an architecture at initialisation, with no training."""
from __future__ import annotations

import copy
import math

import torch
import torch.nn as nn


def synflow(model: nn.Module, input_shape, device="cpu") -> float:
    """log10 SynFlow score (Tanaka et al., 2020): data-free, sum_theta |theta * dR/dtheta|
    with R = sum(output) for an all-ones input, |weights|, BatchNorm removed."""
    m = copy.deepcopy(model).to(device).double().eval()

    def strip_bn(mod):
        for n, ch in mod.named_children():
            if isinstance(ch, nn.BatchNorm2d):
                setattr(mod, n, nn.Identity())
            else:
                strip_bn(ch)
    strip_bn(m)
    with torch.no_grad():
        for p in m.parameters():
            p.abs_()
    x = torch.ones(1, *input_shape[1:], dtype=torch.double, device=device)
    m.zero_grad()
    m(x).sum().backward()
    score = sum((p * p.grad).abs().sum().item() for p in m.parameters() if p.grad is not None)
    return math.log10(max(score, 1e-30))
