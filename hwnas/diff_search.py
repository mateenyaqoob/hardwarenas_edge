"""Differentiable hardware-aware search (FBNet / ProxylessNAS flavour).

Alternating optimisation: weights on split A, architecture logits (theta) on split B.
The loss couples accuracy and the LUT-based expected latency:
  fbnet : CE * alpha * log(latency_us)^beta            (multiplicative, FBNet-style)
  hinge : CE + lam * relu(E[lat]/target - 1)           (only penalises violating the budget)
  linear: CE + lam * E[lat]/target  (or lam * E[lat] in ms if no target)  (ProxylessNAS-style)
"""
from __future__ import annotations

import itertools
import math

import torch
import torch.nn.functional as F

from .latency import LUT
from .space import SearchSpace
from .supernet import SuperNet


def latency_loss(ce, lat, kind="fbnet", alpha=0.2, beta=0.6, lam=1.0, target=None):
    if kind == "fbnet":
        return ce * alpha * torch.log((lat * 1e6).clamp_min(1.5)) ** beta
    if kind == "hinge":
        assert target, "hinge loss needs a latency target (seconds)"
        return ce + lam * F.relu(lat / target - 1.0)
    if kind == "linear":
        return ce + lam * (lat / target if target else lat * 1e3)
    raise ValueError(kind)


def differentiable_search(space: SearchSpace, lut: LUT, loader_w, loader_a, epochs=20, warmup=5,
                          lat_loss="fbnet", target=None, alpha=0.2, beta=0.6, lam=1.0,
                          tau0=5.0, tau_min=0.5, lr_w=0.05, lr_a=3e-3, device="cpu", verbose=True):
    net = SuperNet(space, lut).to(device)
    opt_w = torch.optim.SGD(net.weight_parameters(), lr=lr_w, momentum=0.9, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt_w, epochs)
    opt_a = torch.optim.Adam(net.arch_parameters(), lr=lr_a, betas=(0.5, 0.999), weight_decay=5e-4)
    log = []
    for ep in range(epochs):
        tau = tau0 * (tau_min / tau0) ** (ep / max(1, epochs - 1))
        net.train()
        tot, lats, n = 0.0, 0.0, 0
        for (xw, yw), (xa, ya) in zip(loader_w, itertools.cycle(loader_a)):
            xw, yw, xa, ya = xw.to(device), yw.to(device), xa.to(device), ya.to(device)
            logits, _ = net(xw, tau)                              # 1) update weights
            loss_w = F.cross_entropy(logits, yw)
            opt_w.zero_grad(set_to_none=True); loss_w.backward(); opt_w.step()
            tot, n = tot + loss_w.item(), n + 1
            if ep >= warmup:                                      # 2) update architecture logits
                logits, ws = net(xa, tau)
                lat = net.expected_latency(ws)
                loss_a = latency_loss(F.cross_entropy(logits, ya), lat, lat_loss, alpha, beta, lam, target)
                opt_a.zero_grad(set_to_none=True); loss_a.backward(); opt_a.step()
                lats += float(lat.detach())
        sched.step()
        arch = net.derive()
        rec = dict(epoch=ep, tau=round(tau, 3), train_loss=round(tot / max(n, 1), 4),
                   expected_lat_ms=round(1e3 * lats / max(n, 1), 4) if ep >= warmup else None,
                   derived_lut_ms=round(1e3 * lut.latency(arch), 4), arch=list(arch))
        log.append(rec)
        if verbose:
            print(rec)
    return net.derive(), log
