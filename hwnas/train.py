from __future__ import annotations

import torch
import torch.nn.functional as F

from .cost import arch_metrics
from .measure import measure_latency
from .space import Arch, SearchSpace
from .torch_nets import build_network


def train_arch(space: SearchSpace, arch: Arch, train_loader, test_loader, epochs=30, lr=0.05,
               wd=5e-4, label_smoothing=0.1, device="cpu") -> dict:
    """Train a discovered architecture from scratch; report accuracy, cost and host latency."""
    net = build_network(space, arch).to(device)
    opt = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=wd, nesterov=True)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    for _ in range(epochs):
        net.train()
        for x, y in train_loader:
            loss = F.cross_entropy(net(x.to(device)), y.to(device), label_smoothing=label_smoothing)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        sched.step()
    net.eval(); correct = total = 0
    with torch.no_grad():
        for x, y in test_loader:
            correct += (net(x.to(device)).argmax(1).cpu() == y).sum().item(); total += len(y)
    m = arch_metrics(space, arch)
    host = measure_latency(net.cpu(), (1, space.cfg.in_ch, space.cfg.input_res, space.cfg.input_res), iters=30)
    return dict(ops=space.describe(arch), test_acc=round(correct / total, 4), params=m["params"],
                macs_M=round(m["macs"] / 1e6, 2), peak_sram_kb=round(m["peak_sram_bytes"] / 1024, 1),
                host_cpu_ms=round(host * 1e3, 3))
