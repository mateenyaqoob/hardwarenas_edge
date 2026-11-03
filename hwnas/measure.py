"""On-host latency measurement and measured-LUT construction (the 'real' counterpart of build_lut)."""
from __future__ import annotations

import statistics
import time

import numpy as np
import torch

from .latency import LUT
from .space import OPS, SearchSpace
from .torch_nets import make_head, make_op, make_stem


@torch.inference_mode()
def measure_latency(module, input_shape, warmup=10, iters=50, threads=1) -> float:
    """Median wall-clock seconds of one forward pass (batch 1 by default)."""
    prev = torch.get_num_threads()
    torch.set_num_threads(threads)
    try:
        module.eval()
        x = torch.randn(*input_shape)
        for _ in range(warmup):
            module(x)
        ts = []
        for _ in range(iters):
            t = time.perf_counter(); module(x); ts.append(time.perf_counter() - t)
        return statistics.median(ts)
    finally:
        torch.set_num_threads(prev)


def build_measured_lut(space: SearchSpace, iters=50, warmup=10, threads=1) -> LUT:
    """Measure every (layer, op) block in isolation on this machine. Run it on the TARGET device
    (e.g. a Raspberry Pi) to get a LUT that reflects that device's kernels and memory system."""
    c = space.cfg
    table = np.full((space.n_layers, len(OPS)), np.nan)
    for L in space.layers:
        for a in space.valid_ops[L.idx]:
            table[L.idx, a] = 0.0 if a == 0 else measure_latency(
                make_op(L, a), (1, L.cin, L.res_in, L.res_in), warmup, iters, threads)
    base = (measure_latency(make_stem(space), (1, c.in_ch, c.input_res, c.input_res), warmup, iters, threads)
            + measure_latency(make_head(space), (1, space.last_ch, space.final_res, space.final_res),
                              warmup, iters, threads))
    return LUT(table, base, dict(device="host-cpu", source="torch-measured", torch=torch.__version__,
                                 threads=threads))
