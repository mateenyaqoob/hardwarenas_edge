"""Illustrative roofline-style device profiles.

!! These numbers are NOT measurements of any specific board. They are order-of-magnitude
placeholders chosen so that the profiles disagree about which operators are cheap
(depthwise convs are launch/memory bound on the GPU-like profile, compute bound on the MCU).
Replace them with measured look-up tables (hwnas.measure) for real work.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .cost import Part


@dataclass(frozen=True)
class DeviceProfile:
    name: str
    peak_macs: float      # MAC/s at 100% efficiency
    mem_bw: float         # bytes/s to off-chip memory
    overhead: float       # seconds per kernel launch / layer
    cache_bytes: int      # tensors up to this size stay on chip
    dtype_bytes: int = 1  # int8 deployment
    eff: dict = field(default_factory=dict)  # kind -> fraction of peak achieved
    description: str = ""
    overlap: float = 0.0  # fraction of a block's first launch overhead hidden by async queuing


PROFILES = {
    "mcu": DeviceProfile("mcu", 4e8, 8e8, 15e-6, 128 * 1024, 1,
                         {"conv": .8, "pw": .9, "dw": .5, "fc": .7},
                         "Cortex-M7-class MCU, int8 kernels (illustrative)", 0.0),
    "rpi4": DeviceProfile("rpi4", 3e10, 6e9, 8e-6, 1024 * 1024, 1,
                          {"conv": .5, "pw": .6, "dw": .2, "fc": .3},
                          "Quad Cortex-A72 CPU, NEON int8 (illustrative)", 0.3),
    "edge_gpu": DeviceProfile("edge_gpu", 2.5e11, 2.5e10, 60e-6, 256 * 1024, 1,
                              {"conv": .5, "pw": .6, "dw": .08, "fc": .2},
                              "Jetson-Nano-class embedded GPU (illustrative)", 0.6),
}


def part_latency(p: Part, prof: DeviceProfile, cold_input: bool = False) -> float:
    """Roofline: max(compute time, memory time) + launch overhead.

    Weights always stream from off-chip memory. Activations only cost memory traffic when
    they do not fit in 'cache_bytes' -- or, for look-up-table entries measured in isolation,
    when the input is read cold (cold_input=True)."""
    db = prof.dtype_bytes
    compute = p.macs / (prof.peak_macs * prof.eff[p.kind])
    traffic = p.params * db
    in_b, out_b = p.in_elems * db, p.out_elems * db
    if cold_input or in_b > prof.cache_bytes:
        traffic += in_b
    if out_b > prof.cache_bytes:
        traffic += out_b
    return max(compute, traffic / prof.mem_bw) + prof.overhead
