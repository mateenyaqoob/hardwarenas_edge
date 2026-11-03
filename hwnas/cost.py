"""Analytical cost model: MACs, parameters, activation working set per block.

Everything is derived from layer shapes, so it matches the torch networks built by
hwnas.torch_nets exactly (a unit test checks the parameter count against torch).
"""
from __future__ import annotations

from dataclasses import dataclass

from .space import OPS, Arch, Layer, Op, SearchSpace


@dataclass(frozen=True)
class Part:
    """One primitive kernel launch: a convolution or FC layer (BN params folded in)."""
    kind: str       # 'conv' | 'pw' | 'dw' | 'fc'
    macs: int
    params: int
    in_elems: int
    out_elems: int


def block_parts(L: Layer, op: Op) -> list[Part]:
    if op.kernel == 0:
        return []
    k, e = op.kernel, op.expand
    m = L.cin * e
    hi, ho = L.res_in ** 2, L.res_out ** 2
    parts = []
    if e > 1:
        parts.append(Part("pw", hi * L.cin * m, L.cin * m + 2 * m, L.cin * hi, m * hi))
    parts.append(Part("dw", ho * m * k * k, m * k * k + 2 * m, m * hi, m * ho))
    parts.append(Part("pw", ho * m * L.cout, m * L.cout + 2 * L.cout, m * ho, L.cout * ho))
    return parts


def block_peak_elems(L: Layer, op: Op) -> int:
    """Peak number of live activation elements while executing the block."""
    parts = block_parts(L, op)
    if not parts:
        return L.cin * L.res_in ** 2
    extra = L.cin * L.res_in ** 2 if L.residual else 0  # residual input stays alive
    peak = 0
    for i, p in enumerate(parts):
        peak = max(peak, p.in_elems + p.out_elems + (extra if i > 0 else 0))
    return peak


def stem_parts(space: SearchSpace) -> list[Part]:
    c = space.cfg
    r = space.layers[0].res_in
    return [Part("conv", r * r * 9 * c.in_ch * c.stem_ch, 9 * c.in_ch * c.stem_ch + 2 * c.stem_ch,
                 c.in_ch * c.input_res ** 2, c.stem_ch * r * r)]


def head_parts(space: SearchSpace) -> list[Part]:
    c, r, last = space.cfg, space.final_res, space.last_ch
    return [Part("pw", r * r * last * c.head_ch, last * c.head_ch + 2 * c.head_ch,
                 last * r * r, c.head_ch * r * r),
            Part("fc", c.head_ch * c.num_classes, c.head_ch * c.num_classes + c.num_classes,
                 c.head_ch, c.num_classes)]


def arch_parts(space: SearchSpace, arch: Arch) -> list[Part]:
    parts = stem_parts(space)
    for L, a in zip(space.layers, arch):
        parts += block_parts(L, OPS[a])
    return parts + head_parts(space)


def arch_metrics(space: SearchSpace, arch: Arch, dtype_bytes: int = 1) -> dict:
    parts = arch_parts(space, arch)
    peak = max(max(p.in_elems + p.out_elems for p in stem_parts(space) + head_parts(space)),
               max(block_peak_elems(L, OPS[a]) for L, a in zip(space.layers, arch)))
    params = sum(p.params for p in parts)
    return dict(macs=sum(p.macs for p in parts), params=params,
                flash_bytes=params * dtype_bytes, peak_sram_bytes=peak * dtype_bytes)
