"""Constrained random / evolutionary / NSGA-II search.

Key practical point: latency, SRAM and flash checks are cheap (LUT + analytic cost) and are
done BEFORE the expensive accuracy evaluation; the search budget counts accuracy evaluations.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .cost import arch_metrics
from .space import Arch, SearchSpace


@dataclass(frozen=True)
class Candidate:
    arch: Arch
    acc: float
    lat: float
    sram: int
    flash: int
    macs: int


@dataclass(frozen=True)
class Constraints:
    max_latency: float = float("inf")   # seconds
    max_sram: float = float("inf")      # bytes (peak activation working set)
    max_flash: float = float("inf")     # bytes (weights)


class Evaluator:
    def __init__(self, space: SearchSpace, acc_fn, lat_fn, dtype_bytes: int = 1):
        self.space, self.acc_fn, self.lat_fn, self.db = space, acc_fn, lat_fn, dtype_bytes
        self._cheap, self._full = {}, {}

    def cheap(self, arch: Arch) -> dict:
        if arch not in self._cheap:
            m = arch_metrics(self.space, arch, self.db)
            m["lat"] = float(self.lat_fn(arch))
            self._cheap[arch] = m
        return self._cheap[arch]

    def feasible(self, arch: Arch, c: Constraints) -> bool:
        m = self.cheap(arch)
        return m["lat"] <= c.max_latency and m["peak_sram_bytes"] <= c.max_sram and m["flash_bytes"] <= c.max_flash

    def evaluate(self, arch: Arch) -> Candidate:
        if arch not in self._full:
            m = self.cheap(arch)
            self._full[arch] = Candidate(arch, float(self.acc_fn(arch)), m["lat"],
                                         m["peak_sram_bytes"], m["flash_bytes"], m["macs"])
        return self._full[arch]

    @property
    def n_acc_evals(self) -> int:
        return len(self._full)

    def all_evaluated(self) -> list[Candidate]:
        return list(self._full.values())


def sample_feasible(space, ev, cons, rng, max_tries: int = 20000) -> Arch:
    for _ in range(max_tries):
        a = space.random_arch(rng)
        if ev.feasible(a, cons):
            return a
    raise RuntimeError("no feasible architecture found; constraints too tight for this space/device")


def mnas_reward(acc: float, lat: float, target: float, w: float = -0.07) -> float:
    """MnasNet soft constraint: ACC * (LAT/T)^w  (w=-1 approximates a hard constraint)."""
    return acc * (lat / target) ** w


def random_search(space, ev, cons, budget, rng) -> tuple[Candidate, list[float]]:
    best, hist = None, []
    for _ in range(budget):
        c = ev.evaluate(sample_feasible(space, ev, cons, rng))
        best = c if best is None or c.acc > best.acc else best
        hist.append(best.acc)
    return best, hist


def _child(space, ev, cons, pop, rng, k=3, p_cx=0.5, tries=50) -> Arch:
    def pick():
        idx = rng.choice(len(pop), size=min(k, len(pop)), replace=False)
        return max((pop[i] for i in idx), key=lambda c: c.acc).arch
    for _ in range(tries):
        a = pick()
        if rng.random() < p_cx:
            a = space.crossover(a, pick(), rng)
        a = space.mutate(a, rng)
        if ev.feasible(a, cons):
            return a
    return sample_feasible(space, ev, cons, rng)


def evolutionary_search(space, ev, cons, budget, rng, pop_size=40) -> tuple[Candidate, list[float]]:
    """(mu + lambda) evolution with tournament selection; budget = number of proposed candidates (duplicates are cached)."""
    pop = [ev.evaluate(sample_feasible(space, ev, cons, rng)) for _ in range(min(pop_size, budget))]
    hist = [max(c.acc for c in pop)] * len(pop)
    while len(hist) < budget:
        kids = []
        for _ in range(min(pop_size, budget - len(hist))):
            kids.append(ev.evaluate(_child(space, ev, cons, pop, rng)))
            hist.append(max(hist[-1], kids[-1].acc))
        pop = sorted({c.arch: c for c in pop + kids}.values(), key=lambda c: -c.acc)[:pop_size]
    return pop[0], hist


# ---- multi-objective -------------------------------------------------------------------
def dominates(a: Candidate, b: Candidate) -> bool:
    return a.acc >= b.acc and a.lat <= b.lat and (a.acc > b.acc or a.lat < b.lat)


def pareto_front(cands) -> list[Candidate]:
    cands = list({c.arch: c for c in cands}.values())
    return sorted((c for c in cands if not any(dominates(o, c) for o in cands)), key=lambda c: c.lat)


def _sort_fronts(pop):
    fronts, rest = [], list(pop)
    while rest:
        f = [c for c in rest if not any(dominates(o, c) for o in rest)]
        fronts.append(f)
        rest = [c for c in rest if c not in f]
    return fronts


def _crowding(front):
    d = {c.arch: 0.0 for c in front}
    for key in (lambda c: c.acc, lambda c: c.lat):
        s = sorted(front, key=key)
        d[s[0].arch] = d[s[-1].arch] = float("inf")
        span = (key(s[-1]) - key(s[0])) or 1.0
        for i in range(1, len(s) - 1):
            d[s[i].arch] += (key(s[i + 1]) - key(s[i - 1])) / span
    return d


def nsga2(space, ev, cons, budget, rng, pop_size=40) -> tuple[list[Candidate], list[Candidate]]:
    """NSGA-II on (accuracy up, latency down). Returns (final Pareto front, all evaluated)."""
    pop = [ev.evaluate(sample_feasible(space, ev, cons, rng)) for _ in range(pop_size)]
    for _gen in range(max(50, 20 * budget // pop_size)):  # cap: cached duplicates do not consume budget
        if ev.n_acc_evals >= budget:
            break

        def tour():
            a, b = rng.choice(len(pop), 2, replace=False)
            return pop[a] if rank[pop[a].arch] < rank[pop[b].arch] or (
                rank[pop[a].arch] == rank[pop[b].arch] and crowd[pop[a].arch] > crowd[pop[b].arch]) else pop[b]
        rank, crowd = {}, {}
        for r, f in enumerate(_sort_fronts(pop)):
            crowd.update(_crowding(f))
            rank.update({c.arch: r for c in f})
        kids = []
        for _ in range(pop_size):
            for _try in range(50):
                a = tour().arch
                if rng.random() < 0.5:
                    a = space.crossover(a, tour().arch, rng)
                a = space.mutate(a, rng)
                if ev.feasible(a, cons):
                    break
            else:
                a = sample_feasible(space, ev, cons, rng)
            kids.append(ev.evaluate(a))
        merged = list({c.arch: c for c in pop + kids}.values())
        new = []
        for f in _sort_fronts(merged):
            if len(new) + len(f) <= pop_size:
                new += f
            else:
                cd = _crowding(f)
                new += sorted(f, key=lambda c: -cd[c.arch])[: pop_size - len(new)]
                break
        pop = new
    return pareto_front(pop), ev.all_evaluated()
