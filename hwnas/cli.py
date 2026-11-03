"""Command-line entry points:  python -m hwnas.cli <command> --help"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from . import (PROFILES, Constraints, Evaluator, LatencyPredictor, SearchSpace, SyntheticAccuracy,
               build_lut, evolutionary_search, nsga2, random_search, sim_latency)
from .cost import arch_metrics
from .latency import LUT, kendall_tau, mape


def _setup(a):
    space = SearchSpace()
    prof = PROFILES[a.device]
    lut = LUT.load(a.lut) if getattr(a, "lut", None) else build_lut(space, prof)
    return space, prof, lut


def _cons(a):
    return Constraints(a.latency_ms * 1e-3 if a.latency_ms else float("inf"),
                       a.sram_kb * 1024 if a.sram_kb else float("inf"),
                       a.flash_kb * 1024 if a.flash_kb else float("inf"))


def _report(space, prof, lut, c):
    m = arch_metrics(space, c.arch)
    return dict(arch=list(c.arch), ops=space.describe(c.arch), proxy_acc=round(c.acc, 3),
                lut_ms=round(lut.latency(c.arch) * 1e3, 4), sim_ms=round(sim_latency(space, c.arch, prof) * 1e3, 4),
                macs_M=round(m["macs"] / 1e6, 3), params_k=round(m["params"] / 1e3, 1),
                peak_sram_kb=round(m["peak_sram_bytes"] / 1024, 1))


def cmd_lut(a):
    space, prof, _ = _setup(a)
    build_lut(space, prof).save(a.out)
    print(f"wrote {a.out}")


def cmd_search(a):
    space, prof, lut = _setup(a)
    rng = np.random.default_rng(a.seed)
    ev = Evaluator(space, SyntheticAccuracy(space), lut.latency)
    cons = _cons(a)
    if a.algo == "nsga2":
        front, _ = nsga2(space, ev, cons, a.budget, rng)
        out = [_report(space, prof, lut, c) for c in front]
    else:
        fn = evolutionary_search if a.algo == "evo" else random_search
        best, _ = fn(space, ev, cons, a.budget, rng)
        out = _report(space, prof, lut, best)
    print(json.dumps(out, indent=1))
    if a.out:
        Path(a.out).write_text(json.dumps(out, indent=1))


def cmd_pareto(a):
    from .viz import plot_pareto
    space, prof, lut = _setup(a)
    ev = Evaluator(space, SyntheticAccuracy(space), lut.latency)
    front, allc = nsga2(space, ev, _cons(a), a.budget, np.random.default_rng(a.seed))
    plot_pareto(allc, front, a.out, f"NSGA-II on {a.device} (LUT latency)")
    print(f"front size {len(front)}; wrote {a.out}")


def cmd_devices(a):
    """Same relative latency budget on every device -> different architectures."""
    from .viz import plot_arch_grid
    space = SearchSpace()
    rng0 = np.random.default_rng(a.seed)
    sample = [space.random_arch(rng0) for _ in range(500)]
    archs = {}
    for name, prof in PROFILES.items():
        lut = build_lut(space, prof)
        T = float(np.quantile([lut.latency(x) for x in sample], a.quantile))
        ev = Evaluator(space, SyntheticAccuracy(space), lut.latency)
        best, _ = evolutionary_search(space, ev, Constraints(max_latency=T), a.budget, np.random.default_rng(a.seed))
        archs[f"{name}\n<= {T*1e3:.2f} ms"] = best.arch
        print(name, _report(space, prof, lut, best))
    plot_arch_grid(archs, space, a.out)
    print(f"wrote {a.out}")


def cmd_additivity(a):
    space, prof, lut = _setup(a)
    rng = np.random.default_rng(a.seed)
    A = [space.random_arch(rng) for _ in range(a.n)]
    true = np.array([sim_latency(space, x, prof) for x in A])
    est = np.array([lut.latency(x) for x in A])
    k = a.n // 3
    pred = LatencyPredictor(space, lut).fit(A[:k], true[:k]).predict(A[k:])
    print(f"{a.device}: LUT MAPE {mape(est, true):.3%}  Kendall tau {kendall_tau(est, true):.4f} | "
          f"ridge-corrected MAPE {mape(pred, true[k:]):.3%}")


# ---- torch-backed commands (lazy imports) ---------------------------------------------
def cmd_measure_lut(a):
    from .measure import build_measured_lut
    space = SearchSpace()
    lut = build_measured_lut(space, iters=a.iters)
    lut.save(a.out)
    print(f"wrote {a.out}")


def cmd_diff_search(a):
    import torch
    from .data import get_cifar10
    from .diff_search import differentiable_search
    space = SearchSpace()
    lut = LUT.load(a.lut) if a.lut else build_lut(space, PROFILES[a.device])
    tr, va = get_cifar10(a.batch, a.data, fake=a.fake, split_search=True)
    arch, log = differentiable_search(space, lut, tr, va, epochs=a.epochs, warmup=a.warmup,
                                      lat_loss=a.lat_loss, target=a.latency_ms * 1e-3 if a.latency_ms else None,
                                      lam=a.lam, device="cuda" if torch.cuda.is_available() else "cpu")
    print("found:", space.describe(arch), "| LUT ms:", round(lut.latency(arch) * 1e3, 3))
    Path(a.out).write_text(json.dumps(dict(arch=list(arch), log=log), indent=1))


def cmd_train(a):
    import torch
    from .data import get_cifar10
    from .train import train_arch
    space = SearchSpace()
    arch = tuple(json.loads(Path(a.arch_json).read_text())["arch"])
    tr, te = get_cifar10(a.batch, a.data, fake=a.fake, split_search=False)
    res = train_arch(space, arch, tr, te, epochs=a.epochs, device="cuda" if torch.cuda.is_available() else "cpu")
    print(json.dumps(res, indent=1))


def main(argv=None):
    p = argparse.ArgumentParser(prog="hwnas")
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(q, dev=True):
        if dev:
            q.add_argument("--device", default="rpi4", choices=list(PROFILES))
        q.add_argument("--lut", help="JSON LUT (default: simulated for --device)")
        q.add_argument("--seed", type=int, default=0)

    q = sub.add_parser("lut", help="write simulated LUT"); common(q); q.add_argument("--out", required=True); q.set_defaults(f=cmd_lut)
    q = sub.add_parser("search", help="constrained search"); common(q)
    q.add_argument("--algo", default="evo", choices=["evo", "random", "nsga2"])
    q.add_argument("--latency-ms", type=float); q.add_argument("--sram-kb", type=float); q.add_argument("--flash-kb", type=float)
    q.add_argument("--budget", type=int, default=1000); q.add_argument("--out"); q.set_defaults(f=cmd_search)
    q = sub.add_parser("pareto", help="NSGA-II Pareto plot"); common(q)
    q.add_argument("--latency-ms", type=float); q.add_argument("--sram-kb", type=float); q.add_argument("--flash-kb", type=float)
    q.add_argument("--budget", type=int, default=1500); q.add_argument("--out", default="pareto.png"); q.set_defaults(f=cmd_pareto)
    q = sub.add_parser("devices", help="same relative budget on all devices"); q.add_argument("--seed", type=int, default=0)
    q.add_argument("--quantile", type=float, default=0.1); q.add_argument("--budget", type=int, default=1500)
    q.add_argument("--out", default="devices.png"); q.set_defaults(f=cmd_devices)
    q = sub.add_parser("additivity", help="LUT additivity error + ridge correction"); common(q)
    q.add_argument("--n", type=int, default=600); q.set_defaults(f=cmd_additivity)
    q = sub.add_parser("measure-lut", help="[torch] measure a real LUT on this host CPU")
    q.add_argument("--iters", type=int, default=50); q.add_argument("--out", default="luts/host_cpu.json"); q.set_defaults(f=cmd_measure_lut)
    q = sub.add_parser("diff-search", help="[torch] differentiable (FBNet-style) search"); common(q)
    q.add_argument("--epochs", type=int, default=20); q.add_argument("--warmup", type=int, default=5)
    q.add_argument("--lat-loss", default="fbnet", choices=["fbnet", "hinge", "linear"])
    q.add_argument("--latency-ms", type=float); q.add_argument("--lam", type=float, default=1.0)
    q.add_argument("--batch", type=int, default=128); q.add_argument("--data", default="data")
    q.add_argument("--fake", action="store_true", help="random data (smoke test only)")
    q.add_argument("--out", default="found_arch.json"); q.set_defaults(f=cmd_diff_search)
    q = sub.add_parser("train", help="[torch] train a found architecture from scratch")
    q.add_argument("arch_json"); q.add_argument("--epochs", type=int, default=30); q.add_argument("--batch", type=int, default=128)
    q.add_argument("--data", default="data"); q.add_argument("--fake", action="store_true"); q.set_defaults(f=cmd_train)
    a = p.parse_args(argv)
    a.f(a)


if __name__ == "__main__":
    main()
