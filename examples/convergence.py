"""Random vs evolutionary search under the same latency budget and evaluation budget.

    python examples/convergence.py --device rpi4 --latency-ms 0.85 --budget 500 --seeds 10
"""
import argparse

import numpy as np

from hwnas import (PROFILES, Constraints, Evaluator, SearchSpace, SyntheticAccuracy, build_lut,
                   evolutionary_search, random_search)
from hwnas.viz import plot_history

p = argparse.ArgumentParser()
p.add_argument("--device", default="rpi4", choices=list(PROFILES))
p.add_argument("--latency-ms", type=float, default=0.85)
p.add_argument("--budget", type=int, default=500)
p.add_argument("--seeds", type=int, default=10)
p.add_argument("--out", default="convergence.png")
a = p.parse_args()

space, acc = SearchSpace(), None
lut = build_lut(space, PROFILES[a.device])
acc = SyntheticAccuracy(space)
cons = Constraints(max_latency=a.latency_ms * 1e-3)
H = {"random": [], "evolution": []}
for s in range(a.seeds):
    for name, fn in (("random", random_search), ("evolution", evolutionary_search)):
        _, h = fn(space, Evaluator(space, acc, lut.latency), cons, a.budget, np.random.default_rng(s))
        H[name].append(h)
for k, v in H.items():
    f = np.array([h[-1] for h in v])
    print(f"{k:10s} final best proxy-acc: {f.mean():.2f} +- {f.std():.2f} (n={a.seeds})")
plot_history({k: np.mean(v, 0) for k, v in H.items()}, a.out,
             f"{a.device}, <= {a.latency_ms} ms, mean of {a.seeds} seeds")
