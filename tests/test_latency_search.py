import numpy as np
import pytest

from hwnas import (PROFILES, Constraints, Evaluator, LatencyPredictor, SearchSpace, SyntheticAccuracy,
                   build_lut, evolutionary_search, mnas_reward, nsga2, pareto_front, random_search, sim_latency)
from hwnas.latency import LUT, kendall_tau, mape
from hwnas.search import dominates


@pytest.fixture(scope="module")
def setup():
    sp = SearchSpace()
    return sp, build_lut(sp, PROFILES["rpi4"]), SyntheticAccuracy(sp)


def test_lut_save_load_roundtrip(setup, tmp_path):
    sp, lut, _ = setup
    lut.save(tmp_path / "l.json")
    l2 = LUT.load(tmp_path / "l.json")
    a = sp.random_arch(np.random.default_rng(0))
    assert lut.latency(a) == pytest.approx(l2.latency(a))


def test_lut_additivity_error_depends_on_device():
    sp = SearchSpace()
    A = [sp.random_arch(np.random.default_rng(s)) for s in range(200)]
    err = {}
    for n, prof in PROFILES.items():
        lut = build_lut(sp, prof)
        err[n] = mape([lut.latency(a) for a in A], [sim_latency(sp, a, prof) for a in A])
    assert err["mcu"] < 1e-6 < err["rpi4"] < err["edge_gpu"]


def test_predictor_corrects_lut_bias():
    sp = SearchSpace(); prof = PROFILES["edge_gpu"]; lut = build_lut(sp, prof)
    rng = np.random.default_rng(0)
    A = [sp.random_arch(rng) for _ in range(400)]
    y = np.array([sim_latency(sp, a, prof) for a in A])
    raw = mape([lut.latency(a) for a in A[200:]], y[200:])
    pred = LatencyPredictor(sp, lut).fit(A[:200], y[:200]).predict(A[200:])
    assert mape(pred, y[200:]) < raw / 10
    assert kendall_tau(pred, y[200:]) > 0.99


def test_synthetic_accuracy_deterministic_and_monotone(setup):
    sp, _, acc = setup
    a = sp.random_arch(np.random.default_rng(0))
    assert acc(a) == acc(a)
    assert acc(tuple(5 for _ in sp.layers)) > acc(tuple(1 for _ in sp.layers))


def test_constraints_respected(setup):
    sp, lut, acc = setup
    cons = Constraints(max_latency=0.9e-3, max_sram=100 * 1024)
    ev = Evaluator(sp, acc, lut.latency)
    best, _ = evolutionary_search(sp, ev, cons, 300, np.random.default_rng(0))
    assert best.lat <= cons.max_latency and best.sram <= cons.max_sram
    assert all(ev.feasible(c.arch, cons) for c in ev.all_evaluated())


def test_infeasible_raises(setup):
    sp, lut, acc = setup
    with pytest.raises(RuntimeError):
        random_search(sp, Evaluator(sp, acc, lut.latency), Constraints(max_latency=1e-6), 5, np.random.default_rng(0))


def test_evolution_beats_random_on_average(setup):
    sp, lut, acc = setup
    cons = Constraints(max_latency=0.85e-3)
    r, e = [], []
    for s in range(4):
        r.append(random_search(sp, Evaluator(sp, acc, lut.latency), cons, 400, np.random.default_rng(s))[0].acc)
        e.append(evolutionary_search(sp, Evaluator(sp, acc, lut.latency), cons, 400, np.random.default_rng(s))[0].acc)
    assert np.mean(e) > np.mean(r)


def test_tighter_budget_never_better(setup):
    sp, lut, acc = setup
    res = []
    for T in (0.7e-3, 0.9e-3, 1.3e-3):
        ev = Evaluator(sp, acc, lut.latency)
        res.append(evolutionary_search(sp, ev, Constraints(max_latency=T), 400, np.random.default_rng(0))[0].acc)
    assert res[0] <= res[1] + 0.3 and res[1] <= res[2] + 0.3  # allow search noise


def test_nsga2_front_is_nondominated(setup):
    sp, lut, acc = setup
    ev = Evaluator(sp, acc, lut.latency)
    front, allc = nsga2(sp, ev, Constraints(), 300, np.random.default_rng(0), pop_size=24)
    assert len(front) >= 5
    assert not any(dominates(a, b) for a in front for b in front if a is not b)
    assert all(not dominates(c, f) for c in allc for f in front)
    assert [c.lat for c in front] == sorted(c.lat for c in front)


def test_pareto_front_simple():
    from hwnas.search import Candidate
    pts = [(80, 2), (79, 1), (78, 3), (81, 4), (79, 2)]
    f = pareto_front([Candidate((i,), a, l, 0, 0, 0) for i, (a, l) in enumerate(pts)])
    assert [(c.acc, c.lat) for c in f] == [(79, 1), (80, 2), (81, 4)]


def test_mnas_reward():
    assert mnas_reward(75, 10e-3, 10e-3) == pytest.approx(75)
    assert mnas_reward(75, 20e-3, 10e-3) < 75 < mnas_reward(75, 5e-3, 10e-3)
    assert mnas_reward(75, 20e-3, 10e-3, w=-1) == pytest.approx(37.5)
