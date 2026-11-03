import numpy as np
import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn

from hwnas import PROFILES, SearchSpace, arch_metrics, build_lut
from hwnas.data import get_cifar10
from hwnas.diff_search import differentiable_search, latency_loss
from hwnas.measure import build_measured_lut, measure_latency
from hwnas.proxies import synflow
from hwnas.supernet import SuperNet
from hwnas.torch_nets import build_network, make_op
from hwnas.train import train_arch

torch.set_num_threads(1)


@pytest.fixture(scope="module")
def space():
    return SearchSpace()


@pytest.fixture(scope="module")
def lut(space):
    return build_lut(space, PROFILES["rpi4"])


def count_macs(net, x):
    macs = []

    def hook(m, i, o):
        if isinstance(m, nn.Conv2d):
            macs.append(o.numel() * (m.in_channels // m.groups) * m.kernel_size[0] * m.kernel_size[1])
        elif isinstance(m, nn.Linear):
            macs.append(m.in_features * m.out_features)
    hs = [m.register_forward_hook(hook) for m in net.modules() if isinstance(m, (nn.Conv2d, nn.Linear))]
    net.eval()
    with torch.no_grad():
        net(x)
    for h in hs:
        h.remove()
    return sum(macs)


def test_params_and_macs_match_analytic_model(space):
    rng = np.random.default_rng(0)
    for _ in range(8):
        a = space.random_arch(rng)
        net = build_network(space, a)
        m = arch_metrics(space, a)
        assert sum(p.numel() for p in net.parameters()) == m["params"]
        assert count_macs(net, torch.randn(1, 3, 32, 32)) == m["macs"]


def test_forward_shape_and_block_shapes(space):
    a = space.random_arch(np.random.default_rng(1))
    assert build_network(space, a)(torch.randn(2, 3, 32, 32)).shape == (2, 10)
    for L in space.layers:
        for op in space.valid_ops[L.idx]:
            y = make_op(L, op)(torch.randn(1, L.cin, L.res_in, L.res_in))
            assert y.shape == (1, L.cout, L.res_out, L.res_out)


def test_supernet_expected_latency_matches_lut_for_onehot(space, lut):
    net = SuperNet(space, lut)
    a = space.random_arch(np.random.default_rng(2))
    ws = [torch.nn.functional.one_hot(torch.tensor(space.valid_ops[l].index(x)), len(space.valid_ops[l])).float()
          for l, x in enumerate(a)]
    assert float(net.expected_latency(ws)) == pytest.approx(lut.latency(a), rel=1e-5)


def test_supernet_forward_and_derive(space, lut):
    net = SuperNet(space, lut)
    logits, ws = net(torch.randn(2, 3, 32, 32), tau=2.0)
    assert logits.shape == (2, 10) and len(ws) == space.n_layers
    assert space.is_valid(net.derive())
    assert len(net.arch_parameters()) == space.n_layers
    assert not set(map(id, net.arch_parameters())) & set(map(id, net.weight_parameters()))


def test_latency_gradient_prefers_cheap_ops(space, lut):
    net = SuperNet(space, lut)
    ws = [torch.softmax(t, 0) for t in net.thetas]
    lat = net.expected_latency(ws)
    latency_loss(torch.tensor(1.0), lat, "fbnet").backward()
    for l, t in enumerate(net.thetas):  # gradient descent must move mass toward cheaper ops
        costs = getattr(net, f"lut_{l}")
        assert (t.grad * (costs - costs.mean())).sum() > 0


def test_hinge_loss_only_active_over_budget(space, lut):
    t = torch.tensor(1e-3)
    assert float(latency_loss(torch.tensor(1.0), torch.tensor(5e-4), "hinge", lam=2, target=t)) == pytest.approx(1.0)
    assert float(latency_loss(torch.tensor(1.0), torch.tensor(2e-3), "hinge", lam=2, target=t)) == pytest.approx(3.0)


def test_differentiable_search_smoke(space, lut):
    a, b = get_cifar10(batch=16, fake=True, split_search=True, n_fake=64)
    arch, log = differentiable_search(space, lut, a, b, epochs=2, warmup=1, verbose=False)
    assert space.is_valid(arch) and len(log) == 2 and log[1]["expected_lat_ms"] > 0


def test_train_arch_smoke(space):
    tr, te = get_cifar10(batch=16, fake=True, split_search=False, n_fake=64)
    r = train_arch(space, tuple(1 for _ in space.layers), tr, te, epochs=1)
    assert 0.0 <= r["test_acc"] <= 1.0 and r["host_cpu_ms"] > 0


def test_measure_lut(space):
    lut = build_measured_lut(space, iters=2, warmup=1)
    assert lut.base > 0 and np.nanmin(lut.table) >= 0
    assert (lut.table[:, 0][[L.can_skip for L in space.layers]] == 0).all()
    assert lut.meta["source"] == "torch-measured"
    assert measure_latency(build_network(space, tuple(1 for _ in space.layers)), (1, 3, 32, 32), 1, 2) > 0


def test_synflow_finite_and_grows_with_capacity(space):
    small = synflow(build_network(space, tuple(1 for _ in space.layers)), (1, 3, 32, 32))
    big = synflow(build_network(space, tuple(5 for _ in space.layers)), (1, 3, 32, 32))
    assert np.isfinite(small) and np.isfinite(big) and big > small
